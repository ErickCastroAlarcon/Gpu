import cupy as cp
from numba import cuda
import numpy as np
import math
import os
import time

# ===============================================
# --- 1. KERNELS UTILITARIOS (GRID) ---
# ===============================================

@cuda.jit
def calculate_grid_indices_kernel(pos, grid_indices, N, cell_size, grid_dim_x, grid_dim_y, grid_dim_z):
    """Calcula el índice lineal de la celda para cada partícula."""
    i = cuda.grid(1)
    if i < N:
        # Calcular coordenadas de la grilla
        x_idx = int(pos[i, 0] / cell_size)
        y_idx = int(pos[i, 1] / cell_size)
        z_idx = int(pos[i, 2] / cell_size)

        # Clampear para evitar salir de la grilla (seguridad)
        x_idx = max(0, min(x_idx, grid_dim_x - 1))
        y_idx = max(0, min(y_idx, grid_dim_y - 1))
        z_idx = max(0, min(z_idx, grid_dim_z - 1))

        grid_indices[i] = x_idx + y_idx * grid_dim_x + z_idx * grid_dim_x * grid_dim_y

@cuda.jit
def find_cell_bounds_kernel(grid_indices, cell_start, cell_end, N):
    """Determina donde empieza y termina cada celda en el array ordenado."""
    i = cuda.grid(1)
    if i < N:
        hash_idx = grid_indices[i]
        
        # Si soy el primero con este hash, marco el inicio
        if i == 0:
            cell_start[hash_idx] = i
        elif hash_idx != grid_indices[i - 1]:
            cell_start[hash_idx] = i
            cell_end[grid_indices[i - 1]] = i # El anterior termina aquí
        
        # Si soy el último elemento absoluto
        if i == N - 1:
            cell_end[hash_idx] = N

# ===============================================
# --- 2. KERNELS DE SIMULACIÓN (MODIFICADOS) ---
# ===============================================

@cuda.jit
def initialize_particles_kernel(pos, vel, force, pressure, density, N, dim, epsilon, n_per_dim):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        pressure[i] = 0.0
        density[i] = 0.0
        for d in range(dim):
            vel[i, d] = 0.0
            force[i, d] = 0.0
        # Inicialización simple en cubo
        temp_i = i
        pos[i, 0] = (temp_i % n_per_dim) * epsilon + 1.0
        temp_i //= n_per_dim
        pos[i, 1] = (temp_i % n_per_dim) * epsilon + 1.0
        temp_i //= n_per_dim
        pos[i, 2] = (temp_i % n_per_dim) * epsilon + 1.0

@cuda.jit
def apply_gravity_kernel(force, g_vector, N, dim):
    i = cuda.grid(1)
    if i < N:
        for d in range(dim):
            force[i, d] += g_vector[d]

@cuda.jit
def integrate_kernel(pos, vel, force, dt, N, dim):
    i = cuda.grid(1)
    if i < N:
        for d in range(dim):
            vel[i, d] += force[i, d] * dt
            pos[i, d] += vel[i, d] * dt
            force[i, d] = 0.0

@cuda.jit
def check_boundaries_kernel(pos, vel, N, damping, bound_max):
    i = cuda.grid(1)
    if i < N:
        for d in range(3): 
            if pos[i, d] < 0.0:
                pos[i, d] = 0.0
                vel[i, d] = -vel[i, d] * damping
            elif pos[i, d] > bound_max:
                pos[i, d] = bound_max
                vel[i, d] = -vel[i, d] * damping

# --- FUNCIONES DEVICE SPH  ---
@cuda.jit(device=True)
def poly6_W(r_sq, H, H_sq):
    if r_sq >= H_sq: return 0.0
    factor = H_sq - r_sq
    return factor * factor * factor 

@cuda.jit(device=True)
def spiky_grad_W(r_vec, r_len, H, H_sq, SPIKY_COEFF): # Pasamos COEFF como arg
    factor = H - r_len
    scale = SPIKY_COEFF * factor * factor / r_len
    r_vec[0] *= scale
    r_vec[1] *= scale
    r_vec[2] *= scale

@cuda.jit(device=True)
def viscosity_lap_W(r_len, H, VISC_COEFF): # Pasamos COEFF como arg
    return VISC_COEFF * (H - r_len)

# --- KERNELS SPH CON BÚSQUEDA DE VECINOS <MODIFICADO> ---

@cuda.jit
def compute_density_pressure_grid(pos, density, pressure, cell_start, cell_end, 
                                  grid_dim_x, grid_dim_y, grid_dim_z, cell_size,
                                  N, H, H_sq, MASS, REST_DENSITY, GAS_CONST, POLY6_COEFF):
    """Calcula densidad usando la grilla de vecinos."""
    i = cuda.grid(1)
    if i >= N: return

    # Mi posición y celda
    p_pos_x = pos[i, 0]
    p_pos_y = pos[i, 1]
    p_pos_z = pos[i, 2]

    # Calcular mi celda actual
    cx = int(p_pos_x / cell_size)
    cy = int(p_pos_y / cell_size)
    cz = int(p_pos_z / cell_size)

    # Densidad propia
    dens = MASS * POLY6_COEFF * (H**9)

    # Bucle sobre celdas vecinas (3x3x3)
    for z in range(cz - 1, cz + 2):
        for y in range(cy - 1, cy + 2):
            for x in range(cx - 1, cx + 2):
                # Verificar límites de la grilla
                if x >= 0 and x < grid_dim_x and y >= 0 and y < grid_dim_y and z >= 0 and z < grid_dim_z:
                    # Obtener hash de la celda vecina
                    cell_idx = x + y * grid_dim_x + z * grid_dim_x * grid_dim_y
                    
                    start_idx = cell_start[cell_idx]
                    end_idx = cell_end[cell_idx]

                    # Iterar sobre partículas en esa celda vecina
                    if start_idx != -1:
                        for j in range(start_idx, end_idx):
                            r_x = p_pos_x - pos[j, 0]
                            r_y = p_pos_y - pos[j, 1]
                            r_z = p_pos_z - pos[j, 2]
                            r_sq = r_x*r_x + r_y*r_y + r_z*r_z
                            
                            if r_sq > 0.0 and r_sq < H_sq:
                                dens += MASS * POLY6_COEFF * poly6_W(r_sq, H, H_sq)
    
    density[i] = dens
    p = GAS_CONST * (dens - REST_DENSITY)
    pressure[i] = p if p > 0.0 else 0.0

@cuda.jit
def compute_forces_grid(pos, vel, force, density, pressure, cell_start, cell_end,
                        grid_dim_x, grid_dim_y, grid_dim_z, cell_size,
                        N, H, H_sq, MASS, VISCOSITY, SPIKY_COEFF, VISC_COEFF):
    """Calcula fuerzas usando la grilla de vecinos."""
    i = cuda.grid(1)
    if i >= N: return

    p_pos_x = pos[i, 0]
    p_pos_y = pos[i, 1]
    p_pos_z = pos[i, 2]
    p_vel_x = vel[i, 0]
    p_vel_y = vel[i, 1]
    p_vel_z = vel[i, 2]
    p_dens = density[i]
    p_press = pressure[i]

    # Variables locales para acumular fuerza
    f_x = 0.0
    f_y = 0.0
    f_z = 0.0

    # Auxiliares
    grad_vec = cuda.local.array(3, dtype=np.float32) 
    
    cx = int(p_pos_x / cell_size)
    cy = int(p_pos_y / cell_size)
    cz = int(p_pos_z / cell_size)

    # Bucle sobre celdas vecinas
    for z in range(cz - 1, cz + 2):
        for y in range(cy - 1, cy + 2):
            for x in range(cx - 1, cx + 2):
                if x >= 0 and x < grid_dim_x and y >= 0 and y < grid_dim_y and z >= 0 and z < grid_dim_z:
                    cell_idx = x + y * grid_dim_x + z * grid_dim_x * grid_dim_y
                    start_idx = cell_start[cell_idx]
                    end_idx = cell_end[cell_idx]

                    if start_idx != -1:
                        for j in range(start_idx, end_idx):
                            if i == j: continue

                            r_x = p_pos_x - pos[j, 0]
                            r_y = p_pos_y - pos[j, 1]
                            r_z = p_pos_z - pos[j, 2]
                            r_sq = r_x*r_x + r_y*r_y + r_z*r_z

                            if r_sq < H_sq:
                                r_len = math.sqrt(r_sq)
                                j_dens = density[j]
                                if j_dens <= 0.0: continue

                                # Presión
                                pressure_avg = 0.5 * (p_press + pressure[j])
                                grad_vec[0] = r_x
                                grad_vec[1] = r_y
                                grad_vec[2] = r_z
                                spiky_grad_W(grad_vec, r_len, H, H_sq, SPIKY_COEFF)
                                
                                scale_p = -MASS * (pressure_avg / j_dens)
                                f_x += grad_vec[0] * scale_p
                                f_y += grad_vec[1] * scale_p
                                f_z += grad_vec[2] * scale_p

                                # Viscosidad
                                v_diff_x = vel[j, 0] - p_vel_x
                                v_diff_y = vel[j, 1] - p_vel_y
                                v_diff_z = vel[j, 2] - p_vel_z
                                
                                lap_W = viscosity_lap_W(r_len, H, VISC_COEFF)
                                scale_v = VISCOSITY * MASS * (MASS / j_dens) * lap_W
                                
                                f_x += v_diff_x * scale_v
                                f_y += v_diff_y * scale_v
                                f_z += v_diff_z * scale_v

    force[i, 0] += f_x
    force[i, 1] += f_y
    force[i, 2] += f_z

# ===============================================
# --- 3. SETUP Y PARÁMETROS ---
# ===============================================

N = 200000       
dim = 3        
epsilon = 0.02  # Espaciado entre particulas

dt = 0.0002  
FRAMES = 500     
damping = 0.5    
BOUND_BOX_SIZE = 6.0 

# SPH Params
H = 0.04
REST_DENSITY = 1000.0             
MASS = 0.1    
GAS_CONST = 1000.0    # Constante de gas más rígida
VISCOSITY = 0.3      

# Coeficientes (calculados en CPU, pasados como args al kernel)
POLY6_COEFF = 315.0 / (64.0 * math.pi * H**9)
SPIKY_COEFF = -45.0 / (math.pi * H**6)
VISC_COEFF = 45.0 / (math.pi * H**6)

# --- Configuración Grilla <NUEVO> ---
CELL_SIZE = H
GRID_DIM_X = int(math.ceil(BOUND_BOX_SIZE / CELL_SIZE))
GRID_DIM_Y = int(math.ceil(BOUND_BOX_SIZE / CELL_SIZE))
GRID_DIM_Z = int(math.ceil(BOUND_BOX_SIZE / CELL_SIZE))
NUM_CELLS = GRID_DIM_X * GRID_DIM_Y * GRID_DIM_Z

print(f"N={N}. Grilla: {GRID_DIM_X}x{GRID_DIM_Y}x{GRID_DIM_Z} ({NUM_CELLS} celdas).")

# --- Memoria GPU ---
d_pos = cp.empty((N, dim), dtype=cp.float32)
d_vel = cp.empty((N, dim), dtype=cp.float32)
d_force = cp.empty((N, dim), dtype=cp.float32)
d_pressure = cp.empty(N, dtype=cp.float32)
d_density = cp.empty(N, dtype=cp.float32)

# Arrays para la búsqueda de vecinos
d_grid_indices = cp.empty(N, dtype=cp.int32)
d_cell_start = cp.full(NUM_CELLS, -1, dtype=cp.int32)
d_cell_end = cp.full(NUM_CELLS, -1, dtype=cp.int32)

gravity_cpu = np.array([0.0, -9.81, 0.0], dtype=np.float32)
d_gravity = cp.asarray(gravity_cpu)

# Config kernel
threads_per_block = 256
blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block
num_blocks_cells = (NUM_CELLS + (threads_per_block - 1)) // threads_per_block

# Inicializar
n_per_dim = int(math.ceil(N**(1.0 / dim)))
initialize_particles_kernel[blocks_per_grid, threads_per_block](
    d_pos, d_vel, d_force, d_pressure, d_density,
    N, dim, epsilon, n_per_dim
)
cuda.synchronize()

# ===============================================
# --- 4. BUCLE PRINCIPAL ---
# ===============================================

os.makedirs("results", exist_ok=True)
all_positions_list = []

start_time = time.time()
H_sq = H * H

print(f"Iniciando simulación Grid SPH ({FRAMES} pasos)...")

for frame_idx in range(FRAMES):

    # --- A. Actualizar Grilla Espacial (Sorting) ---
    # 1. Calcular indices de grilla
    calculate_grid_indices_kernel[blocks_per_grid, threads_per_block](
        d_pos, d_grid_indices, N, CELL_SIZE, GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z
    )
    
    # 2. Ordenar partículas según su índice de grilla (Crucial para acceso a memoria)
    sorted_indices = cp.argsort(d_grid_indices)
    
    # Reordenamos los arrays de datos para que estén contiguos en memoria
    d_pos = d_pos[sorted_indices]
    d_vel = d_vel[sorted_indices]
    d_pressure = d_pressure[sorted_indices]
    d_density = d_density[sorted_indices]
    d_grid_indices = d_grid_indices[sorted_indices] # Necesario ordenar el hash también

    # 3. Encontrar inicio y fin de celdas
    d_cell_start.fill(-1) # Resetear
    d_cell_end.fill(-1)
    
    find_cell_bounds_kernel[blocks_per_grid, threads_per_block](
        d_grid_indices, d_cell_start, d_cell_end, N
    )
    
    # --- B. Física SPH ---
    
    # 1. Densidad y Presión
    compute_density_pressure_grid[blocks_per_grid, threads_per_block](
        d_pos, d_density, d_pressure, d_cell_start, d_cell_end,
        GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z, CELL_SIZE,
        N, H, H_sq, MASS, REST_DENSITY, GAS_CONST, POLY6_COEFF
    )
    
    # 2. Fuerzas
    # Limpiamos fuerza vieja (integrada en el paso anterior)
    d_force.fill(0.0) 
    apply_gravity_kernel[blocks_per_grid, threads_per_block](d_force, d_gravity, N, dim)
    
    compute_forces_grid[blocks_per_grid, threads_per_block](
        d_pos, d_vel, d_force, d_density, d_pressure, d_cell_start, d_cell_end,
        GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z, CELL_SIZE,
        N, H, H_sq, MASS, VISCOSITY, SPIKY_COEFF, VISC_COEFF
    )

    # 3. Integración
    integrate_kernel[blocks_per_grid, threads_per_block](
        d_pos, d_vel, d_force, dt, N, dim
    )

    # 4. Colisiones
    check_boundaries_kernel[blocks_per_grid, threads_per_block](
        d_pos, d_vel, N, damping, BOUND_BOX_SIZE
    )

    cuda.synchronize()

    # Guardar frames
    if frame_idx % 2 == 0: # Guardar cada 2 frames reduce cuello de botella en la transferencia de datos
        all_positions_list.append(d_pos.get())
    
    if frame_idx % 50 == 0:
        print(f"Frame {frame_idx}/{FRAMES} completado.")

end_time = time.time()
print(f"Simulación completada en {end_time - start_time:.4f} s.")

# Guardar
final_data = np.array(all_positions_list, dtype=np.float32)
np.savez_compressed("results/sph_grid_opt.npz", positions=final_data)
print("Guardado.")
