import cupy as cp
from numba import cuda
import numpy as np
import math
import os
import time  # Para medir el tiempo de la simulación

# ===============================================
# --- 1. KERNELS DE SIMULACIÓN EN CUDA ---
# ===============================================

@cuda.jit
def initialize_particles_kernel(pos, vel, force, pressure, density, N, dim, epsilon, n_per_dim):
    """Inicializa las partículas en una grilla."""
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, N, stride):
        pressure[i] = 0.0
        density[i] = 0.0
        for d in range(dim):
            vel[i, d] = 0.0
            force[i, d] = 0.0
        temp_i = i
        for d in range(dim):
            idx = temp_i % n_per_dim
            pos[i, d] = idx * epsilon
            temp_i = temp_i // n_per_dim

@cuda.jit
def apply_gravity_kernel(force, g_vector, N, dim):
    """Suma la gravedad al vector de aceleración (fuerza)."""
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        for d in range(dim):
            force[i, d] += g_vector[d]

@cuda.jit
def integrate_kernel(pos, vel, force, dt, N, dim):
    """Actualiza posición y velocidad (Integrador de Euler)."""
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        for d in range(dim):
            vel[i, d] += force[i, d] * dt
        for d in range(dim):
            pos[i, d] += vel[i, d] * dt
        for d in range(dim):
            force[i, d] = 0.0

# --- <MODIFIED> Kernel de Colisiones (Bounding Box) ---
@cuda.jit
def check_boundaries_kernel(pos, vel, N, damping, bound_max):
    """
    Revisa colisiones con una caja delimitadora (Bounding Box).
    Límites de la caja: [0.0, bound_max] en los 3 ejes.
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    bound_min = 0.0 # Límite inferior (suelo y paredes)

    for i in range(start, N, stride):
        
        # Iteramos sobre las 3 dimensiones (d=0 es X, d=1 es Y, d=2 es Z)
        for d in range(3): 
            
            # 1. Colisión con el límite inferior (0.0)
            if pos[i, d] < bound_min:
                pos[i, d] = bound_min
                vel[i, d] = -vel[i, d] * damping
            
            # 2. Colisión con el límite superior (bound_max)
            elif pos[i, d] > bound_max:
                pos[i, d] = bound_max
                vel[i, d] = -vel[i, d] * damping

# --- <NEW> Funciones SPH (Device Functions) ---
@cuda.jit(device=True)
def poly6_W(r_sq, H, H_sq):
    if r_sq >= H_sq:
        return 0.0
    factor = H_sq - r_sq
    return factor * factor * factor 

@cuda.jit(device=True)
def spiky_grad_W(r_vec, r_len, H, H_sq):
    if r_len > H or r_len == 0.0:
        r_vec[0] = 0.0
        r_vec[1] = 0.0
        r_vec[2] = 0.0
        return
    factor = H - r_len
    scale = SPIKY_COEFF * factor * factor / r_len
    r_vec[0] *= scale
    r_vec[1] *= scale
    r_vec[2] *= scale

@cuda.jit(device=True)
def viscosity_lap_W(r_len, H):
    if r_len > H:
        return 0.0
    return VISC_COEFF * (H - r_len)

# --- <NEW> Kernels Principales de SPH ---
@cuda.jit
def compute_density_pressure_kernel(pos, density, pressure, N, H, H_sq, MASS, REST_DENSITY, GAS_CONST):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        density[i] = MASS * POLY6_COEFF * (H**9) 
        for j in range(N):
            r_x = pos[i, 0] - pos[j, 0]
            r_y = pos[i, 1] - pos[j, 1]
            r_z = pos[i, 2] - pos[j, 2]
            r_sq = r_x*r_x + r_y*r_y + r_z*r_z
            if r_sq == 0.0 or r_sq >= H_sq:
                continue
            density[i] += MASS * POLY6_COEFF * poly6_W(r_sq, H, H_sq)
        pressure[i] = GAS_CONST * (density[i] - REST_DENSITY)
        if pressure[i] < 0.0:
            pressure[i] = 0.0

@cuda.jit
def compute_forces_kernel(pos, vel, force, density, pressure, N, H, H_sq, MASS, VISCOSITY):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    grad_vec = cuda.local.array(3, dtype=np.float32) # Corregido a np.float32
    vel_diff = cuda.local.array(3, dtype=np.float32) # Corregido a np.float32

    for i in range(start, N, stride):
        for j in range(N):
            if i == j:
                continue
            r_x = pos[i, 0] - pos[j, 0]
            r_y = pos[i, 1] - pos[j, 1]
            r_z = pos[i, 2] - pos[j, 2]
            r_sq = r_x*r_x + r_y*r_y + r_z*r_z
            if r_sq >= H_sq:
                continue
            r_len = math.sqrt(r_sq)
            if density[j] <= 0.0:
                continue

            # --- Fuerza de Presión ---
            pressure_avg = 0.5 * (pressure[i] + pressure[j])
            grad_vec[0] = r_x
            grad_vec[1] = r_y
            grad_vec[2] = r_z
            spiky_grad_W(grad_vec, r_len, H, H_sq) 
            scale = -MASS * (pressure_avg / density[j])
            force[i, 0] += grad_vec[0] * scale
            force[i, 1] += grad_vec[1] * scale
            force[i, 2] += grad_vec[2] * scale

            # --- Fuerza de Viscosidad ---
            vel_diff[0] = vel[j, 0] - vel[i, 0]
            vel_diff[1] = vel[j, 1] - vel[i, 1]
            vel_diff[2] = vel[j, 2] - vel[i, 2]
            lap_W = viscosity_lap_W(r_len, H)
            scale = VISCOSITY * MASS * (MASS / density[j]) * lap_W
            force[i, 0] += vel_diff[0] * scale
            force[i, 1] += vel_diff[1] * scale
            force[i, 2] += vel_diff[2] * scale

# ===============================================
# --- 2. CONFIGURACIÓN DE LA SIMULACIÓN ---
# ===============================================

# --- Parámetros de Partículas ---
N = 100000       
dim = 3        
epsilon = 0.1  

# --- Parámetros de Simulación ---
dt = 0.005      
FRAMES = 200     
damping = 0.8    
BOUND_BOX_SIZE = 5.0 

# --- Parámetros de SPH (Fluido) ---
H = 0.1             
MASS = 100          
REST_DENSITY = 1000.0 
GAS_CONST = 10.0
VISCOSITY = 0.5    

# --- Constantes pre-calculadas para los kernels SPH ---
POLY6_COEFF = 315.0 / (64.0 * math.pi * H**9)
SPIKY_COEFF = -45.0 / (math.pi * H**6)
VISC_COEFF = 45.0 / (math.pi * H**6)

# --- Configuración (Host) ---
n_per_dim = int(math.ceil(N**(1.0 / dim)))
print(f"Total de partículas N = {N}, dim = {dim}")
print(f"Grilla calculada de {n_per_dim}^({dim})")

# --- Reservar memoria en GPU (CuPy) ---
d_pos = cp.empty((N, dim), dtype=cp.float32)
d_vel = cp.empty((N, dim), dtype=cp.float32)
d_force = cp.empty((N, dim), dtype=cp.float32)
d_pressure = cp.empty(N, dtype=cp.float32)
d_density = cp.empty(N, dtype=cp.float32)

# --- Vector de Gravedad (en GPU) ---
gravity_cpu = np.zeros(dim, dtype=np.float32)
gravity_cpu[2] = -9.81  
d_gravity = cp.asarray(gravity_cpu)
print(f"Vector de gravedad: {gravity_cpu}")

# --- Configuración del Kernel de CUDA ---
threads_per_block = 256
blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block

# --- 3. INICIALIZAR PARTÍCULAS ---
initialize_particles_kernel[blocks_per_grid, threads_per_block](
    d_pos, d_vel, d_force, d_pressure, d_density,
    N, dim, epsilon, n_per_dim
)
cuda.synchronize()
print("Partículas inicializadas en la GPU.")

# ===============================================
# --- 4. BUCLE DE SIMULACIÓN Y CAPTURA ---
# ===============================================

print(f"Iniciando simulación de {FRAMES} pasos...")
print("ADVERTENCIA: Usando O(N^2), esto puede ser muy lento.")

# ... (código de directorio y listas) ...
os.makedirs("results", exist_ok=True)
output_filename = "results/simulation_data.npz"
all_positions_list = []
all_positions_list.append(d_pos.get())

start_time = time.time()
H_sq = H * H

# --- Bucle de Simulación ---
for frame_idx in range(1, FRAMES):
    
    # 1. Aplicar gravedad
    apply_gravity_kernel[blocks_per_grid, threads_per_block](
        d_force, d_gravity, N, dim
    )
    
    # 2. Calcular Densidad y Presión
    compute_density_pressure_kernel[blocks_per_grid, threads_per_block](
        d_pos, d_density, d_pressure, 
        N, H, H_sq, MASS, REST_DENSITY, GAS_CONST
    )

    # 3. Calcular Fuerzas SPH
    compute_forces_kernel[blocks_per_grid, threads_per_block](
        d_pos, d_vel, d_force, d_density, d_pressure,
        N, H, H_sq, MASS, VISCOSITY
    )

    # 4. Integrar
    integrate_kernel[blocks_per_grid, threads_per_block](
        d_pos, d_vel, d_force, dt, N, dim
    )
    
    # 5. Revisar colisiones (¡Llamada actualizada!)
    check_boundaries_kernel[blocks_per_grid, threads_per_block](
        d_pos, d_vel, N, damping, BOUND_BOX_SIZE # <--- CAMBIO AQUÍ
    )

    cuda.synchronize()

    # --- B. Copiar datos para Visualización (GPU -> CPU) ---
    pos_np = d_pos.get()
    all_positions_list.append(pos_np)

    if frame_idx % 20 == 0:
        print(f"Calculando paso {frame_idx}/{FRAMES}")

end_time = time.time()
print(f"Simulación (solo cómputo) completada en {end_time - start_time:.4f} segundos.")

# ===============================================
# --- 5. GUARDAR DATOS EN ARCHIVO ---
# ===============================================
print("Guardando datos en archivo...")

final_data_array = np.array(all_positions_list, dtype=np.float32)
print(f"Dimensiones del array final: {final_data_array.shape}")

np.savez_compressed(
    output_filename,
    positions=final_data_array,
    num_particles=N,
    num_frames=FRAMES,
    dt=dt,
    h=H
)

print("¡Datos guardados!")
print(f"Ruta del archivo: {output_filename}")
