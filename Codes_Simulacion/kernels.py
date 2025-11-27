from numba import cuda
import math
import numpy as np

# ===============================================
# --- 1. KERNELS UTILITARIOS (GRID) ---
# ===============================================

@cuda.jit
def calculate_grid_indices_kernel(pos, grid_indices, N, cell_size, grid_dim_x, grid_dim_y, grid_dim_z):
    """Calcula el índice lineal de la celda para cada partícula."""
    i = cuda.grid(1)
    if i < N:
        x_idx = int(pos[i, 0] / cell_size)
        y_idx = int(pos[i, 1] / cell_size)
        z_idx = int(pos[i, 2] / cell_size)

        x_idx = max(0, min(x_idx, grid_dim_x - 1))
        y_idx = max(0, min(y_idx, grid_dim_y - 1))
        z_idx = max(0, min(z_idx, grid_dim_z - 1))

        grid_indices[i] = x_idx + y_idx * grid_dim_x + z_idx * grid_dim_x * grid_dim_y

@cuda.jit
def find_cell_bounds_kernel(grid_indices, cell_start, cell_end, N):
    i = cuda.grid(1)
    if i < N:
        hash_idx = grid_indices[i]
        if i == 0:
            cell_start[hash_idx] = i
        elif hash_idx != grid_indices[i - 1]:
            cell_start[hash_idx] = i
            cell_end[grid_indices[i - 1]] = i
        if i == N - 1:
            cell_end[hash_idx] = N

# ===============================================
# --- 2. FUNCIONES DEVICE SPH (Matemática) ---
# ===============================================

@cuda.jit(device=True)
def poly6_W(r_sq, H, H_sq):
    if r_sq >= H_sq: return 0.0
    factor = H_sq - r_sq
    return factor * factor * factor

@cuda.jit(device=True)
def spiky_grad_W(r_vec, r_len, H, H_sq, SPIKY_COEFF):
    factor = H - r_len
    scale = SPIKY_COEFF * factor * factor / r_len
    r_vec[0] *= scale
    r_vec[1] *= scale
    r_vec[2] *= scale

@cuda.jit(device=True)
def viscosity_lap_W(r_len, H, VISC_COEFF):
    return VISC_COEFF * (H - r_len)

# ===============================================
# --- 3. KERNELS DE SIMULACIÓN PRINCIPALES ---
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
        temp_i = i
        pos[i, 0] = (temp_i % n_per_dim) * epsilon + 1.0
        temp_i //= n_per_dim
        pos[i, 1] = (temp_i % n_per_dim) * epsilon + 1.0
        temp_i //= n_per_dim
        pos[i, 2] = (temp_i % n_per_dim) * epsilon + 1.0

@cuda.jit
def compute_density_pressure_grid(pos, density, pressure, cell_start, cell_end,
                                  grid_dim_x, grid_dim_y, grid_dim_z, cell_size,
                                  N, H, H_sq, MASS, REST_DENSITY, GAS_CONST, POLY6_COEFF):
    i = cuda.grid(1)
    if i >= N: return

    p_pos_x = pos[i, 0]
    p_pos_y = pos[i, 1]
    p_pos_z = pos[i, 2]
    cx = int(p_pos_x / cell_size)
    cy = int(p_pos_y / cell_size)
    cz = int(p_pos_z / cell_size)

    dens = MASS * POLY6_COEFF * (H**9)

    for z in range(cz - 1, cz + 2):
        for y in range(cy - 1, cy + 2):
            for x in range(cx - 1, cx + 2):
                if x >= 0 and x < grid_dim_x and y >= 0 and y < grid_dim_y and z >= 0 and z < grid_dim_z:
                    cell_idx = x + y * grid_dim_x + z * grid_dim_x * grid_dim_y
                    start_idx = cell_start[cell_idx]
                    end_idx = cell_end[cell_idx]
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
    f_x = 0.0
    f_y = 0.0
    f_z = 0.0
    grad_vec = cuda.
