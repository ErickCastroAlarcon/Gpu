# main.py
import cupy as cp
import numpy as np
from numba import cuda
import math
import os
import time

# IMPORTAMOS NUESTROS KERNELS
import kernels as k  # Asumimos que el archivo anterior se llama kernels.py

def run_simulation():
    # --- PARÁMETROS ---
    N = 200000
    dim = 3
    epsilon = 0.02
    dt = 0.0005
    FRAMES = 800
    damping = 0.2
    BOUND_BOX_SIZE = 4.0

    # SPH Params
    H = 0.06
    REST_DENSITY = 1000.0
    MASS = 1000.0
    GAS_CONST = 100.0
    VISCOSITY = 0.3

    # Coeficientes
    POLY6_COEFF = 315.0 / (64.0 * math.pi * H**9)
    SPIKY_COEFF = -45.0 / (math.pi * H**6)
    VISC_COEFF = 45.0 / (math.pi * H**6)

    # Configuración Grilla
    CELL_SIZE = H
    GRID_DIM_X = int(math.ceil(BOUND_BOX_SIZE / CELL_SIZE))
    GRID_DIM_Y = int(math.ceil(BOUND_BOX_SIZE / CELL_SIZE))
    GRID_DIM_Z = int(math.ceil(BOUND_BOX_SIZE / CELL_SIZE))
    NUM_CELLS = GRID_DIM_X * GRID_DIM_Y * GRID_DIM_Z

    print(f"N={N}. Grilla: {GRID_DIM_X}x{GRID_DIM_Y}x{GRID_DIM_Z} ({NUM_CELLS} celdas).")

    # --- MEMORIA GPU ---
    d_pos = cp.empty((N, dim), dtype=cp.float32)
    d_vel = cp.empty((N, dim), dtype=cp.float32)
    d_force = cp.empty((N, dim), dtype=cp.float32)
    d_pressure = cp.empty(N, dtype=cp.float32)
    d_density = cp.empty(N, dtype=cp.float32)

    d_grid_indices = cp.empty(N, dtype=cp.int32)
    d_cell_start = cp.full(NUM_CELLS, -1, dtype=cp.int32)
    d_cell_end = cp.full(NUM_CELLS, -1, dtype=cp.int32)

    gravity_cpu = np.array([0.0, -9.81, 0.0], dtype=np.float32)
    d_gravity = cp.asarray(gravity_cpu)

    # Config Kernel
    threads_per_block = 256
    blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block

    # Inicializar
    n_per_dim = int(math.ceil(N**(1.0 / dim)))

    # LLAMADA A KERNEL IMPORTADO (k.nombre_del_kernel)
    k.initialize_particles_kernel[blocks_per_grid, threads_per_block](
        d_pos, d_vel, d_force, d_pressure, d_density,
        N, dim, epsilon, n_per_dim
    )
    cuda.synchronize()

    # --- BUCLE PRINCIPAL ---
    os.makedirs("results", exist_ok=True)
    all_positions_list = []
    start_time = time.time()
    H_sq = H * H

    print(f"Iniciando simulación Grid SPH ({FRAMES} pasos)...")

    for frame_idx in range(FRAMES):
        # 1. Grilla
        k.calculate_grid_indices_kernel[blocks_per_grid, threads_per_block](
            d_pos, d_grid_indices, N, CELL_SIZE, GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z
        )

        # Ordenamiento (Sorting) - Esto se queda igual con CuPy
        sorted_indices = cp.argsort(d_grid_indices)
        d_pos = d_pos[sorted_indices]
        d_vel = d_vel[sorted_indices]
        d_pressure = d_pressure[sorted_indices]
        d_density = d_density[sorted_indices]
        d_grid_indices = d_grid_indices[sorted_indices]

        # Resetear celdas
        d_cell_start.fill(-1)
        d_cell_end.fill(-1)

        k.find_cell_bounds_kernel[blocks_per_grid, threads_per_block](
            d_grid_indices, d_cell_start, d_cell_end, N
        )

        # 2. Física
        k.compute_density_pressure_grid[blocks_per_grid, threads_per_block](
            d_pos, d_density, d_pressure, d_cell_start, d_cell_end,
            GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z, CELL_SIZE,
            N, H, H_sq, MASS, REST_DENSITY, GAS_CONST, POLY6_COEFF
        )

        d_force.fill(0.0)
        k.apply_gravity_kernel[blocks_per_grid, threads_per_block](d_force, d_gravity, N, dim)

        k.compute_forces_grid[blocks_per_grid, threads_per_block](
            d_pos, d_vel, d_force, d_density, d_pressure, d_cell_start, d_cell_end,
            GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z, CELL_SIZE,
            N, H, H_sq, MASS, VISCOSITY, SPIKY_COEFF, VISC_COEFF
        )

        k.integrate_kernel[blocks_per_grid, threads_per_block](
            d_pos, d_vel, d_force, dt, N, dim
        )

        k.check_boundaries_kernel[blocks_per_grid, threads_per_block](
            d_pos, d_vel, N, damping, BOUND_BOX_SIZE
        )

        cuda.synchronize()

        if frame_idx % 2 == 0:
            all_positions_list.append(d_pos.get())

        if frame_idx % 50 == 0:
            print(f"Frame {frame_idx}/{FRAMES} completado.")

    end_time = time.time()
    print(f"Simulación completada en {end_time - start_time:.4f} s.")

    final_data = np.array(all_positions_list, dtype=np.float32)
    np.savez_compressed("results/sph_grid_opt.npz", positions=final_data)
    print("Guardado.")

# --- PUNTO DE ENTRADA SEGURO ---
if __name__ == "__main__":
    run_simulation()
