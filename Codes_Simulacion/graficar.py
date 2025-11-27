import numpy as np
import pyvista as pv

# --- CONFIG ---
npz_path = "results/sph_grid_opt.npz"
output_video = "results/simulation_animation_4k.mp4"
frame_step = 1
fps = 60  # 60 fps para fluidez máxima

# --- LOAD ---
data = np.load(npz_path)
positions = data["positions"][::frame_step]
num_frames, N, dim = positions.shape

# --- SETUP PLOTTER (CALIDAD ULTRA) ---
# window_size=(3840, 2160) es 4K real.
# multi_samples=8 activa el Anti-aliasing por hardware (MSAA) al máximo.
plotter = pv.Plotter(off_screen=True, window_size=(3840, 2160), multi_samples=8)

# Activar "Super Sampling" (renderiza aún más grande y reduce, elimina bordes pixelados)
plotter.enable_anti_aliasing("ssaa")

# Activar "Eye Dome Lighting". 
# Esto da volumen y sombras a los puntos. Sin esto se ven planos.
plotter.enable_eye_dome_lighting()  

# Opcional: Si prefieres sombras realistas tradicionales (puede ser más lento/oscuro)
# plotter.enable_ssao(radius=2, bias=0.5) # Screen Space Ambient Occlusion

# --- OBJETOS ---
point_cloud = pv.PolyData(positions[0])

# specular=1.0 hace que las esferas brillen (reflejen luz), dando sensación 3D metálica/plástica
# ambient=0.3 asegura que las partes en sombra no sean negras totales
plotter.add_mesh(point_cloud, 
                 scalars=positions[0, :, 2], 
                 cmap="viridis",
                 point_size=10,             # Un poco más grandes para 4K
                 render_points_as_spheres=True,
                 specular=1.0,              # Brillo especular
                 specular_power=50,         # Qué tan concentrado es el brillo
                 ambient=0.3)               # Luz ambiental

# --- CAMARA Y FONDO ---
bounds = [-3, 6, 0, 6, -4, 4]
box = pv.Box(bounds)
plotter.add_mesh(box, style='wireframe', opacity=0.1, color='black', line_width=2) # line_width 2 para que se vea en 4K

plotter.set_background("white") 
# Opcional: Un fondo gradiente suele verse más "pro" que blanco puro
# plotter.set_background("black", top="royablue") 

plotter.camera.up = (0, 0, 1)
plotter.camera.focal_point = (1.5, 3, 0)
plotter.camera.position = (15, -5, 10)

# --- ANIMACIÓN ---
print(f"Renderizando 4K a {fps} FPS con Anti-aliasing y EDL...")

# quality=10 es el máximo en imageio para MP4
plotter.open_movie(output_video, framerate=fps, quality=10)

for i in range(num_frames):
    current_pos = positions[i]
    point_cloud.points = current_pos
    
    # Actualizar color scalar
    # Nota: Si cmap="viridis", a veces hay que indicar el rango si cambia mucho
    # point_cloud.point_data["Scalars"] = current_pos[:, 2] 
    # Para evitar errores si el nombre interno varía, usamos:
    point_cloud.active_scalars[:] = current_pos[:, 2]

    plotter.write_frame()
    
    if i % 25 == 0:
        print(f"Frame {i}/{num_frames}")

plotter.close()
print("¡Renderizado Ultra finalizado!")
