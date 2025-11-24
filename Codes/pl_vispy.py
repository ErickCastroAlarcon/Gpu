import numpy as np
import pyvista as pv

# --- CONFIG ---
npz_path = "results/sph_grid_opt.npz"
output_video = "results/sim_animation.mp4"
frame_step = 1
fps = 30

# --- LOAD ---
print("Cargando datos...")
data = np.load(npz_path)
positions = data["positions"]  # shape (frames, N, 3)
# Aplicar frame step inmediatamente para ahorrar memoria/tiempo
positions = positions[::frame_step]
num_frames, N, dim = positions.shape
print(f"Datos cargados: {num_frames} frames, {N} partículas.")

# --- SETUP ESCENA (GPU) ---
# PyVista usa 'PolyData' para nubes de puntos
point_cloud = pv.PolyData(positions[0])

# Creamos el plotter (la ventana de renderizado)
# off_screen=True es importante para generar video sin abrir ventana
plotter = pv.Plotter(off_screen=True, window_size=(1080, 920)) 

# Añadimos los puntos. 
# render_points_as_spheres hace que se vea mucho mejor y más rápido
plotter.add_mesh(point_cloud, color="c", point_size=4, render_points_as_spheres=True)

# --- FIJAR LA CÁMARA Y LÍMITES ---
# En Matplotlib definiste límites fijos (-3 a 6, etc). 
# En PyVista, dibujamos una caja invisible (wireframe) para fijar la escala de la cámara
bounds = [-1, 1, -1, 1, -1, 1] # [xmin, xmax, ymin, ymax, zmin, zmax]
box = pv.Box(bounds)
plotter.add_mesh(box, style='wireframe', opacity=0.05, color='black')

plotter.set_background("white")
plotter.show_axes()

# 1. Definimos qué dirección es "arriba" (0,0,1 significa Z positivo)
plotter.camera.up = (0, 0, 1)

# 2. Hacia dónde mira la cámara (el centro aproximado de tus datos para que no se pierdan)
# Tus datos van aprox de X[-3,6], Y[0,6], Z[-4,4]. El centro es aprox (1.5, 3, 0)
plotter.camera.focal_point = (1.5, 3, 1)

# 3. Dónde está la cámara. 
# La ponemos lejos en X e Y, y un poco elevada en Z para ver en perspectiva.
plotter.camera.position = (1, 1, 2)

plotter.camera_position = 'xy' # O ajusta manualmente la vista inicial
plotter.camera.azimuth = 45    # Un poco de ángulo para ver 3D
plotter.camera.elevation = 30

plotter.open_movie(output_video, framerate=fps, quality=9)

# Barra de progreso simple
for i in range(num_frames):
    # Aquí está la magia: NO borramos y recreamos.
    # Solo actualizamos las coordenadas de los puntos existentes en memoria.
    point_cloud.points = positions[i]
   # point_cloud.active_scalars = positions[:, 2]
    plotter.write_frame()  # Renderiza el frame actual al video

    if i % 50 == 0:
        print(f"Procesando frame {i}/{num_frames}")

plotter.close()
print("¡Listo!")
