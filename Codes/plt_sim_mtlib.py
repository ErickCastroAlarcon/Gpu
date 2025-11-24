#nimate_npz_matplotlib.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# --- CONFIG ---
npz_path = "results/simulation_data.npz"
output_video = "results/simulation_animation_1.mp4"
frame_step = 1              # usa cada frame_step-ésimo frame
dpi = 300

# --- LOAD ---
data = np.load(npz_path)
positions = data["positions"]  # shape (frames, N, 3)
num_frames, N, dim = positions.shape
print("Loaded", positions.shape)

# fig
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect((7,7,7))

# límites
min_xyz = positions.min()
max_xyz = positions.max()
ax.set_xlim(-3, 6)
ax.set_ylim(0, 6)
ax.set_zlim(-4, 4)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

scat = ax.scatter(positions[0, :, 0], positions[0, :, 1], positions[0, :, 2], s=2)

def update(i):
    i = i * frame_step
    xyz = positions[i]
    # Para 3D scatter hay que actualizar offsets así
    scat._offsets3d = (xyz[:,0], xyz[:,1], xyz[:,2])
    ax.set_title(f"Frame {i}/{num_frames-1}")
    return scat,

# animar y guardar
frames_to_use = num_frames // frame_step
ani = FuncAnimation(fig, update, frames=frames_to_use, interval=30, blit=False)
writer = FFMpegWriter(fps=30)
ani.save(output_video, writer=writer, dpi=dpi)
print("Vídeo guardado en", output_video)

