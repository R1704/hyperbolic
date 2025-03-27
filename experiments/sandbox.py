import sys
import numpy as np
from vispy import app, scene
from vispy.color import Colormap

import os
import vispy
vispy.use(app='pyqt5')


# Set up the canvas
canvas = scene.SceneCanvas(keys='interactive', bgcolor='w')
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(up='z', fov=60)

# Wave equation parameters - CORRECTED FOR STABILITY
nx, ny = 250, 250
dx = dy = 0.04
c = 1.0     # Reduced wave speed
dt = 0.02  # Much smaller time step for stability
r = (c * dt / dx) ** 2  # Stability parameter
print(f"Stability parameter r = {r}")  # Should be < 0.25 for 2D wave equation stability

# Initialize wave field
u = np.zeros((nx, ny))  # current
u_prev = np.zeros((nx, ny))  # previous
u_next = np.zeros((nx, ny))  # next

# Create a more pronounced initial pulse
rang = 2 * np.pi 
x = np.linspace(-rang, rang, nx)
y = np.linspace(-rang, rang, ny)
X, Y = np.meshgrid(x, y)
# Initial displacement - larger amplitude
u = np.exp(-1.0*(X**2 + Y**2))

# Simple initial conditions - create an outgoing wave
u_prev = u.copy() * 0.9  # This creates an outgoing wave

# Create surface plot
p1 = scene.visuals.SurfacePlot(z=u, color=(0.3, 0.3, 1, 1))
p1.transform = scene.transforms.MatrixTransform()
p1.transform.scale([1/249., 1/249., 0.5/249.])  # Scale z less for better visualization
p1.transform.translate([-0.5, -0.5, 0])

view.add(p1)

# Create and apply colormap with more contrast
colors = np.array([
    [0.0, 0.0, 1.0],  # blue
    [0.5, 0.5, 1.0],  # light blue
    [1.0, 1.0, 1.0],  # white
    [1.0, 0.5, 0.5],  # light red
    [1.0, 0.0, 0.0],  # red
])
positions = [0.0, 0.25, 0.5, 0.75, 1.0]
colormap = Colormap(colors, positions)
p1.clim = (-0.3, 0.3)  # More sensitive range to see smaller amplitudes
p1.cmap = colormap

frame_count = 0
skip_frames = 10  # Only update visualization every N physics steps

# Timer for animation
def update(event):
    global u, u_prev, u_next, frame_count
    
    # Run multiple physics steps per frame for speed
    for _ in range(skip_frames):
        # Wave equation step - using numpy operations for speed
        laplacian = (
            np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) + 
            np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4*u
        )
        u_next = 2*u - u_prev + r*laplacian
        
        # Simple absorbing boundary conditions
        u_next[0:2, :] = 0
        u_next[-2:, :] = 0
        u_next[:, 0:2] = 0
        u_next[:, -2:] = 0
        
        # Apply damping to prevent instability
        u_next *= 0.999
        
        # Update arrays - doing it efficiently
        u_prev, u, u_next = u, u_next, u_prev
    
    # Print max amplitude occasionally to verify wave is moving
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Max amplitude: {np.max(np.abs(u))}, Frame: {frame_count}")
    
    # Update visualization
    p1.set_data(z=u)
    
timer = app.Timer(interval=1, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()