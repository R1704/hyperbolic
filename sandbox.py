import numpy as np
from vispy import app, gloo
import math

# Define cube vertices.
cube_vertices = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1]
], dtype=np.float32)

# Define faces (as indices for triangles) -- not shown for brevity.

def spherify_vertex(v):
    x, y, z = v
    x2, y2, z2 = x*x, y*y, z*z
    factor_x = np.sqrt(1 - (y2/2) - (z2/2) + (y2*z2/3))
    factor_y = np.sqrt(1 - (z2/2) - (x2/2) + (z2*x2/3))
    factor_z = np.sqrt(1 - (x2/2) - (y2/2) + (x2*y2/3))
    return np.array([x * factor_x, y * factor_y, z * factor_z], dtype=np.float32)

def morph_vertex(v, alpha):
    v_sphere = spherify_vertex(v)
    return (1 - alpha) * v + alpha * v_sphere

# Write vertex/fragment shaders for 3D.
vertex_shader = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
attribute vec3 a_position;
void main(void) {
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
}
"""
fragment_shader = """
uniform vec4 u_color;
void main(void) {
    gl_FragColor = u_color;
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(1200, 600), keys='interactive')
        # Create two programs: one for the left (Euclidean) and one for the right (hyperbolic)
        self.program_euclid = gloo.Program(vertex_shader, fragment_shader)
        self.program_hyper = gloo.Program(vertex_shader, fragment_shader)
        
        # Initialize the cube (and later morph it)
        self.cube = cube_vertices  # For simplicity; you’d usually have an indexed mesh.
        
        self.alpha = 0.0
        self.time = 0.0
        
        # Set up camera matrices (you need to compute model, view, projection matrices)
        # For Euclidean view:
        self.proj_euclid = ...  # e.g., perspective projection matrix.
        self.view_euclid = ...  # e.g., lookat matrix.
        # For hyperbolic view:
        self.proj_hyper = ...  # Might be similar, but with a scaling to mimic the Poincaré ball.
        self.view_hyper = ...  # Adjusted view for hyperbolic space.
        
        self.program_euclid['u_color'] = (0.1, 0.8, 0.5, 1.0)
        self.program_hyper['u_color'] = (0.1, 0.8, 0.5, 1.0)
        
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)
        
    def on_draw(self, event):
        gloo.clear()
        width, height = self.size
        
        # Compute the current vertex positions via morphing.
        alpha = 0.5 * (1 + math.sin(self.time))
        morphed_vertices = np.array([morph_vertex(v, alpha) for v in self.cube])
        
        # Set up model matrix, etc.
        model = ...  # Possibly identity or a rotation.
        
        # Left viewport: Euclidean view.
        gloo.set_viewport(0, 0, width//2, height)
        self.program_euclid['u_model'] = model
        self.program_euclid['u_view'] = self.view_euclid
        self.program_euclid['u_projection'] = self.proj_euclid
        self.program_euclid['a_position'] = morphed_vertices
        self.program_euclid.draw('points')  # or draw triangles for faces.
        
        # Right viewport: Hyperbolic view.
        gloo.set_viewport(width//2, 0, width//2, height)
        self.program_hyper['u_model'] = model
        self.program_hyper['u_view'] = self.view_hyper
        self.program_hyper['u_projection'] = self.proj_hyper
        self.program_hyper['a_position'] = morphed_vertices  # Assume they lie in the unit ball.
        self.program_hyper.draw('points')
    
    def on_timer(self, event):
        self.time += event.dt
        self.update()
        
if __name__ == '__main__':
    canvas = Canvas()
    canvas.show()
    app.run()
