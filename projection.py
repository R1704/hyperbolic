# TODO: smoother grid lines, better inversion visualization
# TODO: zoom in/out, pan, tilt, etc.
# TODO: better color selection
# TODO: wave function visualization


import sys
import numpy as np
from vispy import app, gloo

from utils import *




# Euclidean (left side) vertex shader: Identity mapping.
vertex_euclid = """
uniform float u_rotation;
uniform float u_scale;   // New uniform for zooming.
uniform float u_tilt;    // New uniform for tilting.
attribute vec2 a_position;
void main(void) {
    float c = cos(u_rotation);
    float s = sin(u_rotation);
    mat2 rotation = mat2(c, -s, s, c);
    
    // Apply scale (zoom)
    vec2 pos = (a_position * u_scale);
    
    // Apply rotation.
    pos = rotation * pos;
    
    // Apply a tilt (simple skew as an example).
    pos.y += u_tilt * pos.x;
    
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

# Hyperbolic (right side) vertex shader:
vertex_hyper = """
uniform float u_rotation;
uniform float u_scale;   // New uniform for zooming.
uniform float u_tilt;    // New uniform for tilting.
attribute vec2 a_position;
void main(void) {
    float c = cos(u_rotation);
    float s = sin(u_rotation);
    mat2 rotation = mat2(c, -s, s, c);
    
    // Apply scale (zoom)
    vec2 pos = (a_position * u_scale);
    
    // Apply rotation.
    pos = rotation * pos;
    
    // Apply a tilt (simple skew as an example).
    pos.y += u_tilt * pos.x;
    
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""


# Fragment shader: Colors the fragment.
fragment = """
uniform vec4 u_color;
void main(void) {
    gl_FragColor = u_color;
}
"""









radius = 0.5






class Canvas(app.Canvas):
    def __init__(self):
        # Set canvas size to twice as wide.
        app.Canvas.__init__(self, size=(1200, 600), keys='interactive')
        
        # Create shader programs.
        self.program_euclid = gloo.Program(vertex_euclid, fragment)
        self.program_hyper = gloo.Program(vertex_hyper, fragment)

        # Set clear color.
        gloo.set_clear_color('black')

        self.program_euclid['u_scale'] = 1.0
        self.program_hyper['u_scale'] = 1.0

        self.tilt = 0.0
        self.program_euclid['u_tilt'] = self.tilt
        self.program_hyper['u_tilt'] = self.tilt

        
        
        # Set an initial color.
        init_color = (0.1, 0.8, 0.5, 1.0)
        self.program_euclid['u_color'] = init_color
        self.program_hyper['u_color'] = init_color
        

        # Draw the unit circle in the Poincare disk.
        self.unit_circle = get_unit_circle(scale=radius)

        self.inversions = []

        
        # Generate grid vertices for Euclidean space.
        n_lines = 11  # Adjust for clarity.
        scale = 0.5
        xs = np.linspace(-scale, scale, n_lines)
        ys = np.linspace(-scale, scale, n_lines)
        
        grid_vertices = []
        
        # Vertical lines.
        for x in xs:
            grid_vertices.append([x, -scale])
            grid_vertices.append([x, scale])
        
        # Horizontal lines.
        for y in ys:
            grid_vertices.append([-scale, y])
            grid_vertices.append([scale, y])
        
        grid_positions = np.array(grid_vertices, dtype=np.float32)
        self.geodesics = self.calculate_polygon_segments(grid_positions)

        # Calculate inversions.
        self.inversions = self.calculate_inversions()

        # Assign vertex data.
        self.program_euclid['a_position'] = grid_positions
        # self.program_hyper['a_position'] = self.geodesics
        
        

        # Initialize common uniforms.
        self.rotation = 0.0
        self.program_euclid['u_rotation'] = self.rotation
        self.program_hyper['u_rotation'] = self.rotation

        # Timer for animation.
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

    def calculate_polygon_segments(self, grid_positions):
        segments = []

        # Each line is defined by two consecutive points.
        for i in range(0, len(grid_positions), 2):
            z1 = complex(*grid_positions[i])
            z2 = complex(*grid_positions[i+1])
            
            # Compute the geodesic (a hyperbolic arc) between those endpoints.
            arc = get_geodesic(z1, z2)
            seg = np.column_stack((arc.real, arc.imag)).astype(np.float32)
            
            # Only keep points that are inside the unit disc.
            norms = np.sqrt(np.sum(seg**2, axis=1))
            seg_filtered = seg[norms <= radius]
            
            segments.append(seg_filtered)
            # segments.append(seg)
        return segments
    
    def calculate_inversions(self):
        inversions = []
        for seg in self.geodesics:  # each seg is an array of shape (n,2)
            for pt in seg:  # iterate over points in the segment
                z = complex(pt[0], pt[1])
                z_inverted = circle_inversion(z, 0, radius)
                inversions.append((z_inverted.real, z_inverted.imag))
        return np.array(inversions, dtype=np.float32)

    def on_draw(self, event):
        gloo.clear()
        width, height = self.size
        
        # Left Half: Euclidean grid.
        gloo.set_viewport(0, 0, width//2, height)
        self.program_euclid['u_rotation'] = self.rotation
        self.program_euclid.draw('lines')
        
        # Right Half: Hyperbolic geodesic grid.
        gloo.set_viewport(width//2, 0, width//2, height)
        self.program_hyper['u_rotation'] = self.rotation
        
        # Draw each geodesic segment separately.
        for seg in self.geodesics:
            self.program_hyper['a_position'] = seg
            self.program_hyper.draw('line_strip')
        
        # Draw unit disc boundary.
        self.program_hyper['a_position'] = self.unit_circle
        self.program_hyper['u_color'] = (0.1, 0.1, 0.1, 0.0)
        self.program_hyper.draw('line_strip')

        if self.inversions is not None:
            self.program_hyper['u_color'] = (1.0, 0.0, 1.0, 1.0)
            self.program_hyper['a_position'] = self.inversions
            self.program_hyper.draw('points')
        
    
    def on_timer(self, event):
        # self.rotation += 0.005
        self.update()
    
    
    def on_mouse_press(self, event):
        # Change color based on mouse click position.
        x, y = event.pos
        width, height = self.size
        red = x / width
        green = y / height
        blue = 0.5
        new_color = (red, green, blue, 1.0)
        self.program_euclid['u_color'] = new_color
        self.program_hyper['u_color'] = new_color

    def on_mouse_drag(self, event):
        # Use event.delta[1] instead of computing a difference
        delta_tilt = event.delta[1] / 100.0
        self.tilt += delta_tilt
        self.program_euclid['u_tilt'] = self.tilt
        self.program_hyper['u_tilt'] = self.tilt
        self.update()

    def on_key_press(self, event):
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()

        if event.key.name == 'Up':
            self.program_euclid['u_scale'] *= 1.1
            self.program_hyper['u_scale'] *= 1.1
            self.update()

        if event.key.name == 'Down':
            self.program_euclid['u_scale'] /= 1.1
            self.program_hyper['u_scale'] /= 1.1
            self.update()

        if event.text == 'Q':
            self.close()


if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas = Canvas()
    canvas.show()
    app.run()
