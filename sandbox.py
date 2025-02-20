import sys
import numpy as np
from vispy import app, gloo

def mobius_transform(z, a):
    """Map z such that point a goes to 0."""
    return (z - a) / (1 - np.conj(a) * z)

def inverse_mobius_transform(z, a):
    """Invert the Mobius transformation that sends a to 0."""
    return (a + z) / (1 + np.conj(a) * z)

def compute_geodesic(x1, y1, x2, y2, n_samples=50):
    """Compute points along the hyperbolic geodesic between (x1, y1) and (x2, y2)
    using the Mobius transform approach.
    If the endpoints are collinear with the origin (or one is at 0), a straight line is returned.
    """
    z1, z2 = complex(x1, y1), complex(x2, y2)
    
    # If z1 is nearly 0 or both points lie along the same ray, return a straight Euclidean segment.
    # if abs(z1) < 1e-6 or abs(np.angle(z2) - np.angle(z1)) < 1e-6:
    #     return np.linspace([x1, y1], [x2, y2], n_samples)
    
    # Apply Mobius transform to send z1 to 0.
    z2_transformed = mobius_transform(z2, z1)
    
    # In the transformed domain, the geodesic is the straight line (radial segment) from 0 to z2_transformed.
    t = np.linspace(0, 1, n_samples)
    geodesic_transformed = t * z2_transformed
    
    # Map back to the original disk.
    geodesic_original = inverse_mobius_transform(geodesic_transformed, z1)
    return np.column_stack((geodesic_original.real, geodesic_original.imag))

# Euclidean (left side) vertex shader: Identity mapping.
vertex_euclid = """
uniform float u_rotation;
attribute vec2 a_position;
void main(void) {
    float c = cos(u_rotation);
    float s = sin(u_rotation);
    mat2 rotation = mat2(c, -s, s, c);
    vec2 pos = rotation * a_position;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

# Hyperbolic (right side) vertex shader:
vertex_hyper = """
uniform float u_rotation;
attribute vec2 a_position;
void main(void) {
    float c = cos(u_rotation);
    float s = sin(u_rotation);
    mat2 rotation = mat2(c, -s, s, c);
    vec2 pos = rotation * a_position;
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

class Canvas(app.Canvas):
    def __init__(self):
        # Set canvas size to twice as wide.
        app.Canvas.__init__(self, size=(1200, 600), keys='interactive')
        
        # Create shader programs.
        self.program_euclid = gloo.Program(vertex_euclid, fragment)
        self.program_hyper = gloo.Program(vertex_hyper, fragment)
        
        # Generate grid vertices for Euclidean space.
        n_lines = 201  # Adjust for clarity.
        xs = np.linspace(-1, 1, n_lines)
        ys = np.linspace(-1, 1, n_lines)
        grid_vertices = []
        
        # Vertical lines.
        for x in xs:
            grid_vertices.append([x, -1])
            grid_vertices.append([x, 1])
        
        # Horizontal lines.
        for y in ys:
            grid_vertices.append([-1, y])
            grid_vertices.append([1, y])
        
        grid_positions = np.array(grid_vertices, dtype=np.float32)
        
        # Compute hyperbolic geodesics using the Mobius transform.
        hyp_geodesics = []
        for i in range(0, len(grid_positions), 2):
            x1, y1 = grid_positions[i]
            x2, y2 = grid_positions[i+1]
            geodesic_points = compute_geodesic(x1, y1, x2, y2, n_samples=100)
            hyp_geodesics.extend(geodesic_points)
        
        hyp_positions = np.array(hyp_geodesics, dtype=np.float32)

        # Assign vertex data.
        self.program_euclid['a_position'] = grid_positions
        self.program_hyper['a_position'] = hyp_positions
        
        # Set clear color.
        gloo.set_clear_color('black')
        
        # Initialize common uniforms.
        self.rotation = 0.0
        self.program_euclid['u_rotation'] = self.rotation
        self.program_hyper['u_rotation'] = self.rotation
        
        # Set an initial color.
        init_color = (0.1, 0.8, 0.5, 1.0)
        self.program_euclid['u_color'] = init_color
        self.program_hyper['u_color'] = init_color
        
        # Timer for animation.
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

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
        self.program_hyper.draw('lines')
    
    def on_timer(self, event):
        self.rotation += 0.005
        self.update()
    
    def on_key_press(self, event):
        if event.key == 'Escape':
            self.close()
    
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

if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas = Canvas()
    canvas.show()
    app.run()
