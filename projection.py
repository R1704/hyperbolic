# TODO: smoother grid lines, better inversion visualization
# TODO: zoom in/out, pan, tilt, etc.
# TODO: better color selection
# TODO: wave function visualization
# TODO: make it a cube
# TODO: Cube-to-Sphere Morphing
# TODO: Poincaré Ball Model (3D)


import sys
import numpy as np
from vispy import app, gloo
from utils import get_unit_circle, get_geodesic, circle_inversion
import streamlit as st
import numpy as np
from PIL import Image
from vispy import app


# Import your Canvas from projection.py.
from projection import Canvas

def main():
    st.title("Hyperbolic Projection (Offscreen)")
    
    # Create the canvas without showing an interactive window.
    canvas = Canvas()
    
    # Force one draw cycle.
    canvas.on_draw(None)
    # Render the current buffer to an image array.
    img_data = canvas.render()  # Returns an RGBA array (float32, [0,1])
    
    # Convert to an 8-bit image for display.
    img = Image.fromarray((np.clip(img_data, 0, 1) * 255).astype(np.uint8))
    
    st.image(img, caption="Projection", use_column_width=True)

if __name__ == "__main__":
    main()

# Vertex shaders (Euclidean and Hyperbolic share similar code):
vertex_shader_template = """
uniform float u_rotation;
uniform float u_scale;
uniform float u_tilt;
uniform vec2 u_translation;  // New translation uniform.
attribute vec2 a_position;
void main(void) {
    float c = cos(u_rotation);
    float s = sin(u_rotation);
    mat2 rotation = mat2(c, -s, s, c);
    
    vec2 pos = rotation * (a_position * u_scale) + u_translation;  // Apply translation.
    pos.y += u_tilt * pos.x;
    
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

fragment = """
uniform vec4 u_color;
void main(void) {
    gl_FragColor = u_color;
}
"""

def transform_vertex(vertex, rotation, scale, tilt, translation):
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    # Apply scale and rotation
    transformed = R @ (vertex * scale)
    # Apply translation
    transformed += translation
    # Apply tilt (skew) transformation
    transformed[1] += tilt * transformed[0]
    return transformed

radius = 0.5

class Canvas(app.Canvas):
    def __init__(self):
        super().__init__(size=(1200, 600), keys='interactive')
        # Create shader programs.
        self.program_euclid = gloo.Program(vertex_shader_template, fragment)
        self.program_hyper = gloo.Program(vertex_shader_template, fragment)

        gloo.set_clear_color('black')
        self._init_uniforms()

        # Create unit circle for Poincare disc.
        self.unit_circle = get_unit_circle(scale=radius)
        
        # Build Euclidean grid and corresponding hyperbolic geodesics.
        self.geodesics = self._create_grid_and_geodesics()
        self.inversions = self._calculate_inversions()

        # Assign initial vertex data.
        self.program_euclid['a_position'] = self.grid_positions
        self.rotation = 0.0
        # New attribute for tilt controlled by drag.
        self.tilt = 0.0
        # Store the last mouse position.
        self._last_mouse_pos = None

        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

    def _init_uniforms(self):
        """Initializes uniforms common to both shader programs."""
        init_color = (0/255.0, 255/255.0, 216/255.0, 1.0)
        for program in (self.program_euclid, self.program_hyper):
            program['u_scale'] = 1.0
            program['u_tilt'] = 0.0
            program['u_rotation'] = 0.0
            program['u_translation'] = (0.0, 0.0)  # Initial translation.
            program['u_color'] = init_color

    def _create_grid_and_geodesics(self):
        """Generates grid positions and computes hyperbolic geodesic segments."""
        n_lines = 21
        scale = 0.5
        xs = np.linspace(-scale, scale, n_lines)
        ys = np.linspace(-scale, scale, n_lines)
        
        grid_vertices = []
        # Vertical lines.
        for x in xs:
            grid_vertices.extend([[x, -scale], [x, scale]])
        # Horizontal lines.
        for y in ys:
            grid_vertices.extend([[-scale, y], [scale, y]])
        
        self.grid_positions = np.array(grid_vertices, dtype=np.float32)
        segments = []
        for i in range(0, len(self.grid_positions), 2):
            z1 = complex(*self.grid_positions[i])
            z2 = complex(*self.grid_positions[i+1])
            arc = get_geodesic(z1, z2)
            seg = np.column_stack((arc.real, arc.imag)).astype(np.float32)
            norms = np.sqrt(np.sum(seg ** 2, axis=1))
            segments.append(seg[norms <= radius])
        return segments

    def _calculate_inversions(self):
        """Calculates the inversion of each point on the hyperbolic geodesics."""
        inv_points = []
        for seg in self.geodesics:
            for pt in seg:
                z = complex(pt[0], pt[1])
                z_inv = circle_inversion(z, 0, radius)
                inv_points.append([z_inv.real, z_inv.imag])
        return np.array(inv_points, dtype=np.float32)

    def on_draw(self, event):
        gloo.clear()
        width, height = self.size
        
        # Draw left side: Euclidean grid.
        gloo.set_viewport(0, 0, width // 2, height)
        self.program_euclid['u_rotation'] = self.rotation
        self.program_euclid.draw('lines')
        
        # Draw right side: Hyperbolic grid and unit circle boundary.
        gloo.set_viewport(width // 2, 0, width // 2, height)
        self.program_hyper['u_rotation'] = self.rotation
        
        for seg in self.geodesics:
            self.program_hyper['a_position'] = seg
            self.program_hyper.draw('line_strip')
        
        self.program_hyper['a_position'] = self.unit_circle
        self.program_hyper['u_color'] = (0.1, 0.1, 0.1, 0.0)
        self.program_hyper.draw('line_strip')

        if self.inversions.size:
            self.program_hyper['u_color'] = (0/255.0, 208/255.0, 255/255.0, 1.0)
            self.program_hyper['a_position'] = self.inversions
            self.program_hyper.draw('points')

    def on_mouse_press(self, event):
        self._last_mouse_pos = event.pos
        print("Mouse press:", event.pos)

    def on_mouse_move(self, event):
        # If the left mouse button (button 1) is pressed, treat it as a drag.
        if event.buttons[0]:
            if self._last_mouse_pos is None:
                self._last_mouse_pos = event.pos
                return

            dx = event.pos[0] - self._last_mouse_pos[0]
            dy = event.pos[1] - self._last_mouse_pos[1]

            self.rotation += dx * 0.005
            self.tilt += dy * 0.005

            self.program_euclid['u_tilt'] = self.tilt
            self.program_hyper['u_tilt'] = self.tilt

            self._last_mouse_pos = event.pos

            print("Mouse move (drag) - dx:", dx, "dy:", dy)
            print("Rotation:", self.rotation, "Tilt:", self.tilt)

            self.update()
        else:
            self._last_mouse_pos = None

    def on_mouse_release(self, event):
        self._last_mouse_pos = None
        print("Mouse release")

    def on_key_press(self, event):
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()
        elif event.key.name == 'Up':
            for program in (self.program_euclid, self.program_hyper):
                program['u_scale'] *= 1.1
            self.update()
        elif event.key.name == 'Down':
            for program in (self.program_euclid, self.program_hyper):
                program['u_scale'] /= 1.1
            self.update()
        elif event.text.upper() == 'Q':
            self.close()

    def on_timer(self, event):
        self.rotation += 0.005
        self.update()

if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas = Canvas()
    canvas.show()
    app.run()