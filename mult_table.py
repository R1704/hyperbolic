import sys
import numpy as np
from vispy import app, gloo
from utils import get_unit_circle, get_geodesic, circle_inversion
from vispy import app
app.use_app('egl')


# Vertex shader (common for Euclidean and Hyperbolic):
vertex_shader_template = """
uniform float u_rotation;
uniform float u_scale;
uniform float u_tilt;
uniform vec2 u_translation;
attribute vec2 a_position;
void main(void) {
    float c = cos(u_rotation);
    float s = sin(u_rotation);
    mat2 rotation = mat2(c, -s, s, c);
    
    vec2 pos = rotation * (a_position * u_scale) + u_translation;
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
    transformed = R @ (vertex * scale)  # Scale and rotate
    transformed += translation           # Translate
    transformed[1] += tilt * transformed[0]  # Apply tilt (skew)
    return transformed

radius = 0.5

class Canvas(app.Canvas):
    def __init__(self):
        super().__init__(size=(1200, 600), keys='interactive')
        # Create shader programs.
        self.program_euclid = gloo.Program(vertex_shader_template, fragment)
        self.program_hyper = gloo.Program(vertex_shader_template, fragment)
        self.program_lines = gloo.Program(vertex_shader_template, fragment)
        self.program_points = gloo.Program("""
        uniform float u_rotation;
        uniform float u_scale;
        uniform float u_tilt;
        uniform vec2 u_translation;
        attribute vec2 a_position;
        void main(void) {
            float c = cos(u_rotation);
            float s = sin(u_rotation);
            mat2 rotation = mat2(c, -s, s, c);
            
            vec2 pos = rotation * (a_position * u_scale) + u_translation;
            pos.y += u_tilt * pos.x;
            
            gl_Position = vec4(pos, 0.0, 1.0);
            gl_PointSize = 3.0;
        }
        """, fragment)

        gloo.set_clear_color('black')
        self._init_uniforms()

        # Use a vectorized unit circle for the Poincaré disk.
        self.unit_circle = get_unit_circle(scale=radius)
        
        # Modular multiplication visualization parameters
        self.t = 0.0
        self.step = 0.01
        self.time_active = True
        self.num_dots = 100  # Adjust for performance as needed
        self.show_modular_viz = True
        self.show_inversions = True
        
        # Cache for HSV-to-RGB conversions.
        self.colors_cache = {}
        
        # Initial calculation
        self.update_modular_data()

        self.rotation = 0.0
        self.tilt = 0.0
        self._last_mouse_pos = None

        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

    def _init_uniforms(self):
        init_color = (0/255.0, 255/255.0, 216/255.0, 1.0)
        for program in (self.program_euclid, self.program_hyper, self.program_lines):
            program['u_scale'] = 1.0
            program['u_tilt'] = 0.0
            program['u_rotation'] = 0.0
            program['u_translation'] = (0.0, 0.0)
            program['u_color'] = init_color
        
        self.program_points['u_scale'] = 1.0
        self.program_points['u_tilt'] = 0.0
        self.program_points['u_rotation'] = 0.0
        self.program_points['u_translation'] = (0.0, 0.0)
        self.program_points['u_color'] = (0.0, 1.0, 1.0, 1.0)

    def _hsv_to_rgb(self, h, s=1.0, v=1.0):
        key = (h, s, v)
        if key in self.colors_cache:
            return self.colors_cache[key]
        
        if s == 0.0:
            rgb = (v, v, v)
        else:
            h /= 60.0
            i = int(h)
            f = h - i
            p = v * (1.0 - s)
            q = v * (1.0 - s * f)
            t = v * (1.0 - s * (1.0 - f))
            if i == 0:
                rgb = (v, t, p)
            elif i == 1:
                rgb = (q, v, p)
            elif i == 2:
                rgb = (p, v, t)
            elif i == 3:
                rgb = (p, q, v)
            elif i == 4:
                rgb = (t, p, v)
            else:
                rgb = (v, p, q)
        
        self.colors_cache[key] = rgb
        return rgb

    def update_modular_data(self):
        """Calculate modular multiplication data for current t value using vectorized operations."""
        angle = 2 * np.pi / self.num_dots
        
        # Vectorized calculation for dot positions.
        angles = np.linspace(np.pi, np.pi + 2 * np.pi, self.num_dots, endpoint=False)
        self.dot_positions = np.stack((radius * np.cos(angles), radius * np.sin(angles)), axis=-1)
        
        # Vectorized modular arithmetic for lines.
        indices = np.arange(self.num_dots)
        targets = (indices * self.t) % self.num_dots
        targets = targets.astype(int)
        self.modular_lines = np.empty((self.num_dots * 2, 2), dtype=np.float32)
        self.modular_lines[0::2] = self.dot_positions
        self.modular_lines[1::2] = self.dot_positions[targets]
        
        # Calculate geodesics for hyperbolic visualization.
        self.geodesic_segments = []
        for i in range(0, len(self.modular_lines), 2):
            z1 = complex(self.modular_lines[i][0], self.modular_lines[i][1])
            z2 = complex(self.modular_lines[i+1][0], self.modular_lines[i+1][1])
            arc = get_geodesic(z1, z2)
            if arc.size > 0:
                seg = np.column_stack((arc.real, arc.imag)).astype(np.float32)
                norms = np.sqrt(np.sum(seg * seg, axis=1))
                valid_seg = seg[norms <= radius]
                if valid_seg.shape[0] > 1:
                    self.geodesic_segments.append(valid_seg)
        
        # Calculate inversion points using a less strict condition (<= instead of <).
        self.inversion_points = []
        for seg in self.geodesic_segments:
            for pt in seg:
                z = complex(pt[0], pt[1])
                z_inv = circle_inversion(z, 0, radius)
                if abs(z_inv) <= radius:  # Allow boundary points
                    self.inversion_points.append([z_inv.real, z_inv.imag])
        
        if self.inversion_points:
            self.inversion_points = np.array(self.inversion_points, dtype=np.float32)
        else:
            self.inversion_points = np.array([], dtype=np.float32)
        print(f"Generated {len(self.inversion_points)} inversion points")

    def on_draw(self, event):
        gloo.clear()
        width, height = self.size
        
        # Draw left side: Euclidean visualization.
        gloo.set_viewport(0, 0, width // 2, height)
        self.program_euclid['u_rotation'] = self.rotation
        
        # Draw circle boundary on left side.
        self.program_euclid['a_position'] = self.unit_circle
        self.program_euclid['u_color'] = (0.3, 0.3, 0.3, 1.0)
        self.program_euclid.draw('line_strip')
        
        # Draw modular lines on left side.
        if self.show_modular_viz and self.modular_lines.size:
            # Batch drawing in chunks.
            batch_size = 20
            for i in range(0, len(self.modular_lines), batch_size):
                batch = self.modular_lines[i:i+batch_size]
                if len(batch) < 2:
                    continue
                h = ((i // 2) % self.num_dots) / self.num_dots
                r, g, b = self._hsv_to_rgb(h * 360)
                self.program_lines['u_color'] = (r, g, b, 1.0)
                self.program_lines['a_position'] = batch
                self.program_lines.draw('lines')
        
        # Draw right side: Hyperbolic visualization.
        gloo.set_viewport(width // 2, 0, width // 2, height)
        self.program_hyper['u_rotation'] = self.rotation
        
        # Draw circle boundary on right side.
        self.program_hyper['a_position'] = self.unit_circle
        self.program_hyper['u_color'] = (0.3, 0.3, 0.3, 1.0)
        self.program_hyper.draw('line_strip')
        
        if self.show_modular_viz:
            # Draw hyperbolic geodesics.
            for i in range(0, len(self.modular_lines), 2):
                if i + 1 >= len(self.modular_lines):
                    break
                p1 = self.modular_lines[i]
                p2 = self.modular_lines[i+1]
                z1 = complex(p1[0], p1[1])
                z2 = complex(p2[0], p2[1])
                arc = get_geodesic(z1, z2)
                if arc.size > 0:
                    seg = np.column_stack((arc.real, arc.imag)).astype(np.float32)
                    norms = np.sqrt(np.sum(seg * seg, axis=1))
                    valid_seg = seg[norms <= radius]
                    if valid_seg.shape[0] > 1:
                        h = ((i // 2) % self.num_dots) / self.num_dots
                        r, g, b = self._hsv_to_rgb(h * 360)
                        self.program_lines['u_color'] = (r, g, b, 1.0)
                        self.program_lines['a_position'] = valid_seg
                        self.program_lines.draw('line_strip')
            
        # Draw inversions.
        if self.show_inversions and self.inversion_points.size:
            self.program_points['u_color'] = (0.0, 1.0, 1.0, 1.0)  # Brighter cyan
            self.program_points['u_rotation'] = self.rotation
            self.program_points['u_tilt'] = self.tilt
            self.program_points['a_position'] = self.inversion_points
            self.program_points.draw('points')

    def on_mouse_press(self, event):
        self._last_mouse_pos = event.pos

    def on_mouse_move(self, event):
        if event.buttons[0]:
            if self._last_mouse_pos is None:
                self._last_mouse_pos = event.pos
                return

            dx = event.pos[0] - self._last_mouse_pos[0]
            dy = event.pos[1] - self._last_mouse_pos[1]

            self.rotation += dx * 0.005
            self.tilt += dy * 0.005

            for prog in (self.program_euclid, self.program_hyper, self.program_lines, self.program_points):
                prog['u_tilt'] = self.tilt

            self._last_mouse_pos = event.pos
            self.update()
        else:
            self._last_mouse_pos = None

    def on_mouse_release(self, event):
        self._last_mouse_pos = None

    def on_key_press(self, event):
        if event.text == ' ':
            self.time_active = not self.time_active
            if self.time_active:
                self.timer.start()
            else:
                self.timer.stop()
        elif event.key.name == 'Up':
            self.step += 0.01
            print(f"Step size: {self.step:.2f}")
        elif event.key.name == 'Down':
            self.step = max(0.001, self.step - 0.01)
            print(f"Step size: {self.step:.2f}")
        elif event.key.name == 'Right':
            self.num_dots += 1
            print(f"Number of dots: {self.num_dots}")
            self.update_modular_data()
        elif event.key.name == 'Left':
            self.num_dots = max(3, self.num_dots - 1)
            print(f"Number of dots: {self.num_dots}")
            self.update_modular_data()
        elif event.text.upper() == 'V':
            self.show_modular_viz = not self.show_modular_viz
        elif event.text.upper() == 'I':
            self.show_inversions = not self.show_inversions
            print(f"Inversions {'enabled' if self.show_inversions else 'disabled'}")
        elif event.text.upper() == 'R':
            self.num_dots = 50
            self.step = 0.01
            self.update_modular_data()
            print("Reset to optimal performance settings")
        elif event.text.upper() == 'P':
            if hasattr(self, 'inversion_density'):
                self.inversion_density = (self.inversion_density + 1) % 3
            else:
                self.inversion_density = 0
            densities = ["Low", "Medium", "High"]
            print(f"Inversion point density: {densities[self.inversion_density]}")
            self.update_modular_data()
        elif event.text == '2':
            self.t = 2
            self.update_modular_data()
        elif event.text == '3':
            self.t = 3
            self.update_modular_data()
        elif event.text == '5':
            self.t = 51
            self.update_modular_data()
        elif event.text == '9':
            self.t = 99
            self.update_modular_data()
        elif event.text.upper() == 'Q':
            self.close()
        self.update()

    def on_timer(self, event):
        if self.time_active:
            self.t += self.step
            if int(self.t * 100) % 4 == 0:
                self.update_modular_data()
        self.update()

if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas = Canvas()
    canvas.show()
    app.run()
