import numpy as np
from vispy import app, gloo
import vispy
# You can switch between PyQt5 and PyQt6 by changing this line.
vispy.use(app='PyQt6')

# --- Simulation Parameters ---
dt = 0.1
c = 1.0
damping = 1.0
N = 100  # grid resolution

# Create physical grid: x, y in [-2pi, 2pi]
x = np.linspace(-2 * np.pi, 2 * np.pi, N)
y = np.linspace(-2 * np.pi, 2 * np.pi, N)
X, Y = np.meshgrid(x, y)

# Initial condition: Gaussian bump
Z_init = np.exp(-(X**2 + Y**2)).astype(np.float32)

# Pack into an RGBA texture (store the simulation value in the red channel)
def pack_rgba(Z):
    A = np.zeros((N, N, 4), dtype=np.float32)
    A[..., 0] = Z  # simulation value in red
    A[..., 3] = 1.0  # full opacity
    return A

Z_rgba = pack_rgba(Z_init)

# --- Prepare Mesh for Rendering ---
positions = np.empty((N, N, 3), dtype=np.float32)
positions[..., 0] = X
positions[..., 1] = Y
positions[..., 2] = 0.0
positions = positions.reshape(-1, 3)

texcoords = np.empty((N, N, 2), dtype=np.float32)
u = np.linspace(0, 1, N)
v = np.linspace(0, 1, N)
U, V = np.meshgrid(u, v)
texcoords[..., 0] = U
texcoords[..., 1] = V
texcoords = texcoords.reshape(-1, 2)

indices = []
for i in range(N - 1):
    for j in range(N - 1):
        i0 = i * N + j
        i1 = i0 + 1
        i2 = i0 + N
        i3 = i0 + N + 1
        indices.extend([i0, i2, i1])
        indices.extend([i1, i2, i3])
indices = np.array(indices, dtype=np.uint32)

# --- Shader Programs ---
# The simulation shader now uses a uniform 'u_offset' for all texture coordinate steps.
sim_vertex = """
attribute vec2 a_position;
varying vec2 v_texcoord;
void main(){
    v_texcoord = (a_position + 1.0) * 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

sim_fragment = """
uniform sampler2D u_current;
uniform sampler2D u_previous;
uniform float u_dt;
uniform float u_offset; // texture coordinate step = 1/(N-1)
uniform float u_c;
uniform float u_damping;
varying vec2 v_texcoord;
void main(){
    float center = texture2D(u_current, v_texcoord).r;
    float up     = texture2D(u_current, v_texcoord + vec2(0.0, u_offset)).r;
    float down   = texture2D(u_current, v_texcoord - vec2(0.0, u_offset)).r;
    float left   = texture2D(u_current, v_texcoord - vec2(u_offset, 0.0)).r;
    float right  = texture2D(u_current, v_texcoord + vec2(u_offset, 0.0)).r;
    
    // Laplacian computed entirely in texture coordinate space.
    float laplacian = (up + down + left + right - 4.0 * center) / (u_offset * u_offset);
    
    float previous = texture2D(u_previous, v_texcoord).r;
    float next = (2.0 * center - previous + (u_c * u_dt) * (u_c * u_dt) * laplacian) * u_damping;
    gl_FragColor = vec4(next, 0.0, 0.0, 1.0);
}
"""

render_vertex = """
attribute vec3 a_position;
attribute vec2 a_texcoord;
uniform sampler2D u_height;
uniform float u_height_scale;
varying vec3 v_color;
void main(){
    float h = texture2D(u_height, a_texcoord).r;
    vec3 pos = a_position;
    pos.z = h * u_height_scale;
    gl_Position = vec4(pos, 1.0);
    v_color = vec3(0.5 + h*0.5, 0.5 - h*0.5, 1.0 - abs(h));
}
"""

render_fragment = """
varying vec3 v_color;
void main(){
    gl_FragColor = vec4(v_color, 1.0);
}
"""

# --- Canvas Class Definition ---
class WaveCanvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600))
        gloo.set_state(clear_color='black', depth_test=True)

        # Setup simulation shader program and full-screen quad.
        self.sim_program = gloo.Program(sim_vertex, sim_fragment)
        quad = np.array([[-1, -1], [ 1, -1], [-1,  1], [ 1,  1]], dtype=np.float32)
        self.sim_program['a_position'] = quad

        # Use RGBA textures for simulation state.
        self.texture_current = gloo.Texture2D(Z_rgba, interpolation='linear')
        self.texture_previous = gloo.Texture2D(Z_rgba, interpolation='linear')
        self.texture_next = gloo.Texture2D(shape=Z_rgba.shape, interpolation='linear')

        self.fbo = gloo.FrameBuffer(color=self.texture_next)

        self.sim_program['u_current'] = self.texture_current
        self.sim_program['u_previous'] = self.texture_previous
        self.sim_program['u_dt'] = dt
        self.sim_program['u_offset'] = 1.0 / (N - 1)
        self.sim_program['u_c'] = c
        self.sim_program['u_damping'] = damping

        # Setup rendering shader program.
        self.render_program = gloo.Program(render_vertex, render_fragment)
        self.render_program['a_position'] = positions
        self.render_program['a_texcoord'] = texcoords
        self.render_program['u_height_scale'] = 1.0
        self.render_program['u_height'] = self.texture_current
        self.index_buffer = gloo.IndexBuffer(indices)

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.show()

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.render_program['u_height'] = self.texture_current
        self.render_program.draw('triangles', self.index_buffer)

    def on_timer(self, event):
        # Update simulation: draw simulation shader to offscreen FBO.
        self.fbo.activate()
        gloo.clear(color=True, depth=True)
        self.sim_program.draw('triangle_strip')
        self.fbo.deactivate()

        # Ping–pong textures.
        temp = self.texture_previous
        self.texture_previous = self.texture_current
        self.texture_current = self.texture_next
        self.texture_next = temp

        self.sim_program['u_current'] = self.texture_current
        self.sim_program['u_previous'] = self.texture_previous
        self.render_program['u_height'] = self.texture_current

        self.fbo = gloo.FrameBuffer(color=self.texture_next)
        self.update()

if __name__ == '__main__':
    canvas = WaveCanvas()
    app.run()
