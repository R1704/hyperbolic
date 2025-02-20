import numpy as np 
from vispy import app, gloo
import sys
try:
    from PyQt6 import QtGui, QtCore
except ImportError:
    QtGui, QtCore = None, None




# Define some helper functions for the Poincare disk model.
def get_unit_circle(n_segments=100):
    theta = np.linspace(0, 2*np.pi, n_segments, endpoint=True)
    unit_circle = np.column_stack((np.cos(theta), np.sin(theta))).astype(np.float32) * 0.5
    unit_circle = np.concatenate([unit_circle, unit_circle[0:1]], axis=0)  # Close loop 
    return unit_circle

def mobius_transform(z, a):
    """Apply a Mobius transformation to map z to the origin."""
    return (z - a) / (1 - np.conj(a) * z)

def inverse_mobius_transform(z, a):
    """Apply a Mobius transformation to map the origin to z."""
    return (a + z) / (1 + np.conj(a) * z)

def get_arc(z):
    """Compute the geodesic circle through the origin and z."""
    t = np.linspace(0, 1, 1000)
    return t * z

def hyperbolic_isometry(z, t):
    # Example: a rotation by t radians.
    return np.exp(1j * t) * z

def get_geodesic(z1, z2):
    z2_transformed = mobius_transform(z2, z1)
    geodesic_transformed = get_arc(z2_transformed)
    geodesic_original = inverse_mobius_transform(geodesic_transformed, z1)
    return geodesic_original

def distance(z1, z2):
    return np.arccosh(1 + 2 * abs(z1 - z2)**2 / ((1 - abs(z1)**2) * (1 - abs(z2)**2)))

def circle_inversion(z, c, R):
    return c + (R**2) / (np.conj(z - c))



# Define the vertex and fragment shaders.
vertex = """
attribute vec2 a_position;
void main(void) {
    gl_Position = vec4(a_position, 0.0, 1.0);
    gl_PointSize = 10.0;
}
"""

fragment = """
uniform vec4 u_color;
void main(void) {
    gl_FragColor = u_color;
}
"""



# Define the canvas class.
class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, title='Poincare Disk', size=(800, 800), keys='interactive')
        self.program = gloo.Program(vertex, fragment)
        
        self.unit_circle = get_unit_circle()
        self.points = []          # Store clicked endpoints.
        self.geodesic = None      # Geodesic arc between two points.
        self.inversions = []      # Store inverted points.
        self.active_point_index = None  # Track point being dragged.

        self.program['u_color'] = (1.0, 1.0, 1.0, 1.0)

        # Start the timer for dynamic updates.
        self.t = 0
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

    def on_draw(self, event):
        gloo.clear()
        # Draw unit disc boundary.
        self.program['a_position'] = self.unit_circle
        self.program['u_color'] = (1.0, 1.0, 1.0, 1.0)
        self.program.draw('line_strip')
        
        # If a geodesic exists, draw it in red.
        if self.geodesic is not None:
            self.program['u_color'] = (1.0, 0.0, 0.0, 1.0)
            self.program['a_position'] = self.geodesic
            self.program.draw('line_strip')

        # Draw the inverted points (displayed in blue).
        if self.inversions is not None:
            self.program['u_color'] = (0.0, 0.0, 1.0, 1.0)
            self.program['a_position'] = self.inversions
            self.program.draw('points')
        
        # Draw the clicked points (displayed in green).
        if self.points:
            pts = np.array(self.points, dtype=np.float32)
            self.program['u_color'] = (0.0, 1.0, 0.0, 1.0)
            self.program['a_position'] = pts
            self.program.draw('points')
        
        # If two points exist, compute and display the hyperbolic distance.
        self.display_distance_between_points()

    def display_distance_between_points(self):
        if len(self.points) == 2 and QtGui is not None:
            (x1, y1), (x2, y2) = self.points
            d = distance(complex(x1, y1), complex(x2, y2))
            text = f"Distance: {d:.2f}"
            
            painter = QtGui.QPainter(self.native)
            painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.white))
            painter.setFont(QtGui.QFont("Helvetica", 16))
            padding = 20
            painter.drawText(self.size[0] - 150 - padding, self.size[1] - padding, text)
            painter.end()
    
    def on_mouse_press(self, event):
        # Convert pixel coordinates to normalized (-1, 1) with y flipped.
        width, height = self.size
        x = (event.pos[0] / width) * 2 - 1
        y = 1 - (event.pos[1] / height) * 2  # Flip y axis.
        threshold = 0.1  # Define a threshold for selecting an existing point.
        self.active_point_index = None

        # Check if click is near an existing point.
        for i, (px, py) in enumerate(self.points):
            if ((x - px)**2 + (y - py)**2)**0.5 < threshold:
                self.active_point_index = i
                break

        # If not clicking near a point, add a new point.
        if self.active_point_index is None:
            self.points.append((x, y))
            self.active_point_index = len(self.points) - 1
        
        # Update geodesic
        self.geodesic = self.calculate_polygon_segments()
        
        # Inversions
        self.inversions = self.calculate_inversions()
        
        self.update()

    def calculate_inversions(self):
        inversions = []
        for p in self.geodesic:
            z = complex(*p)
            z_inverted = circle_inversion(z, 0, 0.5)
            inversions.append((z_inverted.real, z_inverted.imag))
        inversions = np.array(inversions, dtype=np.float32)
        return inversions
    
    def on_mouse_move(self, event):
        # Only update if a point is active.
        if self.active_point_index is not None:
            width, height = self.size
            x = (event.pos[0] / width) * 2 - 1
            y = 1 - (event.pos[1] / height) * 2  # Flip y axis.
            self.points[self.active_point_index] = (x, y)
            if len(self.points) >= 2:
                self.geodesic = self.calculate_polygon_segments()

            self.update()

    def calculate_polygon_segments(self):
        polygon_segments = []
        for i in range(len(self.points)):
            j = (i + 1) % len(self.points)  # Wrap-around for closed polygon.
            z1 = complex(*self.points[i])
            z2 = complex(*self.points[j])
            segment = get_geodesic(z1, z2)
            polygon_segments.append(np.column_stack((segment.real, segment.imag)).astype(np.float32))
        return np.concatenate(polygon_segments, axis=0)

    def on_mouse_release(self, event):
        # Reset active point when mouse is released.
        self.active_point_index = None

    def on_timer(self, event):
        self.t = event.dt * 0.1  # Use elapsed time as parameter.
        transformed_points = []
        for pt in self.points:
            z = complex(*pt)
            z_transformed = hyperbolic_isometry(z, self.t)
            transformed_points.append((z_transformed.real, z_transformed.imag))
        self.points = transformed_points
        self.geodesic = self.calculate_polygon_segments()
        self.inversions = self.calculate_inversions()
        self.update()
        
if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas = Canvas()
    canvas.show()
    app.run()