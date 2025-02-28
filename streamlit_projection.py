import streamlit as st
import numpy as np
from PIL import Image
import time
import base64
from io import BytesIO
import plotly.graph_objects as go

# Import just what we need from projection.py
from projection import get_unit_circle, get_geodesic, circle_inversion, radius

def get_image_base64(img):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def create_poincare_visualization(rotation=0.0, tilt=0.0, scale=1.0):
    """Create a Plotly figure with Poincaré disc model"""
    # Create the figure
    fig = go.Figure()
    
    # Parameters
    resolution = 100
    
    # Unit circle
    circle = get_unit_circle(resolution)
    
    # Add the unit circle
    fig.add_trace(go.Scatter(
        x=circle[:, 0], 
        y=circle[:, 1],
        mode='lines',
        line=dict(color='black', width=2),
        name="Unit Circle"
    ))
    
    # Add grid lines for both Euclidean and hyperbolic
    num_lines = 10
    grid_color = 'rgba(100, 100, 100, 0.3)'
    
    # Euclidean grid
    for i in range(-num_lines, num_lines+1, 2):
        x = i/num_lines
        if abs(x) < 1.0:  # Only draw within the unit circle
            # Horizontal line
            y_vals = np.linspace(-np.sqrt(1-x**2), np.sqrt(1-x**2), 100)
            x_vals = np.full_like(y_vals, x)
            
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines',
                line=dict(color=grid_color, width=1),
                showlegend=False
            ))
            
            # Vertical line
            x_vals = np.linspace(-np.sqrt(1-x**2), np.sqrt(1-x**2), 100)
            y_vals = np.full_like(x_vals, x)
            
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines',
                line=dict(color=grid_color, width=1),
                showlegend=False
            ))
    
    # Hyperbolic geodesics (circles orthogonal to unit circle)
    for i in range(1, num_lines+1, 2):
        # Create geodesics at different positions
        x = i / (num_lines * 1.5)
        if x < 0.9:  # Avoid getting too close to the boundary
            points = get_geodesic(np.array([x, 0]), resolution=50)
            fig.add_trace(go.Scatter(
                x=points[:, 0], y=points[:, 1],
                mode='lines',
                line=dict(color='rgba(200, 100, 100, 0.5)', width=1),
                showlegend=False
            ))
    
    # Setup layout
    fig.update_layout(
        width=800, 
        height=800,
        plot_bgcolor='white',
        xaxis=dict(
            range=[-1.1, 1.1],
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[-1.1, 1.1],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    # Apply rotation transformation to the figure
    for trace in fig.data:
        x = np.array(trace['x'])
        y = np.array(trace['y'])
        
        # Apply rotation
        c, s = np.cos(rotation), np.sin(rotation)
        x_new = x * c - y * s
        y_new = x * s + y * c
        
        # Apply tilt (skew)
        y_new += tilt * x_new
        
        # Apply scale
        x_new *= scale
        y_new *= scale
        
        trace.update(x=x_new, y=y_new)
    
    return fig

def main():
    st.title("Hyperbolic Projection")
    
    st.sidebar.header("Controls")
    
    # Setup session state
    if "rotation" not in st.session_state:
        st.session_state.rotation = 0.0
    if "tilt" not in st.session_state:
        st.session_state.tilt = 0.0
    if "scale" not in st.session_state:
        st.session_state.scale = 1.0
    
    # Create controls
    rotation_speed = st.sidebar.slider("Rotation Speed", 0.0, 0.05, 0.005, 0.001)
    auto_rotate = st.sidebar.checkbox("Auto Rotate", value=True)
    
    # Manual controls
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Rotate Left"):
            st.session_state.rotation -= 0.1
    with col2:
        if st.button("Rotate Right"):
            st.session_state.rotation += 0.1
    
    # Scale control
    scale = st.sidebar.slider("Scale", 0.5, 2.0, 1.0, 0.1)
    st.session_state.scale = scale
    
    # Tilt control
    tilt = st.sidebar.slider("Tilt", -0.5, 0.5, 0.0, 0.05)
    st.session_state.tilt = tilt
    
    # Update rotation if auto-rotate is enabled
    if auto_rotate:
        st.session_state.rotation += rotation_speed
    
    # Create and display the visualization
    fig = create_poincare_visualization(
        rotation=st.session_state.rotation,
        tilt=st.session_state.tilt,
        scale=st.session_state.scale
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Auto-rerun for animation
    if auto_rotate:
        time.sleep(0.05)
        st.rerun()

if __name__ == "__main__":
    main()