import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def visualize_dataset_interactive(edge_index, num_nodes):
    # Create the graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)
    
    # Add edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(src, dst)
    
    # Generate positions
    pos = nx.spring_layout(G, seed=42)
    
    # Extract node positions
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]
    
    # Create edges for visualization
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='lightblue',
            size=8,
            line=dict(width=1, color='#888')))
    
    # Add node labels as hover text
    node_labels = list(G.nodes())
    node_trace.text = node_labels
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Interactive Cora Dataset',
                       showlegend=False,
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       hovermode='closest',
                       dragmode='pan'))  # Enable panning by default
    
    # Show figure with interactive controls enabled
    fig.show(config={
        'scrollZoom': True,         # Enable scroll/wheel zooming
        'displayModeBar': True,     # Show the modebar
        'editable': True,           # Allow editing
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],  # Add drawing tools
    })


# Now visualize these embeddings
def visualize_learned_embeddings(embeddings_2d, edge_index, labels=None):
    # Create a graph
    G = nx.Graph()
    
    # Add nodes with positions from embeddings
    for i in range(len(embeddings_2d)):
        G.add_node(i, pos=(embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    # Add edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(src, dst)
    
    # Draw with these learned positions
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_color='lightblue', with_labels=False)
    plt.title("GCN Learned Embeddings")
    plt.show()
