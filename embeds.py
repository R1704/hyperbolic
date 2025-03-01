import numpy as np 
import torch
from torch_geometric.data import Data


def g(x):
    return 2 * np.exp(-x) / (1 + np.exp(-x))

def d(u_i, u_j, dist_func='euclidean'):
    match dist_func:
        case 'euclidean':
            return euclidean_distance(u_i, u_j)
        case 'cosine':
            return cosine_similarity(u_i, u_j)
        case 'poincare':
            return poincare_distance(u_i, u_j)

def euclidean_distance(u_i, u_j):
    return np.sqrt(np.sum((u_i - u_j)**2))

def cosine_similarity(u_i, u_j):
    return np.dot(u_i, u_j) / (np.sqrt(u_i.dot(u_i)) * np.sqrt(u_j.dot(u_j)))

def poincare_distance(u, v):
    """Distance in Poincaré disk model of hyperbolic space"""
    u_norm = np.sum(u**2)
    v_norm = np.sum(v**2)
    if u_norm >= 1 or v_norm >= 1:
        return float('inf')
    
    euclidean_sq = np.sum((u - v)**2)
    
    numerator = 2 * euclidean_sq
    denominator = (1 - u_norm) * (1 - v_norm)
    return np.arccosh(1 + numerator / denominator)

def J(U, adj_matrix, l=1, dist_func='euclidean'):
    """
    Loss function for embeddings optimization
    U: numpy array of shape (num_nodes, embedding_dim) containing node embeddings
    adj_matrix: adjacency matrix of shape (num_nodes, num_nodes) - represents graph structure
    l: regularization parameter
    """
    n = U.shape[0]  # Number of nodes
    s1 = 0
    s2 = 0
    
    for i in range(n):
        for j in range(n):
            if i != j:  # Avoid computing for same node
                # Compare graph adjacency with embedding distance
                s1 += (adj_matrix[i, j] - g(d(U[i], U[j], dist_func)))**2
                # Regularization term to keep embeddings close
                s2 += l * d(U[i], U[j], dist_func)
    
    return s1 + s2

# Convert edge_index to adjacency matrix
def create_adj_matrix(edge_index, num_nodes):
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_matrix[src, dst] = 1
    return adj_matrix


from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')

# visualize dataset with networkx
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
edge_index = dataset[0].edge_index
import plotly.graph_objects as go
import networkx as nx
import plotly.graph_objects as go
import networkx as nx


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

# Use it with the Cora dataset
num_nodes = dataset[0].num_nodes
print(dataset[0])

def analyze_graph_components(edge_index, num_nodes):
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i)
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(src, dst)
    
    components = list(nx.connected_components(G))
    print(f"Graph has {len(components)} connected components")
    
    # Print size of largest components
    sizes = [len(c) for c in components]
    sizes.sort(reverse=True)
    print(f"Largest component sizes: {sizes[:5]}")
    
    return components

components = analyze_graph_components(dataset[0].edge_index, dataset[0].num_nodes)
# visualize_dataset_interactive(dataset[0].edge_index, num_nodes)












import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')


# Extract embeddings from first layer of trained GCN
def get_node_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        x, edge_index = data.x, data.edge_index
        embeddings = model.conv1(x, edge_index)
        embeddings = F.relu(embeddings)
    return embeddings.cpu().numpy()

# Get embeddings
node_embeddings = get_node_embeddings(model, data)

# Reduce to 2D for visualization (if needed)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(node_embeddings)

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

# Visualize
visualize_learned_embeddings(embeddings_2d, data.edge_index, data.y)















# # Example usage
# num_nodes = 5  # Based on your data
# adj_matrix = create_adj_matrix(edge_index, num_nodes)

# def initialize_embeddings(num_nodes, embedding_dim=2):
#     """Initialize embeddings randomly"""
#     # Small random values work better initially
#     return np.random.normal(0, 0.1, (num_nodes, embedding_dim))

# def train_embeddings(adj_matrix, embedding_dim=2, lr=0.01, epochs=1000, dist_func='euclidean'):
#     """Train node embeddings using gradient descent"""
#     num_nodes = adj_matrix.shape[0]
#     U = initialize_embeddings(num_nodes, embedding_dim)
    
#     losses = []
#     for epoch in range(epochs):
#         # Simple numerical gradient computation
#         grad = np.zeros_like(U)
#         eps = 1e-5
        
#         # Compute initial loss
#         loss = J(U, adj_matrix, dist_func=dist_func)
        
#         # Compute gradient for each element
#         for i in range(num_nodes):
#             for d in range(embedding_dim):
#                 U[i, d] += eps
#                 loss_plus = J(U, adj_matrix, dist_func=dist_func)
#                 U[i, d] -= 2*eps
#                 loss_minus = J(U, adj_matrix, dist_func=dist_func)
#                 U[i, d] += eps  # Restore original value
                
#                 # Central difference approximation of gradient
#                 grad[i, d] = (loss_plus - loss_minus) / (2*eps)
        
#         # Update embeddings
#         U -= lr * grad
        
#         # Track loss
#         if epoch % 100 == 0:
#             current_loss = J(U, adj_matrix, dist_func=dist_func)
#             losses.append(current_loss)
#             print(f"Epoch {epoch}, Loss: {current_loss:.4f}")
    
#     return U, losses


# def visualize_embeddings(U, adj_matrix):
#     """Visualize embeddings using networkx"""
#     import matplotlib.pyplot as plt
#     import networkx as nx
    
#     # Create a directed graph
#     G = nx.DiGraph()
    
#     # Add nodes with positions from embeddings
#     for i in range(len(U)):
#         G.add_node(i, pos=(U[i, 0], U[i, 1]))
    
#     # Add edges
#     for i in range(len(U)):
#         for j in range(len(U)):
#             if adj_matrix[i, j] > 0:
#                 G.add_edge(i, j)
    
#     plt.figure(figsize=(8, 6))
#     pos = nx.get_node_attributes(G, 'pos')
    
#     nx.draw(G, pos, with_labels=True, node_size=500, 
#             node_color='lightblue', font_weight='bold',
#             arrowsize=15, width=1.5)
    
#     plt.title("2D Node Embeddings")
#     plt.show()

# # Example usage
# num_nodes = 3
# embedding_dim = 2
# adj_matrix = create_adj_matrix(data.edge_index, num_nodes)

# # Train the embeddings
# embeddings, losses = train_embeddings(adj_matrix, embedding_dim=embedding_dim, 
#                                      lr=0.05, epochs=1000)

# print("Final embeddings:")
# print(embeddings)

# # Visualize if 2D
# if embedding_dim == 2:
#     visualize_embeddings(embeddings, adj_matrix)