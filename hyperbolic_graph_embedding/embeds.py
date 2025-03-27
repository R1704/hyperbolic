import numpy as np
from torch_geometric.datasets import Planetoid
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from utils import analyze_graph_components, get_node_embeddings
from hyperbolic_graph_embedding.visualization import visualize_dataset_interactive

# visualize dataset with networkx

dataset = Planetoid(root='/tmp/Cora', name='Cora')
G = nx.Graph()
edge_index = dataset[0].edge_index


# Use it with the Cora dataset
num_nodes = dataset[0].num_nodes
print(dataset[0])

components = analyze_graph_components(dataset[0].edge_index, dataset[0].num_nodes)
visualize_dataset_interactive(dataset[0].edge_index, num_nodes)















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