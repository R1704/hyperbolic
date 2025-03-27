import networkx as nx
import numpy as np 
import torch
import torch.nn.functional as F


# Convert edge_index to adjacency matrix
def create_adj_matrix(edge_index, num_nodes):
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_matrix[src, dst] = 1
    return adj_matrix

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


# Extract embeddings from first layer of trained GCN
def get_node_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        x, edge_index = data.x, data.edge_index
        embeddings = model.conv1(x, edge_index)
        embeddings = F.relu(embeddings)
    return embeddings.cpu().numpy()
