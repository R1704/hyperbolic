import torch
from torch_geometric.utils import from_networkx, to_dense_adj
from hyperbolic_graph_embedding.data.tree import ExpressionGenerator, ExpressionVisualizer
from hyperbolic_graph_embedding.embeddings.euclidean_embedding import EuclideanEmbedding
from hyperbolic_graph_embedding.embeddings.hyperbolic_embedding import HyperbolicEmbedding
from hyperbolic_graph_embedding.manifolds.poincare_manifold import PoincareManifold
from hyperbolic_graph_embedding.visualization.plotter import Plotter
from hyperbolic_graph_embedding.experiments.evaluate import compute_map, compute_distortion

def run_experiment():
    # Generate a set of random symbolic expressions.
    generator = ExpressionGenerator(max_depth=5, variables=['x', 'y'])
    expressions = [generator.generate_expression() for _ in range(50)]
    
    # Convert expressions to graphs (one per expression).
    visualizer = ExpressionVisualizer()
    networkx_graphs = [visualizer.expression_to_graph(expr) for expr in expressions]

    # Convert NetworkX graph to PyTorch Geometric Data
    # Add dummy features if not present
    for g in networkx_graphs:
        for node in g.nodes():
            if 'feat' not in g.nodes[node]:
                g.nodes[node]['feat'] = torch.ones(1)  # Dummy feature
    
    # Convert first graph to PyG data
    graph_data = from_networkx(networkx_graphs[0])
    
    # Extract adjacency matrix from graph_data
    adj_matrix = to_dense_adj(graph_data.edge_index)[0]  # Get the first item since to_dense_adj returns a batch
    
    num_nodes = graph_data.num_nodes
    embedding_dim = 2  # For visualization, use 2D.
    
    # Initialize models
    euclidean_model = EuclideanEmbedding(num_nodes, embedding_dim)
    manifold = PoincareManifold(dim=embedding_dim, c=1.0)
    hyperbolic_model = HyperbolicEmbedding(num_nodes, embedding_dim, manifold)
    
    # Set up optimizers
    optimizer_euc = torch.optim.Adam(euclidean_model.parameters(), lr=0.01)
    optimizer_hyp = torch.optim.Adam(hyperbolic_model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(100):
        # Euclidean Training Step
        optimizer_euc.zero_grad()
        embeddings_euc = euclidean_model.forward(graph_data)
        # Pass the adjacency matrix instead of graph_data
        loss_euc = euclidean_model.loss(embeddings_euc, adj_matrix)
        loss_euc.backward()
        optimizer_euc.step()
        
        # Hyperbolic Training Step
        optimizer_hyp.zero_grad()
        embeddings_hyp = hyperbolic_model.forward(graph_data)
        # Pass the adjacency matrix instead of graph_data
        loss_hyp = hyperbolic_model.loss(embeddings_hyp, adj_matrix)
        loss_hyp.backward()
        optimizer_hyp.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Euclidean Loss: {loss_euc.item()}, Hyperbolic Loss: {loss_hyp.item()}")
    
    # Rest of your code remains the same    
    # Evaluate embeddings using your evaluation metrics:
    map_euc = compute_map(embeddings_euc, ground_truth=None)
    distortion_euc = compute_distortion(embeddings_euc, distances=None)
    map_hyp = compute_map(embeddings_hyp, ground_truth=None)
    distortion_hyp = compute_distortion(embeddings_hyp, distances=None)
    print("Euclidean MAP:", map_euc, "Distortion:", distortion_euc)
    print("Hyperbolic MAP:", map_hyp, "Distortion:", distortion_hyp)
    
    # Visualization (plot both embeddings):
    
    plotter_euc = Plotter(title="Euclidean Embeddings")
    plotter_hyp = Plotter(title="Hyperbolic Embeddings")
    plotter_euc.plot_embeddings(embeddings_euc.detach().cpu().numpy())
    plotter_hyp.plot_embeddings(embeddings_hyp.detach().cpu().numpy())

if __name__ == "__main__":
    run_experiment()
