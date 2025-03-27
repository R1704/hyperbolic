import torch
from ..data.tree import ExpressionGenerator, ExpressionVisualizer
from ..embeddings.euclidean_embedding import EuclideanEmbedding
from ..embeddings.hyperbolic_embedding import HyperbolicEmbedding
from ..manifolds.poincare_manifold import PoincareManifold
from ..visualization.plotter import Plotter
from experiments.evaluate import compute_map, compute_distortion

def run_experiment():
    # Generate a set of random symbolic expressions.
    generator = ExpressionGenerator(max_depth=5, variables=['x', 'y'])
    expressions = [generator.generate_expression() for _ in range(50)]
    
    # Convert expressions to graphs (one per expression).
    visualizer = ExpressionVisualizer()
    graphs = [visualizer.expression_to_graph(expr) for expr in expressions]
    
    # For simplicity, assume we merge all graphs into one large graph,
    # or run experiments on each graph separately.
    # Here we use the first graph as an example.
    graph_data = graphs[0]
    
    # For node embeddings, assume each node is represented by a dummy feature.
    num_nodes = graph_data.num_nodes
    embedding_dim = 2  # For visualization, use 2D.
    
    # Compare Euclidean and Hyperbolic Embeddings:
    # Initialize models:
    euclidean_model = EuclideanEmbedding(num_nodes, embedding_dim)
    manifold = PoincareManifold(dim=embedding_dim, c=1.0)
    hyperbolic_model = HyperbolicEmbedding(num_nodes, embedding_dim, manifold)
    
    # Set up optimizers:
    optimizer_euc = torch.optim.Adam(euclidean_model.parameters(), lr=0.01)
    optimizer_hyp = torch.optim.Adam(hyperbolic_model.parameters(), lr=0.01)
    
    # Dummy training loop (replace loss() with your custom loss functions)
    for epoch in range(100):
        # Euclidean Training Step:
        optimizer_euc.zero_grad()
        embeddings_euc = euclidean_model.forward(graph_data)
        loss_euc = euclidean_model.loss(embeddings_euc, graph_data)
        loss_euc.backward()
        optimizer_euc.step()
        
        # Hyperbolic Training Step:
        optimizer_hyp.zero_grad()
        embeddings_hyp = hyperbolic_model.forward(graph_data)
        loss_hyp = hyperbolic_model.loss(embeddings_hyp, graph_data)
        loss_hyp.backward()
        optimizer_hyp.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Euclidean Loss: {loss_euc.item()}, Hyperbolic Loss: {loss_hyp.item()}")
    
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
