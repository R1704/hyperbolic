# visualization/plotter.py

import matplotlib.pyplot as plt

class Plotter:
    """
    A class for visualizing 2D embeddings.
    """
    def __init__(self, title="Embedding Visualization"):
        self.title = title

    def plot_embeddings(self, embeddings, labels=None):
        plt.figure(figsize=(8, 8))
        x, y = embeddings[:, 0], embeddings[:, 1]
        if labels is not None:
            scatter = plt.scatter(x, y, c=labels, cmap="viridis", s=50)
            plt.colorbar(scatter)
        else:
            plt.scatter(x, y, s=50)
        plt.title(self.title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()
