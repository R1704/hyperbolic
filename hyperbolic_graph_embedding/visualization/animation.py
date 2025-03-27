# visualization/animation.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class EmbeddingAnimator:
    """
    Animate the evolution of embeddings over training epochs.
    """
    def __init__(self, embeddings_history):
        self.embeddings_history = embeddings_history

    def animate(self, interval=200):
        fig, ax = plt.subplots(figsize=(8, 8))
        scat = ax.scatter([], [])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_title("Embedding Evolution")

        def init():
            scat.set_offsets([])
            return scat,

        def update(frame):
            data = self.embeddings_history[frame]
            scat.set_offsets(data)
            ax.set_title(f"Epoch: {frame}")
            return scat,

        ani = animation.FuncAnimation(fig, update, frames=len(self.embeddings_history),
                                      init_func=init, interval=interval, blit=True)
        plt.show()
