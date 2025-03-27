import numpy as np 


# Define the function g(x) = 2 * exp(-x) / (1 + exp(-x))
# 2 because then g(0) = 1
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
    for i in range(n):
        # Regularization term to keep embeddings close
        s2 += l * np.norm(U[i])
    
    return s1 + s2


def loss_hyp():
    pass

