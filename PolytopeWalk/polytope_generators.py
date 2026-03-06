import math
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, lil_matrix, csr_array

###########################################################################################################
def generate_birkhoff(dim):
    d = int(math.floor(np.sqrt(dim)))
    k = d * d
    A = lil_matrix((2 * d - 1, d**2))
    b = np.ones(2 * d - 1)

    for i in range(1, d + 1):
        A[i - 1, (i - 1) * d : i * d] = 1
    for i in range(1, d):
        A[d + i - 1, i - 1 : d**2 : d] = 1

    x = np.ones(d * d) / d
    return x, A, b, 'birkhoff'

###########################################################################################################

def generate_simplex(dim):
    return np.array([1 / dim] * dim), np.array([[1] * dim]), np.array([1]), dim, 'simplex', dim


###########################################################################################################

def generate_cube(dim):
    dimension = dim
    dim = dim // 3
    col = np.concatenate((np.arange(0, dim), np.arange(0, dim), np.arange(dim, 3 * dim)))
    row = np.concatenate((np.arange(0, 2 * dim), np.arange(0, 2 * dim)))
    ones = np.ones(dim)
    negative_ones = np.repeat(-1, dim)
    data = np.concatenate((ones, negative_ones, ones, ones))
    A = sparse.csr_array((data, (row, col)), shape=(2 * dim, 3 * dim))
    b = np.array([[1]] * (2 * dim))
    b = b.flatten()
    x = np.concatenate((np.array([[0]] * (dim)), np.array([[1]] * (2 * dim))))
    return x, A, b, 2 * dim, 'hypercube', dimension

###########################################################################################################

def generate_hypercube(dim, angle_deg=0):
    """
    Generates a Hypercube that is rotated by 'angle_deg' across adjacent planes.
    This destroys the sparsity of the constraint matrix.
    
    Returns: Slack Form [ A_rotated | I ] * [x; s] = b
    """
    A_spatial = np.vstack([
        np.eye(dim),
        -np.eye(dim)
    ])
    
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    
    R = np.eye(dim)
    
    for k in range(dim - 1):
        R_k = np.eye(dim)
        R_k[k, k]     = c
        R_k[k, k+1]   = -s
        R_k[k+1, k]   = s
        R_k[k+1, k+1] = c
        
        R = R @ R_k

    A_rotated_spatial = A_spatial @ R
    A_slack = np.eye(2 * dim)
    A_total = np.hstack([A_rotated_spatial, A_slack])
    
    b = np.ones(2 * dim)
    x0 = np.zeros(dim)
    s0 = b.copy()
    
    init = np.concatenate([x0, s0])

    A_final = sparse.csr_matrix(A_total)
    
    return init, A_final, b, 2*dim, f'rotated_hypercube_{angle_deg}deg', dim


###########################################################################################################

def generate_rotated_hypercube_direct(dim, angle_deg=53):
    """
    Generates the Dense Geometric Form (Ax <= b) directly.
    We try to Skip the Slack/Facial Reduction complexity.
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    
    R = np.eye(dim)
    for k in range(dim - 1):
        R_k = np.eye(dim)
        R_k[k, k]     = c
        R_k[k, k+1]   = -s
        R_k[k+1, k]   = s
        R_k[k+1, k+1] = c
        R = R @ R_k
    
    I = np.eye(dim)
    normals = np.vstack([I, -I]) # Shape (2d, d)
    
    # Rotate the normals
    A_dense = normals @ R
    
    b_dense = np.ones(2 * dim)
    
    init = np.zeros(dim)
    
    return init, A_dense, b_dense, f'Hypercube (rotated by {angle_deg}deg)'

import numpy as np

import numpy as np


###########################################################################################################

def generate_rotated_simplex_direct(dim, angle_deg=53):
    """
    Generates the Rotated Solid Simplex (Ax <= b).
    
    Base definition:
      1. x >= 0  -> -x <= 0
      2. sum(x) <= 1 -> [1,1,...] @ x <= 1

    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    
    R = np.eye(dim)
    for k in range(dim - 1):
        R_k = np.eye(dim)
        R_k[k, k]     = c
        R_k[k, k+1]   = -s
        R_k[k+1, k]   = s
        R_k[k+1, k+1] = c
        R = R @ R_k

    neg_identity = -np.eye(dim)

    lid_normal = np.ones((1, dim))

    A_base = np.vstack([neg_identity, lid_normal])
    
    A_dense = A_base @ R

    b_dense = np.concatenate([np.zeros(dim), np.ones(1)])
    
    init = np.zeros(dim)
    
    return init, A_dense, b_dense, f'Solid Simplex (Rotated {angle_deg}deg)'


###########################################################################################################

def generate_birkhoff_direct(ambient_dim):
    n = int(np.sqrt(ambient_dim))
    if n * n != ambient_dim:
        raise ValueError(f"Dim must be square. Got {ambient_dim}.")

    intrinsic_dim = (n - 1) ** 2
    
    A_list = []
    b_list = []

    A_list.append(-np.eye(intrinsic_dim))
    b_list.append(np.zeros(intrinsic_dim))

    A_corner = -np.ones((1, intrinsic_dim))
    b_corner = np.array([-(n - 2)])
    A_list.append(A_corner)
    b_list.append(b_corner)

    A_cols = np.zeros((n - 1, intrinsic_dim))
    for j in range(n - 1):
        idx = np.arange(j, intrinsic_dim, n - 1)
        A_cols[j, idx] = 1.0
    A_list.append(A_cols)
    b_list.append(np.ones(n - 1))

    A_rows = np.zeros((n - 1, intrinsic_dim))
    for i in range(n - 1):
        A_rows[i, i * (n - 1):(i + 1) * (n - 1)] = 1.0
    A_list.append(A_rows)
    b_list.append(np.ones(n - 1))

    dense_A = np.vstack(A_list)
    dense_b = np.concatenate(b_list)

    init = np.full(intrinsic_dim, 1.0 / n)
    name = f"Birkhoff(n={n}, intrinsic_dim={(n - 1) ** 2})"

    return init, dense_A, dense_b, name

###########################################################################################################

def generate_birkhoff_equalities(n):
    dim = n * n
    num_constraints = 2 * n 
    A = lil_matrix((num_constraints, dim), dtype=np.float64)
    b = np.ones(num_constraints, dtype=np.float64)
    for r in range(n):
        for c in range(n):
            A[r, r * n + c] = 1.0
    for c in range(n):
        for r in range(n):
            A[n + c, r * n + c] = 1.0
    A_csr = A.tocsr()
    A_csr.indices = A_csr.indices.astype(np.int32)
    A_csr.indptr = A_csr.indptr.astype(np.int32)
    return A_csr, b

##################################################################################################

def generate_random_order_polytope(dim, m, seed=None):
    """
    Generates a random order polytope matching Volesti's logic.
    
    Args:
        dim: The dimension of the space.
        m: Total number of facets (must be >= 2 * dim).
        seed: Optional integer for reproducible random posets.
        
    Returns:
        init: A strictly interior starting point for MCMC.
        A: Inequality constraint matrix (m x dim).
        b: Inequality constraint vector (m).
        name: String identifier.
    """
    if m < 2 * dim:
        raise ValueError(f"m (facets) must be at least 2*dim ({2*dim}). Got {m}.")
        
    rng = np.random.default_rng(seed)
    
    # 1. Create and shuffle the order
    order = np.arange(dim)
    rng.shuffle(order)
    
    # 2. Initialize constraint matrices (Ax <= b)
    A = np.zeros((m, dim))
    b = np.zeros(m)
    
    # 3. Ambient Bounding Box constraints: 0 <= x_i <= 1
    # First `dim` rows: x_i <= 1
    for i in range(dim):
        A[i, i] = 1.0
        b[i] = 1.0
        
    # Next `dim` rows: -x_i <= 0 (which means x_i >= 0)
    for i in range(dim):
        A[dim + i, i] = -1.0
        b[dim + i] = 0.0
        
    # 4. Random Relational Constraints: x_u <= x_v
    num_relations = m - 2 * dim
    for k in range(num_relations):
        # Sample two distinct indices
        idx = rng.choice(dim, 2, replace=False)
        x, y = np.min(idx), np.max(idx) # Ensure x < y
        
        u = order[x]
        v = order[y]
        
        # Enforce x_u - x_v <= 0
        row = 2 * dim + k
        A[row, u] = 1.0
        A[row, v] = -1.0
        b[row] = 0.0
        
    # 5. Generate a strictly interior initial point
    # Assigns values evenly spaced between 0 and 1 ensuring x_u < x_v
    init = np.zeros(dim)
    for i in range(dim):
        init[order[i]] = (i + 1.0) / (dim + 1.0)
        
    name = f"OrderPolytope(dim={dim}, facets={m})"
    
    return init, A, b, name

############################################################################################

def generate_orderpoly_sparse(dim, m, seed=None):
    if m < 2 * dim:
        raise ValueError(f"m (facets) must be at least 2*dim ({2*dim}). Got {m}.")
        
    rng = np.random.default_rng(seed)
    
    # 1. Create and shuffle the order
    order = np.arange(dim)
    rng.shuffle(order)
    
    # 2. Initialize sparse matrix using lil_matrix
    num_constraints = m
    A = lil_matrix((num_constraints, dim), dtype=np.float64)
    b = np.zeros(num_constraints, dtype=np.float64)
    
    # 3. Ambient Bounding Box constraints: 0 <= x_i <= 1
    for i in range(dim):
        # x_i <= 1
        A[i, i] = 1.0
        b[i] = 1.0
        
        # -x_i <= 0 (equivalent to x_i >= 0)
        A[dim + i, i] = -1.0
        b[dim + i] = 0.0
        
    # 4. Random Relational Constraints: x_u <= x_v
    num_relations = m - 2 * dim
    for k in range(num_relations):
        # Sample two distinct indices
        idx = rng.choice(dim, 2, replace=False)
        x, y = np.min(idx), np.max(idx) # Ensure x < y
        
        u = order[x]
        v = order[y]
        
        # Enforce x_u - x_v <= 0
        row = 2 * dim + k
        A[row, u] = 1.0
        A[row, v] = -1.0
        b[row] = 0.0
        
    # 5. Convert to CSR and enforce int32 exactly like your template
    A_csr = A.tocsr()
    A_csr.indices = A_csr.indices.astype(np.int32)
    A_csr.indptr = A_csr.indptr.astype(np.int32)
    
    return A_csr, b