
import os
import platform
from time import time, sleep
import numpy as np
# Remove notebook-specific imports for standalone script
# from tqdm.notebook import trange, tqdm
from tqdm import trange, tqdm
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

# environment variables for JAX must be set before running any JAX code
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
if platform.system() == "Linux" and 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.environ['SLURM_JOB_CPUS_PER_NODE']}"
import jax
print(jax.devices())
import jax.numpy as jnp
from jax import grad, vmap
from jax.scipy.stats import norm
import equinox as eqx
import jax.nn as jnn
import jax.sharding as jsh
from jax.experimental import checkify
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.pjit import pjit

# Remove dependencies on external modules for standalone script
# from plots import *
# from decoders import *

# Add placeholder for missing functionality if needed
def map_estimate_from_particles(particles, logp_fn):
    """Find MAP estimate from particles"""
    log_probs = jnp.array([logp_fn(p) for p in particles])
    best_idx = jnp.argmax(log_probs)
    return particles[best_idx], log_probs[best_idx]

def estimate_hdr(particles, logp_fn, alpha=0.95):
    """Estimate highest density region"""
    log_probs = jnp.array([logp_fn(p) for p in particles])
    sorted_indices = jnp.argsort(-log_probs)
    n_hdr = int(len(particles) * alpha)
    threshold = log_probs[sorted_indices[n_hdr-1]]
    hdr_mask = log_probs >= threshold
    return particles[hdr_mask], threshold

# matplotlib inline equivalent for script
# %matplotlib inline
# %config InlineBackend.figure_format='retina'

trange = partial(trange, bar_format="{bar}", leave=False)
tqdm = partial(tqdm, bar_format="{bar}", leave=False)

# "iridis" color map (viridis without the deep purple)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
iridis = truncate_colormap(plt.get_cmap('viridis'), 0.2, 1)

#jax.config.update('jax_num_cpu_devices', 8)

def calculate_param_dim(k, m=2):
    """Calculate total parameter dimension for k-dimensional PTD with m states"""
    # m initial state parameters + m*m transition parameters + k*m absorption parameters
    return m + m * m + k * m

@jax.jit
def example_ptd_spec(params, k=1, m=2):
    """
    Generalized PTD specification for k-dimensional absorption
    
    Args:
        params: flattened parameter vector of length calculate_param_dim(k, m)
        k: number of absorption dimensions
        m: number of transient states
    
    Returns:
        alpha, T, t_vecs for k-dimensional PTD
    """
    # Parse parameters
    # First m parameters: initial distribution (softmax normalized)
    alpha_logits = params[:m]
    alpha = jax.nn.softmax(alpha_logits)
    
    # Next m*m parameters: transition matrix (each row softmax normalized)
    T_logits = params[m:m+m*m].reshape((m, m))
    T_unnorm = jax.nn.softmax(T_logits, axis=1)
    
    # Last k*m parameters: absorption vectors (softmax normalized)
    t_logits = params[m+m*m:].reshape((k, m))
    t_vecs_unnorm = jax.nn.softmax(t_logits, axis=1)
    
    # Ensure stochasticity: T + sum(t_vecs) has row sums <= 1
    total_exit_rates = jnp.sum(t_vecs_unnorm, axis=0)  # Sum over k dimensions for each state
    normalization = 1.0 / (T_unnorm.sum(axis=1) + total_exit_rates + 1e-8)
    
    # Normalize to ensure row sums are valid
    T = T_unnorm * normalization[:, None]
    t_vecs = [t_vecs_unnorm[i] * normalization for i in range(k)]
    
    return alpha, T, t_vecs


def simulate_example_data(params, k=1, m=2, samples=1000, key=None):
    """
    Simulate k-dimensional PTD data
    
    Args:
        params: parameter vector
        k: dimension of absorption vectors
        m: number of transient states
        samples: number of samples to generate
        key: JAX random key
    
    Returns:
        Array of shape (samples, k) containing absorption times
    """
    if key is None:
        key = jax.random.key(int(time() * 1e6))

    alpha, T, t_vecs = example_ptd_spec(params, k, m)

    m = T.shape[0]

    k = len(t_vecs)
    t_vecs_stacked = jnp.stack(t_vecs, axis=0)  # shape (k, m)

    def single_sample(key):
        key, subkey = jax.random.split(key)
        state = jax.random.choice(subkey, m, p=alpha)
        t_vec = jnp.zeros(k, dtype=int)

        def cond_fn(carry):
            _, _, _, absorbed = carry
            return ~absorbed

        def body_fn(carry):
            key, state, t, _ = carry
            key, *subkeys = jax.random.split(key, num=k + 2)

            draws = jnp.array([
                jax.random.uniform(subkeys[i]) < t_vecs_stacked[i, state]
                for i in range(k)
            ])
            absorbed = jnp.any(draws)

            # update discrete events if not absorbed
            t_new = jax.lax.cond(absorbed, lambda t_: t_, lambda t_: t_ + 1, t)

            state_new = jax.lax.cond(
                absorbed,
                lambda s: s,
                lambda s: jax.random.choice(subkeys[-1], m, p=T[s]),
                operand=state
            )

            return key, state_new, t_new, absorbed

        init_carry = (key, state, jnp.zeros(k, dtype=int), False)
        _, _, t_final, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        return t_final

    keys = jax.random.split(key, samples)
    return jax.vmap(single_sample)(keys)

@jax.jit
def unpack_theta(theta, k=1, m=2):
    """Unpack parameter vector into structured format"""
    alpha_logits = theta[:m]
    T_logits = theta[m:m+m*m].reshape((m, m))
    t_logits = theta[m+m*m:].reshape((k, m))
    return alpha_logits, T_logits, t_logits

@jax.jit
def log_pmf_dph(z, params, k=1, m=2):
    """
    Log PMF for k-dimensional discrete phase-type distribution
    
    Args:
        z: observation vector of length k
        params: parameter vector
        k: dimension of the distribution
        m: number of transient states
    
    Returns:
        Log probability of observing z
    """
    alpha, T, t_vecs = example_ptd_spec(params, k, m)

    # Use fixed upper bound
    z_max = 50
    t_mat = jnp.stack(t_vecs, axis=1)  # shape (m, k)

    def body_fun(i, carry):
        Tk, probs = carry
        out = alpha @ Tk @ t_mat
        Tk_next = Tk @ T
        return Tk_next, probs.at[i].set(out)

    Tk0 = jnp.eye(m)
    probs_init = jnp.zeros((z_max, k))
    _, probs_filled = jax.lax.fori_loop(0, z_max, body_fun, (Tk0, probs_init))

    # Ensure z is properly shaped and clipped
    z_clipped = jnp.clip(z.astype(int), 0, z_max - 1)
    
    # Pad z_clipped to have exactly k elements
    z_padded = jnp.pad(z_clipped, (0, max(0, k - len(z_clipped))))[:k]
    
    # Use advanced indexing with static shapes
    indices = jnp.arange(k)
    prob_at_z = probs_filled[z_padded, indices]
    logp = jnp.sum(jnp.log(jnp.maximum(prob_at_z, 1e-12)))
    
    return logp

# Use adaptive kernel bandwidth
@jax.jit
def median_heuristic(particles):
    """Compute bandwidth using median heuristic"""
    pairwise_dist = jnp.sum((particles[:, None, :] - particles[None, :, :])**2, axis=-1)
    h = jnp.median(pairwise_dist) / (2 * jnp.log(particles.shape[0] + 1))
    return h

# Local bandwidth adjustment: For highly multimodal distributions, consider
# using particle-specific bandwidths that adapt to local density:
@jax.jit
def local_adaptive_bandwidth(particles, k=5):
    """Compute local bandwidth for each particle based on k nearest neighbors"""
    # Compute pairwise distances
    pairwise_dist = jnp.sum((particles[:, None, :] - particles[None, :, :])**2, axis=-1)
    
    # For each particle, find distance to k-th nearest neighbor
    sorted_dist = jnp.sort(pairwise_dist, axis=1)
    kth_neighbor_dist = sorted_dist[:, k]

    # Set bandwidth proportional to distance to k-th neighbor
    local_h = kth_neighbor_dist / (2 * jnp.log(k + 1))
    return local_h

# Batch-based bandwidth estimation: For very large datasets, compute the
# bandwidth using a random subset of the data to improve computational
# efficiency:
@jax.jit
def batch_median_heuristic(particles, max_particles=1000):
    """Compute bandwidth using a subset of particles for large sample sizes"""
    if len(particles) > max_particles:
        idx = np.random.choice(len(particles), max_particles, replace=False)
        particles_subset = particles[idx]
    else:
        particles_subset = particles
        
    pairwise_dist = jnp.sum((particles_subset[:, None, :] - particles_subset[None, :, :])**2, axis=-1)
    h = jnp.median(pairwise_dist) / (2 * jnp.log(max_particles + 1))
    return h

@jax.jit
def rbf_kernel(x, h):
    pairwise_dists = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    kxy = jnp.exp(-pairwise_dists / h)
    dxkxy = -2.0 / h * (x[:, None, :] - x[None, :, :]) * kxy[:, :, None]
    return kxy, dxkxy

@jax.jit
def rbf_kernel_median(x, h=-1):
    sq_dist = jnp.sum((x[:, None, :] - x[None, :, :])**2, axis=-1)
    h = jnp.median(sq_dist)
    h = jnp.where(h <= 0.0, 1.0, h)
    h = h / jnp.log(x.shape[0] + 1.0)
    k = jnp.exp(-sq_dist / h)
    grad_k = -(x[:, None, :] - x[None, :, :]) * k[:, :, None] * 2.0 / h
    return k, grad_k

# @jax.jit
# def rbf_kernel_local(X, neighbors=5):
#     pairwise_dists = jnp.sum((X[:, None, :] - X[None, :, :])**2, axis=-1)
#     local_bandwidths = local_adaptive_bandwidth(X, k=neighbors)
#     # Compute average bandwidths between all pairs: (n_particles, n_particles)
#     h_matrix = 0.5 * (local_bandwidths[:, None] + local_bandwidths[None, :])
#     h = h_matrix
#     K = jnp.exp(-pairwise_dists / h)
#     grad_K = -2 / h * (X[:, None, :] - X[None, :, :]) * K[:, :, None]
#     return K, grad_K

# Variable-dimension decoder class
class VariableDimPTDDecoder(eqx.Module):
    k: int  # dimension of absorption vectors
    m: int  # number of states
    param_dim: int  # total parameter dimension
    linear: eqx.nn.Linear
    
    def __init__(self, k=1, m=2, key=None):
        self.k = k
        self.m = m
        self.param_dim = calculate_param_dim(k, m)
        if key is None:
            key = jax.random.key(0)
        # Map from latent space to parameter space
        self.linear = eqx.nn.Linear(self.param_dim, self.param_dim, key=key)
    
    def __call__(self, z):
        # Apply linear transformation and constraints
        raw_params = self.linear(z)
        return raw_params  # Constraints are applied in example_ptd_spec

@jax.jit
def logp(theta, k_dim, m_dim):
    log_prob = jnp.sum(vmap(lambda x: log_pmf_dph(x, theta, k_dim, m_dim))(data))
    return log_prob

@jax.jit
@partial(jax.vmap, in_axes=(0,))
def logp_z(z, k_dim, m_dim):
    theta = z_to_theta(z)         
    return jnp.sum(vmap(lambda z_i: log_pmf_dph(z_i, theta, k_dim, m_dim))(data))

@jax.jit
def kl_adaptive_step(phi, kl_target=0.1, min_step=1e-7, max_step=1.0):
    norm_sq = jnp.sum(phi ** 2, axis=1)
    mean_norm_sq = jnp.mean(norm_sq)
    eta = jnp.sqrt((2.0 * kl_target) / (mean_norm_sq + 1e-8))

    return jnp.maximum(jnp.minimum(eta, max_step), min_step)

@jax.jit
def decayed_kl_target(t, base=0.1, decay=0.01):
    return base * jnp.exp(-decay * t)

@jax.jit
def step_size_schedule(i, steps=None, initial=None, final=None):
    """Cosine annealing schedule for step size"""
    return final + 0.5 * (initial - final) * (1 + jnp.cos(jnp.pi * i / steps))

@jax.jit
def z_to_theta(z):
    return decoder(z)


def update_fixed_bw_fixed_step(logp_fn, particles_z, h=0.01, step=0.001):
    grads = vmap(grad(logp_fn))(particles_z)
    K, grad_K = rbf_kernel(particles_z, h)
    phi = (K @ grads + jnp.sum(grad_K, axis=1)) / particles_z.shape[0]
    return particles_z + step * phi


def update_median_bw_fixed_step(logp_fn, particles_z, step=0.001):
    grads = vmap(grad(logp_fn))(particles_z)
    K, grad_K = rbf_kernel_median(particles_z)
    phi = (K @ grads + jnp.sum(grad_K, axis=1)) / particles_z.shape[0]
    return particles_z + step * phi

@jax.jit
def update_median_bw_kl_step(particles_z, k_dim, m_dim, kl_target=0.01, max_step=1.0):

    def logp_z(z):
        theta = z_to_theta(z)         
        return jnp.sum(vmap(lambda z_i: log_pmf_dph(z_i, theta, k_dim, m_dim))(data))

    #phi = svgd_phi(logp_fn, particles_z)
    score = vmap(grad(logp_z))(particles_z)
    kxy, dxkxy = rbf_kernel_median(particles_z)
    phi = (kxy @ score + jnp.sum(dxkxy, axis=1)) / particles_z.shape[0]

    eta = kl_adaptive_step(phi, kl_target, max_step=max_step)
    return particles_z + eta * phi      

@jax.jit
def update_local_bw_kl_step(particles, k_dim, m_dim, neighbors=5, kl_target=0.01, max_step=0.1):
    """
    Vectorized implementation of SVGD with adaptive bandwidth
    
    Args:
        particles: array of shape (n_particles, dim)
        k_dim: dimension of absorption vectors
        m_dim: number of states
        neighbors: number of neighbors for local bandwidth
        kl_target: target KL divergence
        max_step: maximum step size
        
    Returns:
        Updated particles after one SVGD step
    """

    def logp_z(z):
        theta = z_to_theta(z)         
        return jnp.sum(vmap(lambda z_i: log_pmf_dph(z_i, theta, k_dim, m_dim))(data))
    

    # Compute local bandwidths for each particle
    local_bandwidths = local_adaptive_bandwidth(particles)
    
    # Compute gradients of log probability for all particles
    log_prob_grads = jax.vmap(jax.grad(logp_z))(particles)
    
    # Number of particles
    n_particles = particles.shape[0]
    
    # Compute pairwise differences between particles: (n_particles, n_particles, dim)
    pairwise_diff = particles[:, None, :] - particles[None, :, :]
    
    # Compute pairwise squared distances: (n_particles, n_particles)
    pairwise_dist_sq = jnp.sum(pairwise_diff**2, axis=-1)
    
    # Compute average bandwidths between all pairs: (n_particles, n_particles)
    h_matrix = 0.5 * (local_bandwidths[:, None] + local_bandwidths[None, :])
    
    # Compute kernel matrix: (n_particles, n_particles)
    kernel_matrix = jnp.exp(-0.5 * pairwise_dist_sq / h_matrix)
    
    # Compute kernel-weighted average of gradients (kernel term)
    # Fix: Use direct matrix multiplication instead of einsum
    kernel_term_sum = jnp.matmul(kernel_matrix, log_prob_grads) / n_particles
    
    # Compute gradient of kernel w.r.t. particles (repulsive term)
    grad_kernel = (kernel_matrix[:, :, None] * pairwise_diff) / h_matrix[:, :, None]
    
    # Sum over particles j: (n_particles, dim)
    repulsive_term_sum = jnp.sum(grad_kernel, axis=1) / n_particles
    
    # Combine terms for the SVGD update
    phi = kernel_term_sum + repulsive_term_sum
    
    step_size = kl_adaptive_step(phi, kl_target, max_step=max_step)

    # Update particles
    updated_particles = particles + step_size * phi
    
    return updated_particles


@pjit
def distributed_svgd_step(x, k_dim, m_dim, kl_target, max_step):
    # return update_median_bw_kl_step(x, k_dim, m_dim, kl_target=kl_target, max_step=max_step)
    return update_local_bw_kl_step(x, k_dim, m_dim, kl_target=kl_target, max_step=max_step, neighbors=5)


# Main execution function for variable dimensions
def run_variable_dim_svgd(k=1, m=2, n_samples=64000, n_particles=80, n_iter=100):
    """
    Run SVGD for k-dimensional PTD with m states
    
    Args:
        k: dimension of absorption vectors
        m: number of transient states
        n_samples: number of data samples to generate
        n_particles: number of SVGD particles
        n_iter: number of SVGD iterations
    
    Returns:
        particles: final parameter particles
        data: generated data
        particle_history: history of particles during optimization
    """
    global data, decoder, z_to_theta
    
    key = jax.random.key(0)
    
    # Generate true parameters based on k and m
    param_dim = calculate_param_dim(k, m)
    
    # Create realistic true parameters
    true_alpha_logits = jnp.array([2.0, -1.0]) if m == 2 else jax.random.normal(key, (m,))
    true_T_logits = jnp.array([[0.5, -0.5], [-2.0, 0.5]]) if m == 2 else jax.random.normal(key, (m, m))
    true_t_logits = jax.random.normal(key, (k, m)) * 0.5
    
    true_params = jnp.concatenate([
        true_alpha_logits.flatten(),
        true_T_logits.flatten(),
        true_t_logits.flatten()
    ])
    
    print(f"True parameter dimension: {len(true_params)}")
    print(f"Expected parameter dimension: {param_dim}")
    
    # Generate data
    data = simulate_example_data(true_params, k=k, m=m, samples=n_samples, key=key)
    print(f"Generated data shape: {data.shape}")  # Should be (n_samples, k)
    
    # Setup decoder
    decoder = VariableDimPTDDecoder(k=k, m=m, key=key)
    
    def z_to_theta(z):
        return decoder(z)
    
    # SVGD parameters
    n_devices = min(8, n_particles)  # Don't exceed available devices
    kl_target_base = 0.1
    kl_target_decay = 0.01
    max_step = 0.001
    min_step = 1e-7
    max_step_scaler = 0.1
    
    if n_particles % n_devices != 0:
        n_particles = (n_particles // n_devices) * n_devices
        print(f"Adjusted n_particles to {n_particles} for even sharding")
    
    # Initial particles
    particles_z = jax.random.normal(key, shape=(n_particles, param_dim))
    
    # Shard particles over devices
    devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = Mesh(devices, axis_names=("i",))
    sharding = NamedSharding(mesh, P("i", None))
    particles_z = jax.device_put(particles_z, sharding)
    
    # SVGD iterations
    particle_z_history = [particles_z]
    every = max(1, n_iter // 10)  # Save every 10% of iterations
    prev = None
    
    with mesh:
        for i in trange(n_iter):
            kl_target = decayed_kl_target(i, base=kl_target_base, decay=kl_target_decay)
            particles_z = distributed_svgd_step(particles_z, k, m, kl_target=kl_target, max_step=max_step)
            
            # Adaptive step size based on log probability
            if i % 10 == 0:  # Check every 10 iterations
                this = jnp.median(vmap(lambda z: logp_z(z, k, m))(particles_z[None, :, :]), axis=1)
                if prev is not None and this < prev:
                    max_step = max(max_step * max_step_scaler, min_step)
                prev = this
            
            if not i % every:
                particle_z_history.append(particles_z)
    
    # Extract final results
    particles = jnp.array([z_to_theta(z) for z in particles_z])
    
    print(f"\nResults for k={k}, m={m}:")
    print(f"True parameters shape: {true_params.shape}")
    print(f"Estimated parameters shape: {particles.shape}")
    print(f"Parameter means: {jnp.mean(particles, axis=0)}")
    print(f"True parameters: {true_params}")
    
    return particles, data, np.array(particle_z_history), true_params


# N^(-1/(4+d))


# Example usage - modify these parameters to test different dimensions
if __name__ == "__main__":
    # Test different configurations
    print("=" * 60)
    print("Testing univariate case (k=1)")
    particles_1d, data_1d, history_1d, true_1d = run_variable_dim_svgd(k=1, m=2, n_samples=10000, n_particles=40, n_iter=50)
    
    print("\n" + "=" * 60)
    print("Testing bivariate case (k=2)") 
    particles_2d, data_2d, history_2d, true_2d = run_variable_dim_svgd(k=2, m=2, n_samples=10000, n_particles=40, n_iter=50)
    
    print("\n" + "=" * 60)
    print("Testing 3-dimensional case (k=3)")
    particles_3d, data_3d, history_3d, true_3d = run_variable_dim_svgd(k=3, m=2, n_samples=10000, n_particles=40, n_iter=50)