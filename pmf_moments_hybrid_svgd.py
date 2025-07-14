def combined_loss(alpha, T, t_vecs, target_pmf, target_moments, target_covs, lambda_pmf=1.0, lambda_moments=0.5):
    """
    Compute weighted loss combining PMF fitting and moment matching
    """
    # PMF-based loss
    pred_pmf = calculate_pmf(alpha, T, t_vecs)
    pmf_loss = jnp.sum((pred_pmf - target_pmf)**2)  # L2 distance, could use KL instead
    
    # Moment-based loss
    pred_moments = calculate_moments(alpha, T, t_vecs)
    pred_covs = calculate_covariances(alpha, T, t_vecs)
    moment_loss = jnp.sum((pred_moments - target_moments)**2)
    cov_loss = jnp.sum((pred_covs - target_covs)**2)
    
    # Combined loss with weighting parameters
    return lambda_pmf * pmf_loss + lambda_moments * (moment_loss + cov_loss)

@jax.jit
def svgd_update_combined(particles_z, data, target_moments, target_covs, step_size):
    """SVGD update using the combined loss function"""
    def logp_z(z):
        a, b, rho = z_to_params(z)
        alpha, T, t_vecs, _ = ptd_spec_hybrid(a, b, rho)
        
        # Calculate empirical PMF from data
        z_max = 50
        target_pmf = calculate_empirical_pmf(data, z_max)
        
        # Calculate predicted PMF
        pred_pmf = calculate_pmf(alpha, T, t_vecs)
        
        # Calculate log probability using combined loss
        neg_loss = -combined_loss(alpha, T, t_vecs, target_pmf, target_moments, 
                                  target_covs, lambda_pmf=1.0, lambda_moments=0.5)
        return neg_loss
    
    grads = vmap(jax.grad(logp_z))(particles_z)
    K, grad_K = rbf_kernel(particles_z)
    phi = (K @ grads + grad_K.sum(axis=1)) / particles_z.shape[0]
    return particles_z + step_size * phi

def calculate_empirical_pmf(data, z_max):
    """Calculate empirical PMF from observed data"""
    # For multidimensional data, we need to count occurrences of each value combination
    # This is a simplified version for 1D or 2D data
    counts = jnp.zeros(z_max)
    
    def count_value(counts, value):
        # Clip to ensure within bounds
        idx = jnp.clip(value.astype(int), 0, z_max-1)
        return counts.at[idx].add(1)
    
    counts = jax.lax.foldl(count_value, data, counts)
    # Normalize to get probability
    return counts / jnp.sum(counts)

def calculate_pmf(alpha, T, t_vecs):
    """Calculate PMF from phase-type parameters"""
    m = T.shape[0]
    k = len(t_vecs)
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
    
    # Sum across all exit vectors for each time step
    return jnp.sum(probs_filled, axis=1)

def calculate_moments(alpha, T, t_vecs):
    """Calculate first few moments of the phase-type distribution"""
    m = T.shape[0]
    I = jnp.eye(m)
    
    # First moment (mean)
    T_inv = jnp.linalg.inv(I - T)
    mean = alpha @ T_inv @ jnp.ones(m)
    
    # Second raw moment
    second_moment = 2 * alpha @ T_inv @ T_inv @ jnp.ones(m)
    
    # Third raw moment
    third_moment = 6 * alpha @ T_inv @ T_inv @ T_inv @ jnp.ones(m)
    
    return jnp.array([mean, second_moment, third_moment])

def calculate_covariances(alpha, T, t_vecs):
    """Calculate covariances for multivariate phase-type distribution"""
    m = T.shape[0]
    k = len(t_vecs)
    I = jnp.eye(m)
    T_inv = jnp.linalg.inv(I - T)
    
    # Calculate means for each component
    t_mat = jnp.stack(t_vecs, axis=1)  # shape (m, k)
    component_means = jnp.zeros(k)
    
    for i in range(k):
        t_i = t_vecs[i]
        component_means = component_means.at[i].set(alpha @ T_inv @ T_inv @ t_i)
    
    # Calculate covariances between components
    covs = jnp.zeros((k, k))
    
    for i in range(k):
        for j in range(i, k):
            t_i, t_j = t_vecs[i], t_vecs[j]
            if i == j:
                # Variance
                cov_ij = 2 * alpha @ T_inv @ T_inv @ T_inv @ t_i - component_means[i]**2
            else:
                # Covariance
                cov_ij = alpha @ T_inv @ T_inv @ t_i @ T_inv @ t_j + \
                         alpha @ T_inv @ T_inv @ t_j @ T_inv @ t_i - \
                         component_means[i] * component_means[j]
            
            covs = covs.at[i, j].set(cov_ij)
            covs = covs.at[j, i].set(cov_ij)  # Symmetric
    
    return covs

# Modified training function to use combined loss
def train_hybrid_model_combined(data, target_moments, target_covs, 
                               num_particles=50, num_iterations=1000, step_size=0.01):
    """Train the hybrid model using SVGD with combined loss"""
    # Initialize particles
    key = jax.random.PRNGKey(0)
    particles_z = jax.random.normal(key, (num_particles, 2))
    
    # Training loop
    for i in range(num_iterations):
        particles_z = svgd_update_combined(particles_z, data, target_moments, target_covs, step_size)
        
        if i % 100 == 0:
            # Compute and print loss for monitoring
            a, b, rho = z_to_params(particles_z[0])  # Just check first particle
            alpha, T, t_vecs, _ = ptd_spec_hybrid(a, b, rho)
            
            # Calculate empirical PMF
            z_max = 50
            target_pmf = calculate_empirical_pmf(data, z_max)
            
            # Calculate loss
            loss = combined_loss(alpha, T, t_vecs, target_pmf, target_moments, target_covs)
            print(f"Iteration {i}, Combined Loss: {loss}")
    
    return particles_z