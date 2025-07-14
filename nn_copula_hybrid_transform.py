class IndependentProbDecoder(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int, *, key):
        self.linear = eqx.nn.Linear(latent_dim, 2, key=key)

    def __call__(self, z: jnp.ndarray):
        logits = self.linear(z)  # shape (2,)
        a, b = jax.nn.sigmoid(logits[0]), jax.nn.sigmoid(logits[1])
        return a, b

@jax.jit
def log_pmf_dph(z, theta):
    """
    Fully static version for JIT compilation
    """
    a, b = theta

    alpha, T, t_vecs = ptd_spec(a, b)


    m = T.shape[0]
    k = len(t_vecs)

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
    
    # Use a fully vectorized approach without conditionals
    # Pad z_clipped to always have k elements (for univariate case)
    z_padded = jnp.pad(z_clipped, (0, max(0, k - len(z_clipped))))[:k]
    
    # Use advanced indexing with static shapes
    indices = jnp.arange(k)
    prob_at_z = probs_filled[z_padded, indices]
    logp = jnp.sum(jnp.log(jnp.maximum(prob_at_z, 1e-12)))
    
    return logp
    
@jax.jit
def svgd_update_z(particles_z, data, m, k, step_size):
    def logp_z(z):
        theta = z_to_theta(z, m, k) 
        
        return jnp.sum(vmap(lambda z_i: log_pmf_dph(z_i, theta))(data))
    grads = vmap(grad(logp_z))(particles_z)
    K, grad_K = rbf_kernel(particles_z)
    phi = (K @ grads + jnp.sum(grad_K, axis=1)) / particles_z.shape[0]
    return particles_z + step_size * phi
    
decoder = IndependentProbDecoder(latent_dim=2, key=key)

@jax.jit
def z_to_theta(z, m, k):
    # return unpack_theta(z, m, k)
    return decoder(z)
    # jax.debug.print("🤯 {}", jnp.array(theta))