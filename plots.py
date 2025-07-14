from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
from jax import vmap
from jax import numpy as jnp


# "iridis" color map (viridis without the deep purple)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
iridis = truncate_colormap(plt.get_cmap('viridis'), 0.2, 1)



def plot_svgd_posterior_1d(particles, true_params=None, obs_stats=None, 
                           map_est=None,
                           ax=None, title="SVGD Posterior Approximation"):
    """
    Plot 1D posterior approximation from SVGD particles
    
    Args:
        particles: shape (n_particles, 1) array of SVGD particles
        true_params: optional true parameter value for comparison
        title: plot title
    """
    if ax is None:        
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    
    # Extract 1D values
    x = particles.flatten()
    
    # Plot histogram of particles
    ax.hist(x, bins=30, density=True, alpha=0.4, label='Particle histogram')
    
    # Plot KDE of posterior
    kde = gaussian_kde(x)
    xx = np.linspace(min(x), max(x), 1000)
    ax.plot(xx, kde(xx), color='orange', lw=2, label='KDE posterior')
    
    # Add true parameter if provided
    if true_params is not None:
        ax.axvline(true_params, color='hotpink', linestyle='--', 
                   label=f'True value: {true_params:.2f}')
        
    # Add data statistics
    if obs_stats is not None:
        ax.axvline(obs_stats, color='magenta',
                   label=f'Observed value: {obs_stats:.2f}')    
    if map_est is not None:
        ax.axvline(map_est, color='orange', linestyle='dashed',
                   label=f'MAP value: {map_est:.2f}')       
    
    ax.set_title(title)
    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Density')
    ax.legend()
    sns.despine(ax=ax)

def plot_svgd_posterior_2d(particles, true_params=None, obs_stats=None,
                          map_est=None, idx=(0, 1),
                          figsize=(10, 8),
                          labels=["Parameter 1", "Parameter 2"],
                          title=None):
    """
    Plot 2D posterior approximation from SVGD particles
    
    Args:
        particles: shape (n_particles, 2) array of SVGD particles
        true_params: optional array of true parameter values [p1, p2]
        labels: parameter names for axes
        title: plot title
    """
    print(true_params)
    plt.rcParams['animation.embed_limit'] = 100 # Mb

    plt.figure(figsize=figsize)
    
    # Extract parameters
    x = particles[:, idx[0]]
    y = particles[:, idx[1]]
    
    # Create 2D histogram
    plt.subplot(2, 2, 1)
    plt.hist2d(x, y, bins=30, cmap='viridis')
    plt.colorbar(label='Particle count')
    if true_params is not None:
        plt.plot(true_params[0], true_params[1], ls='', color='hotpink', marker='*', markersize=10, 
                label='True value')        
    if obs_stats is not None:
        plt.plot(obs_stats[0], obs_stats[1], ls='', color='magenta', marker='*', markersize=10, 
            label='Obs value')        
    if map_est is not None:
        plt.plot(map_est[0], map_est[1], ls='', color='orange', marker='*', markersize=10, 
                 label=f'MAP value')

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title('2D Histogram')
    
    # Create scatter plot
    plt.subplot(2, 2, 2)
    if true_params is not None:
        plt.gca().axvline(true_params[0], color='hotpink', linewidth=0.5, linestyle='--', zorder=-1)   
        plt.gca().axhline(true_params[1], color='hotpink', linewidth=0.5, linestyle='--', zorder=-1, label='True value')   
        # plt.plot(true_params[0], true_params[1], ls='', color='red', marker='*', markersize=10, 
        #         label='True value')
        plt.legend()
    plt.scatter(x, y, alpha=0.5, s=5, edgecolor='none')
    if obs_stats is not None:
        plt.plot(obs_stats[0], obs_stats[1], ls='', color='magenta', marker='*', markersize=10, 
            label='Obs value')        
    if map_est is not None:
        plt.plot(map_est[0], map_est[1], ls='', color='orange', marker='*', markersize=10, 
                 label='MAP value')

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title('Particle Distribution')
    
    plt.subplot(2, 2, 3)
    plot_svgd_posterior_1d(
        x,
        true_params=true_params[0] if true_params is not None else None,
        # obs_stats=data[:, 0].mean() if data is not None else None,
        map_est=map_est[0] if map_est is not None else None,
        ax=plt.gca(),
        title="Posterior Distribution of Mean Parameter"
    )

    plt.subplot(2, 2, 4)
    plot_svgd_posterior_1d(
        y, # Select only the second parameter
        true_params=true_params[1] if true_params is not None else None,
        # obs_stats=data[:, 1].mean() if data is not None else None,
        map_est=map_est[1] if map_est is not None else None,
        ax=plt.gca(),
        title="Posterior Distribution of Standard Deviation Parameter"
    )
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

def animate_svgd_2d(particle_history, true_params=None, obs_stats=None,
                    map_est=None, idx=[0, 1],
                    skip=0,
                    figsize=(5, 5),
                   labels=["Parameter 1", "Parameter 2"],
                    save_as_gif=None, save_as_mp4=None,
                   title=None):
    """
    Create an animation of SVGD particle evolution
    
    Args:
        particle_history: list of particle arrays at each iteration
        true_params: optional true parameter values [p1, p2]
        labels: parameter names for axes
        title: animation title
    """
    fig, ax = plt.subplots(figsize=figsize)

    assert skip < particle_history.shape[0], "Skip value must be less than number of iterations"
    particle_history = particle_history[skip:, idx, :]  # Select only the parameters of interest

    # min/max for axis limits
    x_min = np.min(particle_history)
    x_max = np.max(particle_history)
    y_min = np.min(particle_history)
    y_max = np.max(particle_history)

    # padding
    x_pad = 0.1 * (x_max - x_min)
    y_pad = 0.1 * (y_max - y_min)
    
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    
    # Plot true parameters if provided
    if true_params is not None:
        ax.plot(true_params[0], true_params[1], ls='', color='hotpink', marker='*', markersize=5, label='True value')
    if obs_stats is not None:
        ax.plot(obs_stats[0], obs_stats[1], ls='', color='magenta', marker='*', markersize=5, label='Observed value')
    if map_est is not None:
        ax.plot(map_est[0], map_est[1], ls='', color='orange', marker='*', markersize=5, label='MAP value')

    # Initialize scatter plot
    scatter = ax.scatter([], [], alpha=1, s=5, edgecolor='none')
    iteration_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        iteration_text.set_text('')
        return scatter, iteration_text
    
    def update(frame):
        scatter.set_offsets(particle_history[frame, :, :].T)  # Transpose to get shape (2, n_particles)
        # scatter.set_offsets(particle_history[frame][:, :first_params])  # Just first two parameters

        iteration_text.set_text(f'Iteration: {frame}')
        return scatter, iteration_text
    
    anim = FuncAnimation(fig, update, frames=len(particle_history),
                         init_func=init, blit=True, interval=100)
    
    plt.close()  # Prevent duplicate display in notebooks

    # Save animation as a gif or mp4
    if save_as_gif:
        anim.save(save_as_gif, writer='pillow', fps=10)
    if save_as_mp4:
        anim.save(save_as_mp4, writer='ffmpeg', fps=10)


    from IPython.display import HTML
    return HTML(anim.to_jshtml())
#    return anim


def check_convergence(particle_history, log_p_fn, data, every=1, text=None):
    """Monitor convergence of SVGD by tracking statistics"""
    mean_params = []
    std_params = []
    log_probs = []
    
    def scale_labels(ax, every):
        """Scale x-ticks to match parameter values"""
        vals = ax.get_xticks()[1:-1]
        labels = (vals * every).astype(int)
        ax.set_xticks(vals, labels=labels)

    for i in range(particle_history.shape[0]):
        particles = particle_history[i, :, :].T
        # track parameter statistics
        mean_params.append(np.mean(particles, axis=0))
        std_params.append(np.std(particles, axis=0))
        # track average log probability
        avg_log_p = np.mean([log_p_fn(data, p) for p in particles])
        log_probs.append(avg_log_p)
    
    if text is not None:
        fig = plt.figure(figsize=(10, 4))
        gs = GridSpec(2, 3, figure=fig, height_ratios=(4, 1))
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        if type(text) is str:
            text = [text]
            text_ax = [fig.add_subplot(gs[1, :])]
        else:
            text_ax = [
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[1, 1]),
                fig.add_subplot(gs[1, 2])
            ]
        [ax.set_axis_off() for ax in text_ax]
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))

    # Plot mean parameters
    ax1.plot([p[0] for p in mean_params], label='Param 1')
    ax1.plot([p[1] for p in mean_params], label='Param 2')
    ax1.set_title('Parameter Means')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Value')
    ax1.legend()
    scale_labels(ax1, every)

    # Plot parameter standard deviations
    ax2.plot([p[0] for p in std_params], label='Param 1')
    ax2.plot([p[1] for p in std_params], label='Param 2')
    ax2.set_title('Parameter Standard Deviations')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Value')
    ax2.legend()
    scale_labels(ax2, every)

    # Plot log probabilities
    ax3.plot(log_probs)
    ax3.set_title('Average Log Probability')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Log Prob')
    scale_labels(ax3, every)
    
    if text is not None:
        for i, ax in enumerate(text_ax):
            ax.text(0, 0.9, text[i], fontsize=10,
                        #  horizontalalignment='left',
                        verticalalignment='top',
                        fontname='monospace', 
                        #  traansform=ax.transAxes,
                        # bbox=dict(facecolor='red', alpha=0.5)
                        )
    
    # axes[0].annotate('axes fraction',
    #         xy=(2, 1), xycoords='data',
    #         xytext=(0.36, 0.68), textcoords='axes fraction',
    #         arrowprops=dict(facecolor='black', shrink=0.05),
    #         horizontalalignment='right', verticalalignment='top')

    plt.tight_layout()


def estimate_hdr(particles, log_prob_fn, alpha=0.95):
    """
    Estimate the Highest Density Region (HDR) from particles.
    
    Args:
        particles: Array of shape (n_particles, dim)
        log_prob_fn: Function that computes log probability
        alpha: Coverage probability (e.g., 0.95 for 95% HDR)
    
    Returns:
        List of particles that are within the HDR
        The log probability threshold that defines the HDR
    """
    n_particles = particles.shape[0]
    
    # Compute log probability for each particle
    # log_probs = jnp.array([log_prob_fn(particles[i]) for i in range(n_particles)])
    log_probs = vmap(log_prob_fn)(particles)

    # Sort particles by log probability (descending)
    sorted_indices = jnp.argsort(-log_probs)
    sorted_log_probs = log_probs[sorted_indices]
    
    # Find the log probability threshold for the HDR
    n_hdr = int(n_particles * alpha)
    threshold = sorted_log_probs[n_hdr-1]
    
    # Get particles in the HDR
    hdr_mask = log_probs >= threshold
    hdr_particles = particles[hdr_mask]
    
    return hdr_particles, threshold


def visualize_hdr_2d(particles, log_prob_fn, idx=[0, 1], alphas=[0.95], 
                     grid_size=50, margin=0.1, xlim=None, ylim=None):
    """
    Visualize the Highest Density Region (HDR) for a 2D distribution.
    
    Args:
        particles: Array of shape (n_particles, 2)
        log_prob_fn: Function that computes log probability
        alpha: Coverage probability (e.g., 0.95 for 95% HDR)
        grid_size: Size of the grid for visualization
        xlim, ylim: Limits for the grid
    
    Returns:
        Figure with HDR visualization
    """
    # if particles.shape[1] != 2:
    #     raise ValueError("This function only works for 2D distributions")
    
    # Determine limits if not provided

    if xlim is None:
        x_min, x_max = particles[:, idx[0]].min(), particles[:, idx[0]].max()
        _margin = (x_max - x_min) * margin
        xlim = (x_min - _margin, x_max + _margin)
    
    if ylim is None:
        y_min, y_max = particles[:, idx[1]].min(), particles[:, idx[1]].max()
        _margin = (y_max - y_min) * margin
        ylim = (y_min - _margin, y_max + _margin)
    
    # Create grid
    x = jnp.linspace(xlim[0], xlim[1], grid_size)
    y = jnp.linspace(ylim[0], ylim[1], grid_size)
    X, Y = jnp.meshgrid(x, y)

    # Evaluate log probability on grid
    theta_mean = jnp.mean(particles, axis=0)
    Z = jnp.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            p = theta_mean.copy()
            p = p.at[idx[0]].set(X[i, j])
            p = p.at[idx[1]].set(Y[i, j])
            Z = Z.at[i, j].set(log_prob_fn(p))

    # Get HDR threshold
    levels = []
    for alpha in alphas:
        _, threshold = estimate_hdr(particles, log_prob_fn, alpha)
        levels.append((threshold.item(), alpha))

    fig, ax = plt.subplots(figsize=(7, 5))

    # plot grid log likelihoods
    x, y, z = X.ravel(), Y.ravel(), Z.ravel()
    scatter = sns.scatterplot(x=x, y=y, 
                              hue=z, palette=iridis,
                              edgecolor='none', alpha=0.5, s=5, legend=False)
    # Find and mark logL grid point
    idx = jnp.argmax(z)
    ax.scatter(x[idx], y[idx], color='red', s=70, 
            marker='x', alpha=1,
            label='Max grid LogL')

    # Find and mark MAP estimate
    map_particle, _ = map_estimate_from_particles(particles, log_prob_fn)
    ax.scatter(map_particle[0], map_particle[1], color='orange', s=70, 
               marker='x', alpha=1,
               label='MAP estimate')
    # plot particles
    logLikelihoods = vmap(lambda p: log_prob_fn(p))(particles)    
    scatter = sns.scatterplot(x=particles[:, 0], y=particles[:, 1], 
                              hue=logLikelihoods, palette=iridis,
                              edgecolor='none', alpha=0.5, s=10, legend=False)
    # plot controur lines for HDR
    levels, alphas = zip(*sorted(levels))
    contour = ax.contour(X, Y, Z, levels=levels, cmap=iridis, linestyles='dashed', alpha=0.7)
    
    
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    # ax.set_title(f'Highest Density Region ({alpha*100:.0f}%)')
    ax.legend()
    # ax.grid(alpha=0.3)
    
    return fig


def map_estimate_from_particles(particles, log_prob_fn):
    """
    Find the MAP estimate from a set of particles by finding the particle
    with the highest log probability.
    
    Args:
        particles: Array of shape (n_particles, dim)
        log_prob_fn: Function that computes log probability
    
    Returns:
        The particle with highest log probability
    """
    n_particles = particles.shape[0]
    
    # Compute log probability for each particle
    log_probs = jnp.array([log_prob_fn(particles[i]) for i in range(n_particles)])
    
    # Find the particle with the highest log probability
    map_idx = jnp.argmax(log_probs)
    
    return particles[map_idx], log_probs[map_idx]


def map_estimate_with_optimization(particles, log_prob_fn, n_steps=100, step_size=0.01):
    """
    Refine MAP estimate by starting from the best particle and performing
    gradient ascent on the log probability.
    
    Args:
        particles: Array of shape (n_particles, dim)
        log_prob_fn: Function that computes log probability
        n_steps: Number of optimization steps
        step_size: Step size for gradient ascent
    
    Returns:
        The refined MAP estimate after optimization
    """
    # Start with the best particle
    map_particle, _ = map_estimate_from_particles(particles, log_prob_fn)
    
    # Define gradient of log probability
    grad_log_prob = jax.grad(log_prob_fn)
    
    # Perform gradient ascent to refine the MAP estimate
    x = map_particle
    for _ in range(n_steps):
        grad = grad_log_prob(x)
        x = x + step_size * grad
    
    return x, log_prob_fn(x)

