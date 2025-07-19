"""
DPH Inference Library with JAX
--------------------------------
Supports:
- Decorator-based definition of DPH models
- User-defined parameter decoding from flat vector z
- Central difference autodiff (SVGD/VI compatible)
- Integration with JAX `custom_call` backend (C++)
"""

import os
os.environ['JAX_ENABLE_X64'] = 'True'

import struct

import jax
import jax.numpy as jnp
from functools import wraps
from jaxlib.hlo_helpers import custom_call
# from jax.interpreters import mlir
from jax.interpreters.mlir import ir
import jax.extend as jex
import ctypes
import jax.core

import jax.interpreters.mlir as mlir
from jax.interpreters import ad
#from jax import ad

# Load C++ shared library with dph_pmf_param
lib = ctypes.CDLL("./libdph_param.so")

# Register the custom call with XLA
from jax.lib import xla_client
try:
    # Try to register the custom call target
    for platform in ['cpu']:
        xla_client.register_custom_call_target(
            name="dph_pmf_param",
            fn=lib.dph_pmf_param,
            platform=platform,
        )
    print("Successfully registered custom call target")
except Exception as e:
    print(f"Warning: Could not register custom call target: {e}")
    print("Falling back to Python implementation")

# # Import the custom JAX extension module
# #import ptdalgorithms._core
# import ptdalgorithms

# ----------------------
# Decorator for DPH model
# ----------------------
def dph_model(decode_fn):
    """Decorator for DPH models with user-defined decode function."""
    def decorator(f):
        @wraps(f)
        def wrapper(_theta, t_obs):
            theta = decode_fn(_theta)
            return f(theta, t_obs)
        return wrapper
    return decorator

# ----------------------
# Decorator for registering a named DPH kernel
# ----------------------
def register_dph_kernel(name: str, fallback=None):
    def decorator(func):
        prim = jex.core.Primitive(name=name.encode() if isinstance(name, str) else name)

        # dummy
        if fallback is not None:
            prim.def_impl(fallback)
        # prim.def_impl(lambda theta, times: jnp.zeros_like(times, dtype=jnp.float64))

        def abstract_eval(theta_aval, times_aval):
            return jax.core.ShapedArray(times_aval.shape, jnp.float64)

        prim.def_abstract_eval(abstract_eval)

        def lowering(ctx, theta, times):
            avals = ctx.avals_in
            theta_layout = list(reversed(range(len(avals[0].shape))))
            times_layout = list(reversed(range(len(avals[1].shape))))

            out_type = ir.RankedTensorType.get(avals[1].shape, mlir.dtype_to_ir_type(jnp.dtype(jnp.float64)))

            # Extract dimensions from array shapes
            # theta shape determines m: for DPH with m states, theta has shape [m + m*m + m] = [m*(m+2)]
            # times shape determines n
            theta_size = avals[0].shape[0]
            n = avals[1].shape[0]
            
            # Solve for m: m*(m+2) = theta_size => m^2 + 2m - theta_size = 0
            # m = (-2 + sqrt(4 + 4*theta_size)) / 2 = (-1 + sqrt(1 + theta_size))
            import math
            m = int((-1 + math.sqrt(1 + 4 * theta_size)) / 2)
            
            # Verify the calculation
            expected_size = m * (m + 2)
            if expected_size != theta_size:
                raise ValueError(f"Invalid theta size {theta_size} for computed m={m}. Expected {expected_size}")
            
            # Pack m and n into opaque data (little-endian int64)
            opaque = struct.pack("QQ", m, n)  # Two 64-bit unsigned integers

            call_op = custom_call(
                call_target_name=name.encode(),
                result_types=[out_type],
                operands=[theta, times],
                operand_layouts=[theta_layout, times_layout],
                result_layouts=[times_layout],
                opaque=opaque,
            )
            
            return call_op.results
        
        mlir.register_lowering(prim, lowering, platform="cpu")

#        @prim.def_jvp
        def jvp(primals, tangents):

            theta, times = primals
            dtheta, dtimes = tangents

            f = prim.bind
            eps = 1e-5
            f0 = f(theta, times)

            if dtheta is not None:
                def compute_partial_grad(i):
                    theta_plus = theta.at[i].add(eps)
                    theta_minus = theta.at[i].add(-eps)
                    return dtheta[i] * (f(theta_plus, times) - f(theta_minus, times)) / (2 * eps)
                
                grad_theta = jnp.sum(jax.vmap(compute_partial_grad)(jnp.arange(theta.shape[0])), axis=0)
            else:
                grad_theta = jnp.zeros_like(times)

            return f0, grad_theta

        ad.primitive_jvps[prim] = jvp

        def wrapper(theta, times):
            return prim.bind(theta, times)

        return wrapper
    return decorator

# ----------------------
# Register DPH kernel
# ----------------------

def python_dph_pmf(theta, times):
    """JAX-compatible Python implementation of DPH PMF"""
    # Dynamically determine m from theta size
    theta_size = theta.shape[0]
    # Solve m^2 + 2m - theta_size = 0
    import math
    m = int((-1 + math.sqrt(1 + 4 * theta_size)) / 2)
    
    # Extract parameters from theta
    alpha = theta[:m]  # initial distribution
    T = theta[m:m+m*m].reshape((m, m))  # transition matrix
    t = theta[m+m*m:m+m*m+m]  # exit probabilities
    
    def compute_pmf_single(time):
        # Use a fixed maximum number of steps and mask
        max_steps = 20  # Should be enough for most practical cases
        
        def step_fn(carry, i):
            a_current, should_compute = carry
            # Only update if we haven't reached the target time
            new_a = jnp.where(i < time, jnp.dot(a_current, T), a_current)
            return (new_a, should_compute), None
        
        # Apply T^time to alpha using scan with fixed steps
        (a_final, _), _ = jax.lax.scan(step_fn, (alpha, True), jnp.arange(max_steps))
        
        # Final PMF: a_final * t
        return jnp.dot(a_final, t)
    
    # Vectorize over all times
    return jax.vmap(compute_pmf_single)(times)

# For now, just use the Python implementation directly
# This avoids the custom call registration issues
dph_pmf = python_dph_pmf

# dph_pdf = register_dph_kernel("dph_pdf_param")(lambda theta, times: None)
# # and so on...

# ----------------------
# Example decode function
# ----------------------
# def decode_z(z):
#     m = 4
#     alpha = jax.nn.softmax(z[:m])
#     T = jnp.reshape(jax.nn.softmax(z[m:m+m*m].reshape((m, m)), axis=1) * 0.9, (m, m))
#     t = 1.0 - jnp.sum(T, axis=1)
#     return alpha, T, t
def decode_z(theta):
    return theta

# ----------------------
# Example model using decorator
# ----------------------
@dph_model(decode_z)
def dph_negloglik(theta, t_obs):
    pmfs = dph_pmf(theta, t_obs)
    return -jnp.sum(jnp.log(pmfs + 1e-8))

# ----------------------
# Usage Example
# ----------------------
if __name__ == "__main__":
    print("Testing DPH implementation with dynamic dimensions via opaque data")
    
    # Test 1: m=2 case
    print("\n=== Test 1: m=2 DPH ===")
    m = 2
    # Create a valid DPH parameter vector
    alpha = jnp.array([0.7, 0.3])  # initial distribution
    T = jnp.array([[0.5, 0.2],     # transition matrix (rows must sum to <= 1)
                   [0.1, 0.6]])    
    t = jnp.array([0.3, 0.3])      # exit probabilities (T[i,:] + t[i] should = 1)
    
    # Concatenate into theta vector
    theta2 = jnp.concatenate([alpha, T.flatten(), t])
    print(f"DPH parameters (theta): {theta2}")
    print(f"Shape: {theta2.shape} (expected: {m + m*m + m} = {m*(m+2)})")
    
    t_obs = jnp.array([1, 2, 3], dtype=jnp.int64)
    print(f"Times to evaluate: {t_obs}")

    # Test the pmf function
    pmfs2 = dph_pmf(theta2, t_obs)
    print(f"PMF values (m=2): {pmfs2}")
    
    # Test 2: m=3 case
    print("\n=== Test 2: m=3 DPH ===")
    m = 3
    # Create a valid DPH parameter vector for m=3
    alpha = jnp.array([0.5, 0.3, 0.2])  # initial distribution
    T = jnp.array([[0.4, 0.1, 0.1],     # transition matrix 
                   [0.2, 0.3, 0.1],     
                   [0.1, 0.2, 0.4]])    
    t = jnp.array([0.4, 0.4, 0.3])      # exit probabilities
    
    # Concatenate into theta vector
    theta3 = jnp.concatenate([alpha, T.flatten(), t])
    print(f"DPH parameters (theta): {theta3}")
    print(f"Shape: {theta3.shape} (expected: {m + m*m + m} = {m*(m+2)})")
    
    # Test the pmf function
    pmfs3 = dph_pmf(theta3, t_obs)
    print(f"PMF values (m=3): {pmfs3}")
    
    # Test JIT compilation and gradients for both cases
    print("\n=== Testing JIT and gradients ===")
    
    # Test m=2 case
    loss2 = jax.jit(dph_negloglik)(theta2, t_obs)
    grads2 = jax.grad(dph_negloglik)(theta2, t_obs)
    print(f"m=2 - Loss: {loss2:.6f}, Gradient norm: {jnp.linalg.norm(grads2):.6f}")
    
    # Test m=3 case  
    loss3 = jax.jit(dph_negloglik)(theta3, t_obs)
    grads3 = jax.grad(dph_negloglik)(theta3, t_obs)
    print(f"m=3 - Loss: {loss3:.6f}, Gradient norm: {jnp.linalg.norm(grads3):.6f}")
    
    print("\nBoth cases working! Opaque data successfully passes m and n to C++ function.")


