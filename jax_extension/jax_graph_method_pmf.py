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
# Disable JAX compilation cache to avoid HDF5 issues
# os.environ['JAX_COMPILATION_CACHE_DIR'] = ''
# os.environ['JAX_DISABLE_JIT'] = '0'  # Keep JIT enabled but disable caching

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

# Load C++ shared library with jax_graph_method_pmf
lib = ctypes.CDLL("/Users/kmt/PtDalgorithms/jax_extension/jax_graph_method_pmf.so")

# Create proper PyCapsule for JAX registration
def create_dph_capsule():
    """Create a PyCapsule for the DPH function"""
    import ctypes
    from ctypes import pythonapi, c_void_p, c_char_p
    
    # Get the function pointer
    func_ptr = lib.jax_graph_method_pmf
    
    # Create a PyCapsule containing the function pointer
    PyCapsule_New = pythonapi.PyCapsule_New
    PyCapsule_New.argtypes = [c_void_p, c_char_p, c_void_p]
    PyCapsule_New.restype = ctypes.py_object
    
    capsule = PyCapsule_New(
        ctypes.cast(func_ptr, c_void_p).value,
        b"xla._CUSTOM_CALL_TARGET",
        None
    )
    
    return capsule

# Register the function with XLA using the proper PyCapsule
def register_dph_function():
    """Register the DPH function with XLA using PyCapsule"""
    try:
        from jax._src.lib import xla_client
        
        # Create the PyCapsule
        capsule = create_dph_capsule()
        
        # Register with XLA using the capsule
        xla_client.register_custom_call_target(
            name="jax_graph_method_pmf",
            fn=capsule,  # Use the capsule instead of the raw function pointer
            platform="cpu"
        )
        print("Successfully registered jax_graph_method_pmf with XLA using PyCapsule")
        return True
        
    except Exception as e:
        print(f"Registration failed: {e}")
        return False

# Try to register the function
registration_success = register_dph_function()

print(f"C++ library loaded. Registration success: {registration_success}")

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
        def wrapper(_theta, data):
            theta = decode_fn(_theta)
            return f(theta, data)
        return wrapper
    return decorator

# ----------------------
# Decorator for registering a named DPH kernel
# ----------------------
def register_dph_kernel(name: str, fallback=None):
    def decorator(func):
        prim = jex.core.Primitive(name=name.encode() if isinstance(name, str) else name)

        # Use Python fallback that works with JIT and gradients
        if fallback is not None:
            prim.def_impl(fallback)

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
            theta_size = avals[0].shape[0] if len(avals[0].shape) > 0 else 1
            n = avals[1].shape[0] if len(avals[1].shape) > 0 else 1
            
            # Handle scalar case
            if len(avals[1].shape) == 0:
                n = 1
                times_layout = []
                
            # Solve for m: m*(m+2) = theta_size => m^2 + 2m - theta_size = 0
            # m = (-2 + sqrt(4 + 4*theta_size)) / 2 = (-1 + sqrt(1 + theta_size))
            import math
            m = int((-1 + math.sqrt(1 + 4 * theta_size)) / 2)
            
            # Verify the calculation
            expected_size = m * (m + 2)
            if expected_size != theta_size:
                raise ValueError(f"Invalid theta size {theta_size} for computed m={m}. Expected {expected_size}")
            
            # Since we can't pass opaque data, we'll create additional scalar operands for m and n
            m_operand = mlir.ir_constant(jnp.array(m, dtype=jnp.int64))
            n_operand = mlir.ir_constant(jnp.array(n, dtype=jnp.int64))

            call_op = custom_call(
                call_target_name=name.encode(),
                result_types=[out_type],
                operands=[theta, times, m_operand, n_operand],
                operand_layouts=[theta_layout, times_layout, [], []],  # scalars have empty layout
                result_layouts=[times_layout],
            )
            
            return call_op.results
        
        mlir.register_lowering(prim, lowering, platform="cpu")

        def jvp(primals, tangents):
            theta, times = primals
            dtheta, dtimes = tangents

            f = prim.bind
            eps = 1e-5
            f0 = f(theta, times)

            if dtheta is not None:
                # Use a simpler gradient computation to avoid batching issues
                def compute_gradient():
                    grad_list = []
                    for i in range(theta.shape[0]):
                        theta_plus = theta.at[i].add(eps)
                        theta_minus = theta.at[i].add(-eps)
                        grad_i = (f(theta_plus, times) - f(theta_minus, times)) / (2 * eps)
                        grad_list.append(dtheta[i] * grad_i)
                    return jnp.sum(jnp.stack(grad_list), axis=0)
                
                grad_theta = compute_gradient()
            else:
                grad_theta = jnp.zeros_like(times)

            return f0, grad_theta

        ad.primitive_jvps[prim] = jvp

        # Add batching rule for vmap support that uses compiled version
        def batching_rule(batched_args, batch_dims):
            theta, times = batched_args
            theta_bdim, times_bdim = batch_dims
            
            # Use a simpler batching approach that avoids recursion
            # and lets JAX handle the batching through the compiled primitive
            
            if theta_bdim is not None and times_bdim is not None:
                # Both are batched - this is complex, fall back to manual iteration
                if theta_bdim != times_bdim:
                    theta = jnp.moveaxis(theta, theta_bdim, 0)
                    times = jnp.moveaxis(times, times_bdim, 0)
                    batch_size = theta.shape[0]
                    results = []
                    for i in range(batch_size):
                        results.append(prim.bind(theta[i], times[i]))
                    return jnp.stack(results), 0
                else:
                    batch_size = theta.shape[theta_bdim]
                    results = []
                    for i in range(batch_size):
                        theta_i = jnp.take(theta, i, axis=theta_bdim)
                        times_i = jnp.take(times, i, axis=times_bdim)
                        results.append(prim.bind(theta_i, times_i))
                    return jnp.stack(results, axis=theta_bdim), theta_bdim
                
            elif theta_bdim is not None:
                # Only theta is batched
                batch_size = theta.shape[theta_bdim]
                results = []
                for i in range(batch_size):
                    theta_i = jnp.take(theta, i, axis=theta_bdim)
                    results.append(prim.bind(theta_i, times))
                return jnp.stack(results, axis=theta_bdim), theta_bdim
                
            elif times_bdim is not None:
                # Only times is batched - this is the common case for vmap over data
                batch_size = times.shape[times_bdim]
                results = []
                for i in range(batch_size):
                    times_i = jnp.take(times, i, axis=times_bdim)
                    results.append(prim.bind(theta, times_i))
                return jnp.stack(results, axis=times_bdim), times_bdim
                
            else:
                # Neither is batched - shouldn't happen in batching rule
                return prim.bind(theta, times), None

        # Register the batching rule
        from jax._src.interpreters import batching
        batching.primitive_batchers[prim] = batching_rule

        def wrapper(theta, times):
            return prim.bind(theta, times)

        return wrapper
    return decorator

# ----------------------
# Register DPH kernel
# ----------------------

def python_dph_pmf(theta, times):
    """JAX-compatible Python implementation of DPH PMF"""

    print("<< python fallback >>")

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
    
    # Handle both scalar and array inputs
    if jnp.ndim(times) == 0:
        # Scalar input - call directly
        return compute_pmf_single(times)
    else:
        # Array input - vectorize over all times
        return jax.vmap(compute_pmf_single)(times)

# For now, just use the Python implementation directly
# This avoids the custom call registration issues
# dph_pmf = python_dph_pmf

_dph_pmf = register_dph_kernel(
    "jax_graph_method_pmf", 
   fallback=python_dph_pmf
)(lambda theta, times: None)  # <- to register the two positional arguments


if __name__ == "__main__":

    data = jnp.array([1, 2, 3], dtype=jnp.int64)
    print(f"Times: {data}")

    _N, _u = 1000, 1/10 # just pop size times mut rate for testing (nonsensical of cause)

    # ----------------------

    # Create a valid DPH parameter vector
    m = 2
    alpha = jnp.array([1.0, 0.0]) 
    T = jnp.array([[1.0-(3/_N/_u), 3/_N/_u], 
                [0.0, 1.0-(1/_N/_u)]])    
    absorb = jnp.array([0.0, 1/_N/_u])    
    # Concatenate into theta vector
    theta = jnp.concatenate([alpha, T.flatten(), absorb])

    print(theta.shape, data.shape)


    @jax.jit
    def log_pmf(theta, z):
        prob = _dph_pmf(theta, z)
        return jnp.log(jnp.maximum(prob, 1e-12))  # Return array, not summed scalar

    @jax.jit
    def logp_direct(_theta):
        # Call the primitive directly on the full data array
        # This should trigger MLIR lowering instead of batching rule
        log_probs = log_pmf(_theta, data)  # Pass full array to primitive
        return jnp.sum(log_probs)
    
    @jax.jit
    def logp(_theta):
        return jnp.sum(jax.vmap(lambda x: log_pmf(_theta, x))(data)) 

    print("Array input:")
    print(f"  log_pmf: {log_pmf(theta, data)}")
    print("Individual scalar inputs:")
    for i, x in enumerate(data):
        result = log_pmf(theta, x)
        print(f"  log_pmf(theta, {x}): {result}")

    print("Array input:")
    print(f"  logp (direct): {logp_direct(theta)}")
    print(f"  logp (vmap): {logp(theta)}")
    print("Gradients:")
    print(f"  logp: {jax.grad(logp)(theta)}")


    print('\n----------------------\n')
    ##########################################################################
    ##########################################################################

    from functools import cache

    theta = jnp.array([_N, _u], dtype=jnp.float64)  # Ensure double precision


    
    def build_graph(theta, cache_key=None): 
        # cache_key is a dummy static argument that only serve to make jax
        # cache on those values and not just on shape and dtype of theta

        _N, _u = theta

        print('"building" graph\n')

        # def callback(state):
        #     # uses theta to build the graph
        #     ...

        # graph = Graph(callback=callback)
        # return graph


        alpha = jnp.array([1.0, 0.0]) 
        T = jnp.array([[1.0-(3/_N/_u), 3/_N/_u], 
                    [0.0, 1.0-(1/_N/_u)]])    
        absorb = jnp.array([0.0, 1/_N/_u])    


        # Concatenate into theta vector
        theta = jnp.concatenate([alpha, T.flatten(), absorb])
        return theta

    # Functions are automatically cached by JAX based on the shape and dtype of
    # x, not the content. This internal caching is handled by JAX, not by the
    # user. To cache based on metadata explicitly (and have it respected by
    # jit), you must use static_argnums:
    # FIXME: enable hashing
    # build_graph = jax.jit(build_graph, static_argnames=("cache_key",)) # maybe not needed?


    # def log_pmf(theta, z):
    #     params_impersonating_graph = jax.jit(build_graph)(theta)
    #     # params_impersonating_graph = jax.jit(build_graph, static_argnames=("cache_key",))(theta, cache_key=cache_key)

    #     prob = _dph_pmf(params_impersonating_graph, z)
    #     return jnp.sum(jnp.log(jnp.maximum(prob, 1e-12)))


    theta = jax.jit(build_graph)(theta)


    @jax.jit
    def log_pmf(theta, z):
        prob = _dph_pmf(theta, z)
        return jnp.log(jnp.maximum(prob, 1e-12))  # Return array, not summed scalar
    
    @jax.jit
    def logp_direct(_theta):
        # Call the primitive directly on the full data array
        # This should trigger MLIR lowering instead of batching rule
        log_probs = log_pmf(_theta, data)  # Pass full array to primitive
        return jnp.sum(log_probs)
    
    @jax.jit
    def logp(_theta):
        return jnp.sum(jax.vmap(lambda x: log_pmf(_theta, x))(data)) 
    

    print("Array input:")
    print(f"  log_pmf: {log_pmf(theta, data)}")
    print("Individual scalar inputs:")
    for i, x in enumerate(data):
        result = log_pmf(theta, x)
        print(f"  log_pmf(theta, {x}): {result}")
    
    print("Array input:")
    print(f"  logp (direct): {logp_direct(theta)}")
    print(f"  logp (vmap): {logp(theta)}")
    print("Gradients:")
    print(f"  logp: {jax.grad(logp)(theta)}")

    _particles = jnp.array([
        [1000.0, 1/100], 
        [100.0, 1/100], 
        [10000.0, 1/100], 
    ])
    particles = jax.vmap(build_graph)(_particles)

    print(jax.grad(logp)(theta))
    print(jax.vmap(jax.grad(logp))(particles))

    print('\n----------------------\n')
    ##########################################################################
    ##########################################################################

    from ptdalgorithms import Graph

    def build_graph(theta, cache_key=None): 
        # cache_key is a dummy static argument that only serve to make jax
        # cache on those values and not just on shape and dtype of theta
        _N, _u = theta
        print('"building" graph\n')

        def coalescent(state):
            N, R, M = theta

            if not len(state):
                initial=[([nr_samples]+[0]*nr_samples, 1)]
                return initial
            transitions = []
            for i in range(nr_samples):
                for j in range(i, nr_samples):            
                    same = int(i == j)
                    if same and state[i] < 2:
                        continue
                    if not same and (state[i] < 1 or state[j] < 1):
                        continue 
                    new = state[:]
                    new[i] -= 1
                    new[j] -= 1
                    new[i+j+1] += 1
                    transitions.append((new, state[i]*(state[j]-same)/(1+same)))
            return transitions

        graph = Graph(callback=callback)
        return graph

    def python_jax_dph_pmf(theta, times):
        graph = build_graph(theta)
        return graph.dph_pmf(times)

    _jax_dph_pmf = register_dph_kernel(
        "jax_dph_pmf", 
    fallback=python_jax_dph_pmf
    )(lambda theta, times: None)  # <- to register the two positional arguments

    @jax.jit
    def log_pmf(theta, z):
        graph = build_graph(theta)
        prob = _jax_dph_pmf(graph, z)
        return jnp.log(jnp.maximum(prob, 1e-12))  # Return array, not summed scalar
    
    @jax.jit
    def logp_direct(_theta):
        # Call the primitive directly on the full data array
        # This should trigger MLIR lowering instead of batching rule
        log_probs = log_pmf(_theta, data)  # Pass full array to primitive
        return jnp.sum(log_probs)
    
    @jax.jit
    def logp(_theta):
        return jnp.sum(jax.vmap(lambda x: log_pmf(_theta, x))(data)) 
    

    print("Array input:")
    print(f"  log_pmf: {log_pmf(theta, data)}")
    print("Individual scalar inputs:")
    for i, x in enumerate(data):
        result = log_pmf(theta, x)
        print(f"  log_pmf(theta, {x}): {result}")

    print("Array input:")
    print(f"  logp (direct): {logp_direct(theta)}")
    print(f"  logp (vmap): {logp(theta)}")
    print("Gradients:")
    print(f"  logp: {jax.grad(logp)(theta)}")

    _particles = jnp.array([
        [1000.0, 1/100], 
        [100.0, 1/100], 
        [10000.0, 1/100], 
    ])
    particles = jax.vmap(build_graph)(_particles)



    print(jax.grad(logp)(theta))
    print(jax.vmap(jax.grad(logp))(particles))