#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif

// JAX custom call signature with opaque data
__attribute__((visibility("default")))
void dph_pmf_param(void* out_ptr, void** in_ptrs, const char* opaque, size_t opaque_len);

#ifdef __cplusplus
}
#endif