#ifndef PREFAULT_PROVIDER_H
#define PREFAULT_PROVIDER_H

#include <stddef.h>
#include <umf/memory_provider.h>

typedef struct prefault_params_t {
    size_t size;
    int    numa_node;
} prefault_params_t;

const umf_memory_provider_ops_t *umfPrefaultProviderOps(void);

#endif