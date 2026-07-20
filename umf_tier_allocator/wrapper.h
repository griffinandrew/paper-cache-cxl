#ifndef TIER_ALLOCATOR_WRAPPER_H
#define TIER_ALLOCATOR_WRAPPER_H

#include <umf.h>
#include <umf/memory_pool.h>
#include <umf/memory_provider.h>
#include <umf/providers/provider_os_memory.h>
#include <umf/pools/pool_scalable.h>

#ifdef TIER_ALLOCATOR_WITH_JEMALLOC_POOL
#include <umf/pools/pool_jemalloc.h>
#endif

#endif
