# Background Tiering Worker Integration

This implementation adds an event-driven background tiering worker to the paper-cache library for managing DRAM/PMEM tiering based on access patterns.

## Architecture

The tiering system follows the same pattern as existing workers (PolicyWorker and TtlWorker):

```
PaperCache
    ├── WorkerManager (manages all workers)
    │   ├── PolicyWorker
    │   ├── TtlWorker
    │   └── TieringWorker (NEW)
    │       └── TieringManager (tracks access patterns and makes promotion/demotion decisions)
    └── access_event_sender (bounded channel for non-blocking access event communication)
```

## Key Components

### 1. TieringManager
Located in `src/worker/tiering.rs`, this component:
- Tracks access counts for each object
- Maintains DRAM object set and size accounting
- Implements promotion threshold logic (2 accesses)
- Implements demotion logic based on high/low water marks
- Provides APIs: `record_access()`, `should_promote()`, `promote_to_dram()`, `demote_from_dram()`, `demote_until_low_water()`

### 2. TieringWorker
The background worker thread that:
- Receives access events via bounded channel (capacity: 10,000)
- Batches events (up to 1024 keys or 100ms timeout)
- Deduplicates accesses within batches
- Executes lazy promotions when threshold is crossed
- Triggers demotion after each batch if DRAM exceeds high water mark

### 3. Hot-Path Integration
In `PaperCache::get()`:
- Uses `try_send()` to send AccessEvent (non-blocking)
- Drops events if channel is full (acceptable per requirements)
- Zero blocking in the hot path

## Configuration

Water marks are automatically calculated as percentages of max cache size:
- High water mark: 80% of max_size (triggers demotion)
- Low water mark: 60% of max_size (demotion target)

Promotion threshold: 2 accesses (hardcoded)

## Channel Details

- **Access Event Channel**: `crossbeam_channel::bounded(10_000)`
  - Hot path uses `try_send()` - never blocks
  - Events dropped on full are acceptable (graceful degradation)
- **Worker Event Channel**: `crossbeam_channel::unbounded()`
  - Used for control events (Wipe, etc.)
  - Same pattern as PolicyWorker and TtlWorker

## Batch Processing

The worker collects events in batches to reduce overhead:
1. Collects up to 1024 events OR waits 100ms (whichever comes first)
2. Deduplicates and tallies access counts per key
3. Updates global access counts in TieringManager
4. Promotes objects that cross the threshold
5. After batch processing, demotes coldest objects if needed

## Thread Safety

- TieringManager uses `parking_lot::RwLock` for access count and DRAM object tracking
- Uses `AtomicU64` for DRAM size accounting
- Worker runs in dedicated thread spawned by `register_worker()`

## Example Flow

```
1. User calls cache.get(&key)
2. Hot path sends AccessEvent::Get(hashed_key) via try_send (non-blocking)
3. Background worker collects events in batch
4. Worker deduplicates: key X accessed 3 times in this batch
5. Worker updates TieringManager: total count for X is now 5
6. Worker checks: count >= 2 && not in DRAM && not pending → promote
7. Worker promotes X to DRAM (updates accounting)
8. After batch: checks DRAM size > high_water_mark
9. If yes: demotes coldest objects until DRAM <= low_water_mark
```

## Testing

Unit tests are provided in `src/worker/tiering.rs` covering:
- Promotion threshold logic
- Demotion with access count tracking
- Batch deduplication
- Water mark enforcement

## Future Enhancements

- Make promotion threshold configurable
- Make water marks configurable
- Add metrics/telemetry for promotion/demotion events
- Implement actual memory migration (currently just accounting)
- Add support for different demotion policies (LRU, LFU, etc.)
