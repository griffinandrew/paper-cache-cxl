// This suite pins the
// genericity claims on the lru design.
#![cfg(feature = "lru_compact_hybrid_cache")]

// Does the library actually accept a non-integer key type end to end?
use paper_cache::{PaperCache, CacheTierSize, GateConfig, MetadataModel, PaperPolicy, TieredBuffer};

// The per-object metadata model (S5): this test is not about the model, and
// its fast tier is smaller than the cache's own empty structures in the
// merged builds -- under the measured model's key ceiling it
// would refuse every key.
fn per_object() -> GateConfig {
    let mut gate = GateConfig::default();
    gate.metadata_model = MetadataModel::PerObject;
    gate
}

#[test]
fn string_keys_work_end_to_end() {
    let cache = PaperCache::<String, TieredBuffer>::new_with_gate(
        10_000_000,
        CacheTierSize::Bytes(2_000_000),
        PaperPolicy::LruCompactHybrid,
        per_object(),
    )
    .expect("construct");

    for i in 0..500 {
        let key = format!("user:{i}:profile");
        let val = vec![i as u8; 512];
        cache.set(key, &val, None).expect("set");
    }

    let hit = cache.get(&"user:42:profile".to_string()).expect("get");
    assert_eq!(hit, vec![42u8; 512]);
    assert!(cache.get(&"user:9999:profile".to_string()).is_err());

    cache.del(&"user:42:profile".to_string()).expect("del");
    assert!(cache.get(&"user:42:profile".to_string()).is_err());
}

#[test]
fn byte_vec_keys_work_too() {
    let cache =
        PaperCache::<Vec<u8>, TieredBuffer>::new_with_gate(10_000_000, CacheTierSize::Bytes(2_000_000), PaperPolicy::LruCompactHybrid, per_object())
            .expect("construct");

    cache.set(vec![0xDE, 0xAD], b"beef".as_slice(), None).expect("set");
    assert_eq!(cache.get(&vec![0xDE, 0xAD]).expect("get"), b"beef");
}

/// `Box<[u8]>` is the key type a server built on raw request bytes uses, and
/// the third byte-string type `thin_header` holds as bytes inside the tiered
/// item (`value_thin.rs`, "The key"; the other layout holds it as a `K`).
/// Either way every operation must work for it while its values move between
/// the tiers: a fast tier that holds a sixth of the values, so most keys are
/// slow and the reads promote them, keys that are not UTF-8, the empty key,
/// overwrites that change the length, a TTL, and deletes.
#[test]
fn boxed_slice_keys_work_across_the_tiers() {
    let cache = PaperCache::<Box<[u8]>, TieredBuffer>::new_with_gate(
        10_000_000,
        CacheTierSize::Bytes(64 * 1024),
        PaperPolicy::LruCompactHybrid,
        per_object(),
    )
    .expect("construct");

    const KEYS: usize = 300;

    let key = |i: usize| -> Box<[u8]> {
        if i == 0 {
            return Box::default();
        }

        let mut key = format!("user:{i}:").into_bytes();

        key.extend(std::iter::repeat_n(0x80 | (i % 0x7F) as u8, i % 37));
        key.into_boxed_slice()
    };
    let value = |i: usize, generation: u8| vec![(i as u8).wrapping_add(generation); 300 + (i * 37) % 700];

    for i in 0..KEYS {
        cache.set(key(i), &value(i, 0), None).expect("set");
    }

    // Most keys are slow: the fast tier holds a sixth of the values.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);

    while cache.hybrid_stats().slow_objects < (KEYS / 2) as u64 {
        assert!(std::time::Instant::now() < deadline, "the keys never demoted: {:?}", cache.hybrid_stats());
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    for i in 0..KEYS {
        assert_eq!(cache.get(&key(i)).expect("a live key hits"), value(i, 0), "key {i}");
    }

    // Overwrites that change the length, a TTL set on a key that is slow or
    // fast, and deletes.
    for i in (0..KEYS).step_by(3) {
        cache.set(key(i), &value(i, 1), None).expect("overwrite");
    }

    cache.ttl(&key(4), Some(3_600)).expect("a live key takes a ttl");

    for i in (1..KEYS).step_by(3) {
        cache.del(&key(i)).expect("del");
    }

    for i in 0..KEYS {
        let read = cache.get(&key(i));

        match i % 3 {
            0 => assert_eq!(read.expect("overwritten"), value(i, 1), "key {i}"),
            1 => assert!(read.is_err(), "key {i} was deleted"),
            _ => assert_eq!(read.expect("untouched"), value(i, 0), "key {i}"),
        }
    }

    assert!(cache.get(&Box::<[u8]>::from(&b"never set"[..])).is_err());
}

/// A key type that is deliberately NOT Debug: constructing a cache with it is
/// the proof that no internal path formats keys.
#[derive(Clone, PartialEq, Eq, Hash)]
struct OpaqueKey([u8; 16]);

impl typesize::TypeSize for OpaqueKey {}

#[test]
fn keys_need_no_debug_impl() {
    let cache =
        PaperCache::<OpaqueKey, TieredBuffer>::new_with_gate(10_000_000, CacheTierSize::Bytes(2_000_000), PaperPolicy::LruCompactHybrid, per_object())
            .expect("construct");
    let k = OpaqueKey(*b"0123456789abcdef");
    cache.set(k.clone(), b"v".as_slice(), None).expect("set");
    assert_eq!(cache.get(&k).expect("get"), b"v");
}
