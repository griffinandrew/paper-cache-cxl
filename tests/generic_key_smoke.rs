// Constructor signatures differ per hybrid feature; this suite pins the
// genericity claims on the lru design.
#![cfg(feature = "lru_hybrid_cache")]

// Does the library actually accept a non-integer key type end to end?
use paper_cache::{PaperCache, CacheTierSize, TieredBuffer};

#[test]
fn string_keys_work_end_to_end() {
    let cache = PaperCache::<String, TieredBuffer>::new(
        10_000_000,
        CacheTierSize::Bytes(2_000_000),
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
        PaperCache::<Vec<u8>, TieredBuffer>::new(10_000_000, CacheTierSize::Bytes(2_000_000))
            .expect("construct");

    cache.set(vec![0xDE, 0xAD], b"beef".as_slice(), None).expect("set");
    assert_eq!(cache.get(&vec![0xDE, 0xAD]).expect("get"), b"beef");
}

/// A key type that is deliberately NOT Debug: constructing a cache with it is
/// the proof that no internal path formats keys.
#[derive(Clone, PartialEq, Eq, Hash)]
struct OpaqueKey([u8; 16]);

impl typesize::TypeSize for OpaqueKey {}

#[test]
fn keys_need_no_debug_impl() {
    let cache =
        PaperCache::<OpaqueKey, TieredBuffer>::new(10_000_000, CacheTierSize::Bytes(2_000_000))
            .expect("construct");
    let k = OpaqueKey(*b"0123456789abcdef");
    cache.set(k.clone(), b"v".as_slice(), None).expect("set");
    assert_eq!(cache.get(&k).expect("get"), b"v");
}
