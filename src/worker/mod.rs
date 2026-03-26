/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

mod manager;
mod policy;
mod ttl;

#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
mod tiering;

use crossbeam_channel::{Receiver, Sender};
use std::thread;

use crate::{
    CacheSize, HashedKey,
    error::CacheError,
    object::{ExpireTime, ObjectSize},
    policy::PaperPolicy,
};

pub type WorkerSender = Sender<WorkerEvent>;
pub type WorkerReceiver = Receiver<WorkerEvent>;

#[derive(Clone)]
pub enum WorkerEvent {
    Get(HashedKey, bool),
    Promote(HashedKey),
    Set(
        HashedKey,
        ObjectSize,
        ExpireTime,
        Option<(ObjectSize, ExpireTime)>,
    ),
    Del(HashedKey, ExpireTime),

    Ttl(HashedKey, ExpireTime, ExpireTime),

    Wipe,

    Resize(CacheSize),
    Policy(PaperPolicy),
}

pub trait Worker
where
    Self: 'static + Send,
{
    fn run(&mut self) -> Result<(), CacheError>;
}

pub fn register_worker(mut worker: impl Worker) {
    thread::spawn(move || worker.run());
}

pub use crate::worker::{manager::WorkerManager, policy::PolicyWorker, ttl::TtlWorker};

#[cfg(all(feature = "key_value_pmem", feature = "enable_tiering_manager"))]
pub use crate::worker::tiering::TieringWorker;
