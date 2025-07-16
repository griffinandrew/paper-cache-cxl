/*
 * Copyright (c) Kia Shakiba
 *
 * This source code is licensed under the GNU AGPLv3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub mod overhead;

use std::{
	mem,
	sync::Arc,
	time::{Instant, Duration},
};

use typesize::TypeSize;

pub type ObjectSize = u32;
pub type ExpireTime = Option<Instant>;

pub struct Object<K, V> {
	key: K,
	//data: Arc<V>,
	data: CxlPtr<V>,


	expiry: ExpireTime,
}

impl<K, V> Object<K, V> {
	pub fn new(key: K, data: V, ttl: Option<u32>) -> Self
	//this is new and I am not sure if it is right... bc i am now forcing a type?? must have been reason kia didnt have.... 
	where
		K: TypeSize,
		V: TypeSize,
	{
		let expiry = match ttl {
			Some(0) | None => None,
			Some(ttl) => Some(get_expiry_from_ttl(ttl)),
		};

		let cxl_data = CxlPtr::new(data);

		let new_obj = Object {
			key,
			data:cxl_data,
			expiry,
		};
		
		new_obj                
	}


	pub fn data(&self) -> &CxlPtr<V>
		where
		K: TypeSize,
		V: TypeSize,
	{
		&self.data
	}


	pub fn key_matches(&self, key: &K) -> bool
	where
		K: Eq,
	{
		self.key.eq(key)
	}

	fn total_size(&self) -> ObjectSize
	where
		K: TypeSize,
		V: TypeSize,
	{
		(
			self.key.get_size()
				+ self.data.get_size()
				+ mem::size_of::<ExpireTime>()
		) as ObjectSize
	}

	pub fn expiry(&self) -> ExpireTime {
		self.expiry
	}

	pub fn is_expired(&self) -> bool {
		self.expiry.is_some_and(|expiry| expiry <= Instant::now())
	}

	pub fn expires(&mut self, ttl: Option<u32>) {
		self.expiry = match ttl {
			Some(0) | None => None,
			Some(ttl) => Some(get_expiry_from_ttl(ttl)),
		};
	}
}

pub fn get_expiry_from_ttl(ttl: u32) -> Instant {
	Instant::now() + Duration::from_secs(ttl.into())
}



// ****************************

// CXL PTR 

use std::hint::black_box;
use std::ops::Deref;

// ****************************


#[derive(Debug, PartialEq, Eq)]
pub struct CxlPtr<T> {
    inner: Arc<T>,
}

impl<T> CxlPtr<T> {

    pub fn new(inner: T) -> Self {
        CxlPtr {
            inner: Arc::new(inner),
        }
    }

    pub fn get_inner(&self) -> &T {
        &self.inner
    }

	pub fn get_size(&self) -> usize 
	where
		T: TypeSize,
	{
		//size of inner (data and arc) + size of CxlPtr itself
		//i believe this is correct, but could be wrong
		let size = self.inner.get_size() + mem::size_of::<CxlPtr<T>>();
		size
	}

	pub fn get_arc(&self) -> Arc<T> {
        Arc::clone(&self.inner)
    }

}

impl <T> Clone for CxlPtr<T> {

    fn clone(&self) -> Self {
		//println!("CxlPtr clone called from:\n{:?}", std::backtrace::Backtrace::capture());
        CxlPtr {
            inner: Arc::clone(&self.inner),
        }
    }
}

//this might be unsafe behavior, I think i would be violating safety rules but complier never complains
impl<T: TypeSize> Deref for CxlPtr<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target 
	where 
	T: TypeSize,
	{	
		//returns a reference to the inner data that is the arc so &arc
		let target = black_box(&*self.inner);
		target
    }
}


