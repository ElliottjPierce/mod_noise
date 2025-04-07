#![no_std]
#![doc = include_str!("../README.md")]

pub mod noise;

#[cfg(feature = "alloc")]
extern crate alloc;
