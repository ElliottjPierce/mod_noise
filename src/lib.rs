#![no_std]
#![allow(
    clippy::doc_markdown,
    reason = "These rules should not apply to the readme."
)]
#![doc = include_str!("../README.md")]

pub mod noise;

#[cfg(test)]
extern crate alloc;
