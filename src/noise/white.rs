//! This module implements white noise inspiered by the [FxHash](https://crates.io/crates/fxhash)

use bevy_math::{
    U8Vec2, U8Vec3, U8Vec4, U16Vec2, U16Vec3, U16Vec4, U64Vec2, U64Vec3, U64Vec4, UVec2, UVec3,
    UVec4,
};

use super::{DirectNoise, Noise, NoiseExt};

/// This creates a white noise implementation
macro_rules! impl_white {
    ($dt:ty, $name:ident, $key:expr, $(($input:ty, $conv:ty)),* $(,),*) => {
        /// A seeded RNG inspired by [FxHash](https://crates.io/crates/fxhash).
        /// This is similar to a hash function, but does not use std's hash traits, as those produce `u64` outputs only.
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
        pub struct $name(pub $dt);

        impl DirectNoise<$dt> for $name {
            type Output = $dt;

            #[inline(always)]
            fn raw_sample(&self, input: $dt) -> $dt {
                    (input ^ self.0) // salt with the seed
                    .wrapping_mul($key) // multiply to remove any linear artifacts
            }
        }

        impl<const N: usize> DirectNoise< [$dt; N] > for $name {
            type Output = $dt;

            #[inline(always)]
            fn raw_sample(&self, input: [$dt; N]) -> $dt {
                let slice: &[$dt] = &input;
                self.raw_sample(slice)
            }
        }

        impl DirectNoise< Option<$dt> > for $name {
            type Output = $dt;

            #[inline(always)]
            fn raw_sample(&self, input: Option<$dt>) -> $dt {
                if let Some(input) = input {
                    self.raw_sample([input])
                } else {
                    $key
                }
            }
        }

        impl DirectNoise<&'_ [$dt]> for $name {
            type Output = $dt;

            #[inline(always)]
            fn raw_sample(&self, input: &[$dt]) -> $dt {
                let mut val: $dt = $key;
                for &v in input {
                    // The breaker value must depended on both the `v` and the `val` to prevent it getting stuck.
                    // We need addition to keep this getting stuck when `v` or `val` are 0.
                    let breaker = (v ^ val).wrapping_add($key);
                    // We need the multiplication to put each axis on different orders, and we need xor to make each axis "recoverable" from zero.
                    // The multiplication can be pipelined with computing the `breaker`. Effectively the cost is just multiplication.
                    val = v.wrapping_mul(val) ^ breaker;
                }
                self.raw_sample(val)
            }
        }

        $(
            impl DirectNoise< $input > for $name {
                type Output = $dt;

                #[inline(always)]
                fn raw_sample(&self, input: $input) -> $dt {
                    let inner: $conv = input.into();
                    self.raw_sample(inner)
                }
            }
        )*
    };
}

// uses some very large primes I found on the internet
impl_white!(
    u8,
    White8,
    97,
    (U8Vec2, [u8; 2]),
    (U8Vec3, [u8; 3]),
    (U8Vec4, [u8; 4]),
);
impl_white!(
    u16,
    White16,
    1777,
    (U16Vec2, [u16; 2]),
    (U16Vec3, [u16; 3]),
    (U16Vec4, [u16; 4]),
);
impl_white!(
    u32,
    White32,
    104_395_303,
    (UVec2, [u32; 2]),
    (UVec3, [u32; 3]),
    (UVec4, [u32; 4]),
);
impl_white!(
    u64,
    White64,
    982_451_653,
    (U64Vec2, [u64; 2]),
    (U64Vec3, [u64; 3]),
    (U64Vec4, [u64; 4]),
);

impl_white!(u128, White128, 982_451_653_011,);

#[cfg(target_pointer_width = "32")]
impl_white!(usize, WhiteUsize, 104_395_303,);
#[cfg(target_pointer_width = "64")]
impl_white!(usize, WhiteUsize, 982_451_653,);

#[cfg(target_pointer_width = "32")]
impl Noise for WhiteUsize {
    #[inline]
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        self.0 = seed.next_seed() as usize;
    }
}

#[cfg(target_pointer_width = "64")]
impl Noise for WhiteUsize {
    #[inline]
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        self.0 = White64(0).with_seed(seed).0 as usize;
    }
}

impl Noise for White8 {
    #[inline]
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        self.0 = seed.next_seed() as u8;
    }
}

impl Noise for White16 {
    #[inline]
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        self.0 = seed.next_seed() as u16;
    }
}

impl Noise for White32 {
    #[inline]
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        self.0 = seed.next_seed();
    }
}

impl Noise for White64 {
    #[inline]
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        self.0 = seed.next_seed() as u64 | ((seed.next_seed() as u64) << 32);
    }
}

impl Noise for White128 {
    #[inline]
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        self.0 = seed.next_seed() as u128
            | ((seed.next_seed() as u128) << 32)
            | ((seed.next_seed() as u128) << 64)
            | ((seed.next_seed() as u128) << 96);
    }
}

/// A light weight seed generator.
/// This is a stripped down version of an Rng.
pub struct SeedGenerator {
    seed: White32,
    entropy: u32,
}

impl SeedGenerator {
    /// Gets the next seed in the generator.
    #[inline]
    pub fn next_seed(&mut self) -> u32 {
        let next_seed = self.seed.raw_sample(self.entropy);
        self.entropy = self.entropy.wrapping_add(1);
        next_seed
    }

    /// Creates a different [`SeedGenerator`] that will yield values independent of this one.
    #[inline]
    pub fn branch(&mut self) -> Self {
        Self::new(self.next_seed(), self.next_seed())
    }

    /// Creates a [`SeedGenerator`] with standard entropy from a seed.
    #[inline]
    pub fn new_from_seed(seed: u32) -> Self {
        Self::new(seed, 0)
    }

    /// Creates a [`SeedGenerator`] with this entropy and seed.
    #[inline]
    pub fn new(seed: u32, entropy: u32) -> Self {
        Self {
            seed: White32(seed),
            entropy,
        }
    }

    /// Creates a [`SeedGenerator`] with entropy and seed from these `bits`.
    #[inline]
    pub fn new_from_u64(bits: u64) -> Self {
        Self::new((bits >> 32) as u32, bits as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_u32() {
        let rng = White32(5);
        let _tmp = rng.raw_sample(8);
        let _tmp = rng.raw_sample([8, 2]);
        let _tmp = rng.raw_sample([8, 2, 4]);
        let _tmp = rng.raw_sample([8, 2, 9, 3]);
        let _tmp = rng.raw_sample(UVec2::new(1, 2));
        let _tmp = rng.raw_sample(UVec3::new(1, 2, 3));
        let _tmp = rng.raw_sample(UVec4::new(1, 2, 3, 4));
        let _tmp = rng.raw_sample(alloc::vec![1u32, 2, 3, 4, 5].as_slice());
    }
}
