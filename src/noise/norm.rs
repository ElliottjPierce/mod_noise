//! Contains normalized float types.

use core::ops::{Mul, MulAssign};

use super::{CorolatedNoiseType, NoiseValue};

/// An `f32` in range [0, 1)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct UNorm(f32);

impl NoiseValue for UNorm {}

impl UNorm {
    /// Constructs a [`UNorm`] from arbetrary bits, returning unused `u8` bits.
    #[inline]
    pub fn random_with_entropy(bits: u32) -> (Self, u8) {
        // adapted from rand's `StandardUniform`

        let fraction_bits = 23;
        let float_size = size_of::<f32>() as u32 * 8;
        let precision = fraction_bits + 1;
        let scale = 1f32 / ((1u32 << precision) as f32);

        let value = bits >> (float_size - precision);
        (Self(scale * value as f32), bits as u8)
    }

    /// Constructs a [`UNorm`] from arbetrary bits.
    #[inline]
    pub fn random(bits: u32) -> Self {
        Self::random_with_entropy(bits).0
    }

    /// Gets the inner `f32` value.
    #[inline]
    pub fn get(self) -> f32 {
        self.0
    }
}

impl Mul<UNorm> for UNorm {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: UNorm) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl MulAssign<UNorm> for UNorm {
    #[inline]
    fn mul_assign(&mut self, rhs: UNorm) {
        self.0 *= rhs.0;
    }
}

impl CorolatedNoiseType<u32> for UNorm {
    #[inline]
    fn map_from(value: u32) -> Self {
        Self::random(value)
    }
}
