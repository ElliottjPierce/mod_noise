//! Contains normalized float types.

use core::ops::{Mul, MulAssign};

use bevy_math::{
    Curve,
    curve::{Ease, FunctionCurve, Interval},
};

use super::{CorolatedNoiseType, NoiseValue};

/// An `f32` in range 0..=1
#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
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

    /// Constructs a new [`UNorm`], assuming it to be in range 0..=1.
    #[inline]
    pub fn new_unchecked(value: f32) -> Self {
        Self(value)
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

impl CorolatedNoiseType<f32> for UNorm {
    #[inline]
    fn map_from(value: f32) -> Self {
        Self::new_unchecked(value.clamp(0.0, 1.0))
    }
}

impl CorolatedNoiseType<UNorm> for f32 {
    #[inline]
    fn map_from(value: UNorm) -> Self {
        value.get()
    }
}

impl Ease for UNorm {
    fn interpolating_curve_unbounded(start: Self, end: Self) -> impl Curve<Self> {
        let slope = end.0 - start.0;
        FunctionCurve::new(Interval::UNIT, move |t| Self(start.0 + slope * t))
    }
}
