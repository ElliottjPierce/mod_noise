//! Contains a variety of curves of domain and range [0, 1].

use core::ops::{Add, Mul, Sub};

use bevy_math::{Curve, curve::Interval};

/// Linear interpolation between two `T`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LerpCurve<T> {
    /// The starting point of the curve.
    pub start: T,
    /// The slope of the curve.
    pub diff: T,
}

/// Allows easily creating a [`LerpCurve`].
pub trait Lerpable: Sized {
    /// Creates a [`LerpCurve`] that lerps from `self` to `end`.
    fn lerp_to(self, end: Self) -> impl Curve<Self>;
}

impl<T: Copy + Sub<Self, Output = Self> + Add<Self, Output = Self> + Mul<f32, Output = Self>>
    Lerpable for T
{
    #[inline]
    fn lerp_to(self, end: Self) -> impl Curve<Self> {
        LerpCurve::new(self, end)
    }
}

impl<T: Copy + Sub<T, Output = T>> LerpCurve<T> {
    /// Lerps from `start` to `end`.
    #[inline]
    pub fn new(start: T, end: T) -> Self {
        Self {
            start,
            diff: end - start,
        }
    }
}

impl<T: Copy + Add<T, Output = T> + Mul<f32, Output = T>> Curve<T> for LerpCurve<T> {
    #[inline]
    fn domain(&self) -> Interval {
        Interval::UNIT
    }

    #[inline]
    fn sample_unchecked(&self, t: f32) -> T {
        self.start + (self.diff * t)
    }
}

/// Linear interpolation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Linear;

impl Curve<f32> for Linear {
    #[inline]
    fn domain(&self) -> Interval {
        Interval::EVERYWHERE
    }

    #[inline]
    fn sample_unchecked(&self, t: f32) -> f32 {
        t
    }
}

/// Smoothstep interpolation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Smoothstep;

impl Curve<f32> for Smoothstep {
    #[inline]
    fn domain(&self) -> Interval {
        Interval::UNIT
    }

    #[inline]
    fn sample_unchecked(&self, t: f32) -> f32 {
        (3.0 * t * t) - (2.0 * t * t * t)
    }
}
