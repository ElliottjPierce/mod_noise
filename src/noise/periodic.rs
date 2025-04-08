//! Contains noise types that are periodic.

use bevy_math::{Curve, curve::Ease};

use super::Noise;

/// Represents a [`Noise`] that divides its domain into [`PeriodicSegment`] according to some period `T`.
pub trait PeriodicNoise<T>: Noise {
    /// Gets the [`Period`](PeriodicNoise::Period) of the noise.
    fn get_period(&self) -> T;
    /// Sets the [`Period`](PeriodicNoise::Period) of the noise.
    fn set_period(&mut self, period: T);

    /// Sets the [`Period`](PeriodicNoise::Period) of the noise.
    #[inline]
    fn with_period(mut self, period: T) -> Self
    where
        Self: Sized,
    {
        self.set_period(period);
        self
    }
}

/// Represents a segment of a noise result.
/// This should be a well defined subset of a domain, for example, a grid square or other polygon in 2d.
///
/// `P` defines the type of the domain, for example [`Vec2`] in 2d.
pub trait PeriodicSegment<P> {
    /// The kind of point that bounds this segment.
    type Point: PeriodicPoint<P>;
    /// The kind of [`PeriodicPoints`] that can describe each point that bounds this segment.
    type Points: PeriodicPoints<P, Point = Self::Point>;

    /// Gets the primary point associated with this segment.
    /// This is a way to identify a segment by a particular point, for example, the lower left corner of a grid square.
    fn get_main_point(&self) -> Self::Point;
    /// Gets the points that bound this segment.
    fn get_points(self) -> Self::Points;
}

/// This is a bounding point of a [`PeriodicSegment`] which may border multiple segments.
/// This should be a specific element of a domain, for example, a lattce point or vertex.
///
/// `P` defines the type of the domain, for example [`Vec2`] in 2d.
pub trait PeriodicPoint<P> {
    /// Based on this point and some `entropy`, produces a [`RelativePeriodicPoint`] that represents this point.
    fn into_relative(self, entropy: u32) -> RelativePeriodicPoint<P>;
}

/// This is a collection of [`PeriodicPoint`]s that bound a [`PeriodicSegment`].
///
/// `P` defines the type of the domain, for example [`Vec2`] in 2d.
pub trait PeriodicPoints<P> {
    /// The kind of point this contains.
    type Point: PeriodicPoint<P>;

    /// Iterates these points.
    fn iter(&self) -> impl Iterator<Item = Self::Point>;
}

/// Represents some [`PeriodicPoints`] which can be sampled smoothly to interpolate between those points.
pub trait PeriodicPointsSampler<P>: PeriodicPoints<P> {
    /// Interpolates between these points, producing some result.
    /// The bounds of `curve` are not checked.
    /// It is up to the caller to verify that they are valid for this domain.
    fn sample_smooth<T: Ease>(&self, f: impl FnMut(Self::Point) -> T, curve: impl Curve<f32>) -> T;
}

/// Represents some [`PeriodicPoint`] locally by and offset and seed.
pub struct RelativePeriodicPoint<P> {
    /// The offset of the location in question relative to this point.
    pub offset: P,
    /// The loosely identifying seed of this point.
    pub seed: u32,
}

/// Represents a period of this value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Period(pub f32);

impl Default for Period {
    #[inline]
    fn default() -> Self {
        Self(1.0)
    }
}

impl From<Frequency> for Period {
    #[inline]
    fn from(value: Frequency) -> Self {
        Self(1.0 / value.0)
    }
}

/// Represents a frequency of this value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frequency(pub f32);

impl Default for Frequency {
    #[inline]
    fn default() -> Self {
        Self(1.0)
    }
}

impl From<Period> for Frequency {
    #[inline]
    fn from(value: Period) -> Self {
        Self(1.0 / value.0)
    }
}
