//! Contains noise types that are periodic.

use bevy_math::{Curve, curve::Ease};

use super::Noise;

/// Represents a [`Noise`] that divides its domain into [`PeriodicSegment`] according to some period `T`.
pub trait PeriodicNoise<T>: Noise {
    /// Gets the [`Period`](PeriodicNoise::Period) of the noise.
    fn get_period(&self) -> T;
    /// Sets the [`Period`](PeriodicNoise::Period) of the noise.
    fn set_period(&mut self, period: T);
}

/// Represents a segment of a noise result.
/// This should be a well defined subset of a domain, for example, a grid square or other polygon in 2d.
pub trait PeriodicSegment {
    /// The kind of point that bounds this segment.
    type Point: PeriodicPoint;
    /// The kind of [`PeriodicPoints`] that can describe each point that bounds this segment.
    type Points: PeriodicPoints<Point = Self::Point>;

    /// Gets the primary point associated with this segment.
    /// This is a way to identify a segment by a particular point, for example, the lower left corner of a grid square.
    fn get_main_point(&self) -> Self::Point;
    /// Gets the points that bound this segment.
    fn get_points(self) -> Self::Points;
}

/// This is a bounding point of a [`PeriodicSegment`] which may border multiple segments.
/// This should be a specific element of a domain, for example, a lattce point or vertex.
pub trait PeriodicPoint {
    /// The type that points to the sampling location from this point.
    type Relative;
    /// Based on this point and some `entropy`, produces a [`RelativePeriodicPoint`] that represents this point.
    fn into_relative(self, entropy: u32) -> RelativePeriodicPoint<Self::Relative>;
}

/// This is a collection of [`PeriodicPoint`]s that bound a [`PeriodicSegment`].
pub trait PeriodicPoints {
    /// The kind of point this contains.
    type Point: PeriodicPoint;

    /// Iterates these points.
    fn iter(&self) -> impl Iterator<Item = Self::Point>;
}

/// Represents some [`PeriodicPoints`] which can be sampled smoothly to interpolate between those points.
pub trait PeriodicPointsSampler: PeriodicPoints {
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
        Self(2.0)
    }
}

impl From<Frequency> for Period {
    #[inline]
    fn from(value: Frequency) -> Self {
        Self(1.0 / value.0)
    }
}

impl From<WholePeriod> for Period {
    #[inline]
    fn from(value: WholePeriod) -> Self {
        Self(value.0 as f32)
    }
}

impl From<PowerOf2Period> for Period {
    #[inline]
    fn from(value: PowerOf2Period) -> Self {
        Self::from(Into::<WholePeriod>::into(value))
    }
}

/// Represents a frequency of this value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frequency(pub f32);

impl Default for Frequency {
    #[inline]
    fn default() -> Self {
        Self(0.5)
    }
}

impl From<Period> for Frequency {
    #[inline]
    fn from(value: Period) -> Self {
        Self(1.0 / value.0)
    }
}

impl From<WholePeriod> for Frequency {
    #[inline]
    fn from(value: WholePeriod) -> Self {
        Self::from(Into::<Period>::into(value))
    }
}

impl From<PowerOf2Period> for Frequency {
    #[inline]
    fn from(value: PowerOf2Period) -> Self {
        Self::from(Into::<Period>::into(value))
    }
}

/// Represents a period of this value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WholePeriod(pub u32);

impl From<PowerOf2Period> for WholePeriod {
    #[inline]
    fn from(value: PowerOf2Period) -> Self {
        Self(1u32 << value.0)
    }
}

impl Default for WholePeriod {
    #[inline]
    fn default() -> Self {
        Self(2)
    }
}

/// Represents a period of 2 ^ of this value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowerOf2Period(pub u32);

impl Default for PowerOf2Period {
    #[inline]
    fn default() -> Self {
        Self(1)
    }
}
