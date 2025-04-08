//! Contains noise types that are periodic.

use bevy_math::{Curve, UVec2, Vec2, curve::Ease};

use super::DirectNoise;

/// Represents a [`Noise`] that divides its domain into [`PeriodicSegment`].
pub trait PeriodicNoise<P>: DirectNoise<P, Output: PeriodicSegment<P>> {
    /// The interval over which values may repeat.
    type Period: Clone + 'static;

    /// Gets the [`Period`](PeriodicNoise::Period) of the noise.
    fn get_period(&self) -> Self::Period;
    /// Sets the [`Period`](PeriodicNoise::Period) of the noise.
    fn set_period(&mut self, period: Self::Period);

    /// Gets the [`Period`](PeriodicNoise::Period) of the noise.
    fn get_frequency(&self) -> Self::Period
    where
        Self::Period: PeriodAndFrequency;
    /// Sets the [`Period`](PeriodicNoise::Period) of the noise.
    fn set_frequency(&mut self, frequency: Self::Period)
    where
        Self::Period: PeriodAndFrequency;

    /// Sets the [`Period`](PeriodicNoise::Period) of the noise.
    #[inline]
    fn with_period(mut self, period: Self::Period) -> Self
    where
        Self: Sized,
    {
        self.set_period(period);
        self
    }

    /// Sets the [`Period`](PeriodicNoise::Period) of the noise.
    #[inline]
    fn with_frequency(mut self, frequency: Self::Period) -> Self
    where
        Self::Period: PeriodAndFrequency,
        Self: Sized,
    {
        self.set_frequency(frequency);
        self
    }
}

/// Represents a period that can be inverted into a frequency.
pub trait PeriodAndFrequency {
    /// Converts this period to a frequency.
    fn period_to_frequency(self) -> Self;
    /// Converts this frequency to a period.
    fn frequency_to_period(self) -> Self;
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

impl PeriodAndFrequency for f32 {
    #[inline]
    fn period_to_frequency(self) -> Self {
        1.0 / self
    }

    #[inline]
    fn frequency_to_period(self) -> Self {
        1.0 / self
    }
}

/// A [`PeriodicNoise`] that produces [`GridSquare`]
pub struct OrthoGrid {
    frequency: f32,
}

/// Represents a grid square.
pub struct GridSquare<Z, R> {
    /// The least corner of this grid square.
    pub least_corner: Z,
    /// The positive offset from [`least_corner`](Self::least_corner) to the point in the grid square.
    pub offset_from_corner: R,
}

pub struct OrthoGridLattacePoint<Z, R> {
    /// Some corner of a [`GridSquare`].
    pub corner: Z,
    /// The offset from [`corner`](Self::corner) to the point in the [`GridSquare`].
    pub offset: R,
}

macro_rules! impl_grid_dimension {
    ($u:ty, $s:ty, $f:ty, $f_to_u:ident) => {
        impl PeriodicPoint<$f> for OrthoGridLattacePoint<$u, $f> {
            #[inline]
            fn into_relative(self, entropy: u32) -> RelativePeriodicPoint<$f> {
                RelativePeriodicPoint {
                    offset: self.offset,
                    seed: White32(entropy).sample(self.corner),
                }
            }
        }

        impl PeriodicPoints<$f> for GridSquare<$u, $f> {
            #[inline]
            fn into_relative(self, entropy: u32) -> RelativePeriodicPoint<$f> {
                RelativePeriodicPoint {
                    offset: self.offset,
                    seed: White32(entropy).raw_sample(self.corner),
                }
            }
        }

        impl DirectNoise<$f> for OrthoGrid {
            type Output = GridSquare<$u, $f>;

            #[inline]
            fn raw_sample(&self, input: $f) -> Self::Output {
                let scaled = input * self.frequency;
                GridSquare {
                    least_corner: scaled.$f_to_u(),
                    offset_from_corner: scaled.fract_gl(),
                }
            }
        }
    };
}
