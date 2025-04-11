//! Contains noise types that are periodic.

use bevy_math::{Curve, HasTangent, curve::derivatives::SampleDerivative};

use super::{
    DirectNoise, DirectNoiseBuilder, GradientNoise, Noise, NoiseBuilder, NoiseValue,
    white::SeedGenerator,
};

/// Represents a [`Noise`] that divides its domain into [`PeriodicSegment`] according to some period `T`.
pub trait ScalableNoise<T>: Noise {
    /// Gets the scale of the noise.
    fn get_scale(&self) -> T;
    /// Sets the scale of the noise.
    fn set_scale(&mut self, scale: T);
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
pub trait SamplablePeriodicPoints: PeriodicPoints {
    /// Interpolates between these points, producing some result.
    /// The bounds of `curve` are not checked.
    /// It is up to the caller to verify that they are valid for this domain.
    fn sample_smooth<T, L: Curve<T>>(
        &self,
        f: impl FnMut(Self::Point) -> T,
        lerp: impl Fn(T, T) -> L,
        curve: &impl Curve<f32>,
    ) -> T;
}

/// Represents some [`SamplablePeriodicPoints`] that is differentiable.
pub trait DiferentiablePeriodicPoints: SamplablePeriodicPoints {
    /// For some derivative type `D`, returns the gradient vector type that will hold each derivative.
    /// Usually this is `[D; N]`, where `N` is the dimensions of the point.
    type Gradient<D>;

    /// Interpolates between these points, producing the gradient of the interpolation.
    /// The bounds of `curve` are not checked.
    /// It is up to the caller to verify that they are valid for this domain.
    fn sample_gradient_smooth<T: HasTangent, L: Curve<T::Tangent>>(
        &self,
        f: impl FnMut(Self::Point) -> T,
        difference: impl Fn(&T, &T) -> T::Tangent,
        lerp: impl Fn(T::Tangent, T::Tangent) -> L,
        curve: &impl SampleDerivative<f32>,
    ) -> Self::Gradient<T::Tangent>;
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

impl From<Frequency> for WholePeriod {
    #[inline]
    fn from(value: Frequency) -> Self {
        Self::from(Period::from(value))
    }
}

impl From<Period> for WholePeriod {
    #[inline]
    fn from(value: Period) -> Self {
        Self(value.0 as u32)
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

/// Represents slicing a domain into [`PeriodicSegment`]s via `P` and then computing noise within segments via `N`.
/// The noise itself may not tile in the traditional sense, but it is composed of tiles of noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TilingNoise<P, N> {
    /// The noise for making the [`PeriodicSegment`]s.
    pub tilier: P,
    /// The noise for each [`PeriodicSegment`].
    pub tile: N,
}

impl<P: Noise, N: Noise> Noise for TilingNoise<P, N> {
    #[inline]
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        self.tile.set_seed(seed);
        self.tilier.set_seed(seed);
    }
}

impl<S, P: ScalableNoise<S>, N: Noise> NoiseBuilder<TilingNoise<P, N>, S> for DirectNoiseBuilder
where
    Self: NoiseBuilder<N, ()> + NoiseBuilder<P, S>,
{
    #[inline]
    fn build(&self, seed: &mut SeedGenerator, scale: S) -> TilingNoise<P, N> {
        TilingNoise {
            tilier: NoiseBuilder::<P, S>::build(self, seed, scale),
            tile: NoiseBuilder::<N, ()>::build(self, seed, ()),
        }
    }
}

impl<I, P: DirectNoise<I, Output: PeriodicSegment>, N: DirectNoise<P::Output>> DirectNoise<I>
    for TilingNoise<P, N>
{
    type Output = N::Output;

    #[inline]
    fn raw_sample(&self, input: I) -> Self::Output {
        self.tilier.raw_sample(input).and_then(&self.tile)
    }
}

impl<I, P: DirectNoise<I, Output: PeriodicSegment>, N: GradientNoise<P::Output>> GradientNoise<I>
    for TilingNoise<P, N>
{
    type Gradient = N::Gradient;

    #[inline]
    fn sample_gradient(&self, input: I) -> (Self::Gradient, Self::Output) {
        self.tile.sample_gradient(self.tilier.raw_sample(input))
    }
}

impl<T, P: ScalableNoise<T>, N: Noise> ScalableNoise<T> for TilingNoise<P, N> {
    #[inline]
    fn get_scale(&self) -> T {
        self.tilier.get_scale()
    }

    #[inline]
    fn set_scale(&mut self, period: T) {
        self.tilier.set_scale(period);
    }
}
