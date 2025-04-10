//! Contains noise for interpolating within a periodic segment.

use core::ops::Sub;

use bevy_math::{
    Curve, HasTangent,
    curve::{Ease, derivatives::SampleDerivative},
};

use super::{
    DirectNoise, GradientNoise, Noise,
    periodic::{
        DiferentiablePeriodicPoints, PeriodicPoint, PeriodicSegment, SamplablePeriodicPoints,
    },
    white::SeedGenerator,
};

/// Represents some noise on the [`PeriodicPoint`] of a [`PeriodicSegment`] where values between those points are smoothed out accordingly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SegmentalValueNoise<N, C> {
    /// The noise for each of the [`PeriodicPoint`]s.
    pub noise: N,
    /// The curve to interpolate by.
    /// This must be valid for domain 0..=1.
    pub smoothing_curve: C,
    /// The seed for the noise.
    pub seed: u32,
}

impl<N: Noise, C: Curve<f32> + Send + Sync> Noise for SegmentalValueNoise<N, C> {
    #[inline]
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        self.seed = seed.next_seed();
        self.noise.set_seed(seed);
    }
}

impl<
    T: PeriodicSegment<Points: SamplablePeriodicPoints>,
    N: DirectNoise<u32, Output: Ease>,
    C: Curve<f32> + Send + Sync,
> DirectNoise<T> for SegmentalValueNoise<N, C>
{
    type Output = N::Output;

    #[inline]
    fn raw_sample(&self, input: T) -> Self::Output {
        input.get_points().sample_smooth(
            |point| {
                let point = point.into_relative(self.seed).seed;
                self.noise.raw_sample(point)
            },
            Ease::interpolating_curve_unbounded,
            &self.smoothing_curve,
        )
    }
}

impl<
    T: PeriodicSegment<Points: DiferentiablePeriodicPoints>,
    O: Ease + Copy + HasTangent<Tangent = O> + Sub<O, Output = O>,
    N: DirectNoise<u32, Output = O>,
    C: SampleDerivative<f32> + Send + Sync,
> GradientNoise<T> for SegmentalValueNoise<N, C>
{
    type Gradient = <T::Points as DiferentiablePeriodicPoints>::Gradient<O>;

    #[inline]
    fn sample_gradient(&self, input: T) -> (Self::Gradient, Self::Output) {
        let points = input.get_points();
        let gradient = points.sample_gradient_smooth(
            |point| {
                let point = point.into_relative(self.seed).seed;
                self.noise.raw_sample(point)
            },
            |start, end| *end - *start,
            Ease::interpolating_curve_unbounded,
            &self.smoothing_curve,
        );
        let value = points.sample_smooth(
            |point| {
                let point = point.into_relative(self.seed).seed;
                self.noise.raw_sample(point)
            },
            Ease::interpolating_curve_unbounded,
            &self.smoothing_curve,
        );
        (gradient, value)
    }
}
