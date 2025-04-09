//! Contains noise for interpolating within a periodic segment.

use bevy_math::{Curve, curve::Ease};

use super::{
    DirectNoise, Noise,
    periodic::{PeriodicPoint, PeriodicSegment, SamplablePeriodicPoints},
    white::SeedGenerator,
};

/// Represents some noise on the [`PeriodicPoint`] of a [`PeriodicSegment`] where values between those points are smoothed out accordingly.
pub struct SmoothSegmentNoise<N, C> {
    /// The noise for each of the [`PeriodicPoint`]s.
    pub noise: N,
    /// The curve to interpolate by.
    /// This must be valid for domain 0..=1.
    pub smoothing_curve: C,
    /// The seed for the noise.
    pub seed: u32,
}

impl<N: Noise, C: Curve<f32> + Send + Sync> Noise for SmoothSegmentNoise<N, C> {
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
> DirectNoise<T> for SmoothSegmentNoise<N, C>
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
