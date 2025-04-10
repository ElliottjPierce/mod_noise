//! Contains logic for all kinds of gradient noise.

use core::ops::Add;

use bevy_math::{
    Curve,
    curve::{Ease, derivatives::SampleDerivative},
};

use super::{
    DirectNoise, GradientNoise, Noise, NoiseValue,
    norm::UNorm,
    periodic::{
        DiferentiablePeriodicPoints, PeriodicPoint, PeriodicSegment, SamplablePeriodicPoints,
    },
    white::SeedGenerator,
};

/// This trait allows for use as `G` in [`SegmentalGradientNoise`].
///
/// # Implementer note
///
/// For `offset` values where each element is in ±1,
/// [`get_perlin_dot`](GradientGenerator::get_perlin_dot) must return a value x such that x *
/// [`NORMALIZING_FACTOR`](GradientGenerator::NORMALIZING_FACTOR) / √d is within ±1, where d is the
/// number od dimensions in `I`.
pub trait GradientGenerator<I: NoiseValue>: Noise {
    /// See [`GradientGenerator`]'s safety comment for info.
    const NORMALIZING_FACTOR: f32;

    /// Gets the dot product of `I` with some gradient vector based on this seed.
    /// Each element of `offset` can be assumed to be in -1..=1.
    fn get_gradient_dot(&self, seed: u32, offset: I) -> f32;
}

/// Allows accessing the gradients used in [`GradientGenerator`].
pub trait DifferentiableGradientGenerator<I: NoiseValue>: GradientGenerator<I> {
    /// Gets the gradient that would be used in [`get_gradient_dot`](GradientGenerator::get_gradient_dot).
    fn get_gradient(&self, seed: u32) -> I;
}

/// Represents a noise on a [`PeriodicSegment`] where each of its [`PeriodicPoint`]s are assigned a gradient vector,
/// and the resulting noise is formed by interpolating the dot product of the gradient with the relative position of the sample point.
///
/// This can be used to make perlin noise, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SegmentalGradientNoise<G, C> {
    /// A [`GradientGenerator`] that produces the gradient vectors.
    pub gradients: G,
    /// A [`Curve<f32>`] that determines how to interpolate.
    /// For items in domain 0..=1, this must produce results in 0..=1.
    pub smoothing_curve: C,
    /// The seed for the noise.
    pub seed: u32,
}

impl<N: Noise, C: Curve<f32> + Send + Sync> Noise for SegmentalGradientNoise<N, C> {
    #[inline]
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        self.seed = seed.next_seed();
        self.gradients.set_seed(seed);
    }
}

impl<
    T: PeriodicSegment<Points: SamplablePeriodicPoints, Point = P>,
    P: PeriodicPoint<Relative: NoiseValue>,
    G: GradientGenerator<P::Relative>,
    C: Curve<f32> + Send + Sync,
> DirectNoise<T> for SegmentalGradientNoise<G, C>
{
    type Output = UNorm;

    #[inline]
    fn raw_sample(&self, input: T) -> Self::Output {
        let raw = input.get_points().sample_smooth(
            |point| {
                let relative = point.into_relative(self.seed);
                self.gradients
                    .get_gradient_dot(relative.seed, relative.offset)
            },
            Ease::interpolating_curve_unbounded,
            &self.smoothing_curve,
        );
        UNorm::new_unchecked((raw + 1.0) * 0.5)
    }
}

impl<
    T: PeriodicSegment<Points: DiferentiablePeriodicPoints, Point = P>,
    P: PeriodicPoint<
        Relative: NoiseValue
                      + Add<P::Relative, Output = P::Relative>
                      + From<<T::Points as DiferentiablePeriodicPoints>::Gradient<f32>>
                      + Ease,
    >,
    G: DifferentiableGradientGenerator<P::Relative>,
    C: SampleDerivative<f32> + Send + Sync,
> GradientNoise<T> for SegmentalGradientNoise<G, C>
{
    type Gradient = P::Relative;

    #[inline]
    fn sample_gradient(&self, input: T) -> (Self::Gradient, Self::Output) {
        let points = input.get_points();
        let gradient = points.sample_gradient_smooth(
            |point| {
                let relative = point.into_relative(self.seed);
                self.gradients
                    .get_gradient_dot(relative.seed, relative.offset)
            },
            |start, end| *end - *start,
            Ease::interpolating_curve_unbounded,
            &self.smoothing_curve,
        );
        let value = points.sample_smooth(
            |point| {
                let relative = point.into_relative(self.seed);
                self.gradients
                    .get_gradient_dot(relative.seed, relative.offset)
            },
            Ease::interpolating_curve_unbounded,
            &self.smoothing_curve,
        );
        let raw_gradients = points.sample_smooth(
            |point| {
                let relative = point.into_relative(self.seed);
                self.gradients.get_gradient(relative.seed)
            },
            Ease::interpolating_curve_unbounded,
            &self.smoothing_curve,
        );
        (
            P::Relative::from(gradient) + raw_gradients,
            UNorm::new_unchecked((value + 1.0) * 0.5),
        )
    }
}
