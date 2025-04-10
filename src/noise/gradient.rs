//! Contains logic for all kinds of gradient noise.

use core::ops::Add;

use bevy_math::{
    Curve, Vec2, Vec3, Vec3A, Vec4,
    curve::{Ease, derivatives::SampleDerivative},
};

use super::{
    DirectNoise, GradientNoise, Noise, NoiseValue,
    norm::UNorm,
    periodic::{
        DiferentiablePeriodicPoints, PeriodicPoint, PeriodicSegment, SamplablePeriodicPoints,
    },
    white::{SeedGenerator, White32},
};

/// This trait allows for use as `G` in [`SegmentalGradientNoise`].
///
/// # Implementer note
///
/// For `offset` values where each element is in ±1,
/// [`get_gradient_dot`](GradientGenerator::get_gradient_dot) must return a value x such that x *
/// [`NORMALIZING_FACTOR`](GradientGenerator::NORMALIZING_FACTOR) is within ±1, where d is the
/// number od dimensions in `I`.
pub trait GradientGenerator<I: NoiseValue>: Noise {
    /// Gets the dot product of `I` with some gradient vector based on this seed.
    /// Each element of `offset` can be assumed to be in -1..=1.
    fn get_gradient_dot(&self, seed: u32, offset: I) -> f32;

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

fn point_sample<P: PeriodicPoint<Relative: NoiseValue>, G: GradientGenerator<P::Relative>>(
    point: P,
    seed: u32,
    gradients: &G,
) -> f32 {
    let relative = point.into_relative(seed);
    gradients.get_gradient_dot(relative.seed, relative.offset)
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
            |point| point_sample(point, self.seed, &self.gradients),
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
    G: GradientGenerator<P::Relative>,
    C: SampleDerivative<f32> + Send + Sync,
> GradientNoise<T> for SegmentalGradientNoise<G, C>
{
    type Gradient = P::Relative;

    #[inline]
    fn sample_gradient(&self, input: T) -> (Self::Gradient, Self::Output) {
        let points = input.get_points();
        let gradient = points.sample_gradient_smooth(
            |point| point_sample(point, self.seed, &self.gradients),
            |start, end| *end - *start,
            Ease::interpolating_curve_unbounded,
            &self.smoothing_curve,
        );
        let value = points.sample_smooth(
            |point| point_sample(point, self.seed, &self.gradients),
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

/// A simple perlin noise source from uniquely random values.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeRand;

impl Noise for RuntimeRand {}

impl GradientGenerator<Vec2> for RuntimeRand {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec2) -> f32 {
        GradientGenerator::<Vec2>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec2 {
        Vec2::new(
            White32(seed).raw_sample(0).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(1).map_to::<UNorm>().get() * 2.0 - 1.0,
        )
    }
}

impl GradientGenerator<Vec3> for RuntimeRand {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3) -> f32 {
        GradientGenerator::<Vec3>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3 {
        Vec3::new(
            White32(seed).raw_sample(0).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(1).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(2).map_to::<UNorm>().get() * 2.0 - 1.0,
        )
    }
}

impl GradientGenerator<Vec3A> for RuntimeRand {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3A) -> f32 {
        GradientGenerator::<Vec3A>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3A {
        Vec3A::new(
            White32(seed).raw_sample(0).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(1).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(2).map_to::<UNorm>().get() * 2.0 - 1.0,
        )
    }
}

impl GradientGenerator<Vec4> for RuntimeRand {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec4) -> f32 {
        GradientGenerator::<Vec4>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec4 {
        Vec4::new(
            White32(seed).raw_sample(0).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(1).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(2).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(3).map_to::<UNorm>().get() * 2.0 - 1.0,
        )
    }
}

/// Allows making a [`GradientGenerator`] by specifying how it's parts are made.
pub trait GradElementTable {
    /// Gets an element of a gradient in ±1 from this seed.
    fn get_element(&self, seed: u8) -> f32;
}

impl Noise for GradTableQuick {}

impl<T: GradElementTable + Noise> GradientGenerator<Vec2> for T {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec2) -> f32 {
        GradientGenerator::<Vec2>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec2 {
        Vec2::new(
            self.get_element((seed >> 24) as u8),
            self.get_element((seed >> 16) as u8),
        )
    }
}

impl<T: GradElementTable + Noise> GradientGenerator<Vec3> for T {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3) -> f32 {
        GradientGenerator::<Vec3>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3 {
        Vec3::new(
            self.get_element((seed >> 24) as u8),
            self.get_element((seed >> 16) as u8),
            self.get_element((seed >> 8) as u8),
        )
    }
}

impl<T: GradElementTable + Noise> GradientGenerator<Vec3A> for T {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3A) -> f32 {
        GradientGenerator::<Vec3A>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3A {
        Vec3A::new(
            self.get_element((seed >> 24) as u8),
            self.get_element((seed >> 16) as u8),
            self.get_element((seed >> 8) as u8),
        )
    }
}

impl<T: GradElementTable + Noise> GradientGenerator<Vec4> for T {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec4) -> f32 {
        GradientGenerator::<Vec4>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec4 {
        Vec4::new(
            self.get_element((seed >> 24) as u8),
            self.get_element((seed >> 16) as u8),
            self.get_element((seed >> 8) as u8),
            self.get_element(seed as u8),
        )
    }
}

/// A simple perlin noise source that uses vectors with elemental values of only -1, 0, or 1.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GradTableQuick;

impl GradElementTable for GradTableQuick {
    #[inline]
    fn get_element(&self, seed: u8) -> f32 {
        // as i8 as a nop, and as f32 is probably faster than a array lookup or jump table.
        (seed as i8) as f32 * (1.0 / 128.0)
    }
}
