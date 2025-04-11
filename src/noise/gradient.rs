//! Contains logic for all kinds of gradient noise.

use core::ops::Add;

use bevy_math::{
    Curve, Vec2, Vec3, Vec3A, Vec4,
    curve::{Ease, derivatives::SampleDerivative},
};

use super::{
    DirectNoise, DirectNoiseBuilder, GradientNoise, Noise, NoiseBuilder, NoiseValue,
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

impl<G: Noise, C: Default> NoiseBuilder<SegmentalGradientNoise<G, C>, ()> for DirectNoiseBuilder
where
    Self: NoiseBuilder<G, ()>,
{
    #[inline]
    fn build(&self, seed: &mut SeedGenerator, _scale: ()) -> SegmentalGradientNoise<G, C> {
        SegmentalGradientNoise {
            gradients: NoiseBuilder::<G, ()>::build(self, seed, ()),
            seed: seed.next_seed(),
            smoothing_curve: C::default(),
        }
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

/// A simple [`GradientGenerator`] that uses white noise to generate each element of the gradient independently.
///
/// This does not correct for the bunching of directions caused by normalizing.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RandomElementGradients;

impl Noise for RandomElementGradients {}

impl NoiseBuilder<RandomElementGradients, ()> for DirectNoiseBuilder {
    #[inline]
    fn build(&self, _seed: &mut SeedGenerator, _scale: ()) -> RandomElementGradients {
        RandomElementGradients
    }
}

impl GradientGenerator<Vec2> for RandomElementGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec2) -> f32 {
        GradientGenerator::<Vec2>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec2 {
        Vec2::new(
            White32(seed).raw_sample(983475).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(2983754).map_to::<UNorm>().get() * 2.0 - 1.0,
        )
    }
}

impl GradientGenerator<Vec3> for RandomElementGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3) -> f32 {
        GradientGenerator::<Vec3>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3 {
        Vec3::new(
            White32(seed).raw_sample(983475).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(2983754).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(823732).map_to::<UNorm>().get() * 2.0 - 1.0,
        )
    }
}

impl GradientGenerator<Vec3A> for RandomElementGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3A) -> f32 {
        GradientGenerator::<Vec3A>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec3A {
        Vec3A::new(
            White32(seed).raw_sample(983475).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(2983754).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(823732).map_to::<UNorm>().get() * 2.0 - 1.0,
        )
    }
}

impl GradientGenerator<Vec4> for RandomElementGradients {
    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec4) -> f32 {
        GradientGenerator::<Vec4>::get_gradient(self, seed).dot(offset)
    }

    #[inline]
    fn get_gradient(&self, seed: u32) -> Vec4 {
        Vec4::new(
            White32(seed).raw_sample(983475).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(2983754).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(823732).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(208375).map_to::<UNorm>().get() * 2.0 - 1.0,
        )
    }
}

/// Allows making a [`GradientGenerator`] by specifying how it's parts are made.
pub trait GradElementGenerator {
    /// Gets an element of a gradient in ±1 from this seed.
    fn get_element(&self, seed: u8) -> f32;
}

impl<T: GradElementGenerator + Noise> GradientGenerator<Vec2> for T {
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

impl<T: GradElementGenerator + Noise> GradientGenerator<Vec3> for T {
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

impl<T: GradElementGenerator + Noise> GradientGenerator<Vec3A> for T {
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

impl<T: GradElementGenerator + Noise> GradientGenerator<Vec4> for T {
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

/// A simple [`GradientGenerator`] that maps seeds directly to gradient vectors.
/// This is the fastest provided [`GradientGenerator`].
///
/// This does not correct for the bunching of directions caused by normalizing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuickGradients;

impl Noise for QuickGradients {}

impl NoiseBuilder<QuickGradients, ()> for DirectNoiseBuilder {
    #[inline]
    fn build(&self, _seed: &mut SeedGenerator, _scale: ()) -> QuickGradients {
        QuickGradients
    }
}

impl GradElementGenerator for QuickGradients {
    #[inline]
    fn get_element(&self, seed: u8) -> f32 {
        // as i8 as a nop, and as f32 is probably faster than a array lookup or jump table.
        (seed as i8) as f32 * (1.0 / 128.0)
    }
}

/// A simple [`GradientGenerator`] that maps seeds directly to gradient vectors.
/// This is very similar to [`QuickGradients`].
///
/// This approximately corrects for the bunching of directions caused by normalizing.
/// To do so, it maps it's distribution of points onto a cubic curve that distributes more values near ±0.5.
/// That reduces the directional artifacts caused by higher densities of gradients in corners which are mapped to similar directions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ApproximateUniformGradients;

impl Noise for ApproximateUniformGradients {}

impl NoiseBuilder<ApproximateUniformGradients, ()> for DirectNoiseBuilder {
    #[inline]
    fn build(&self, _seed: &mut SeedGenerator, _scale: ()) -> ApproximateUniformGradients {
        ApproximateUniformGradients
    }
}

impl GradElementGenerator for ApproximateUniformGradients {
    #[inline]
    fn get_element(&self, seed: u8) -> f32 {
        // try to bunch more values around ±0.5 so that there is less directional bunching.
        let unorm = (seed >> 1) as f32 * (1.0 / 128.0);
        let snorm = unorm * 2.0 - 1.0;
        let corrected = snorm * snorm * snorm;
        let corrected_unorm = corrected * 0.5 + 0.5;

        // make it positive or negative
        let sign = ((seed & 1) as u32) << 31;
        f32::from_bits(corrected_unorm.to_bits() ^ sign)
    }
}
