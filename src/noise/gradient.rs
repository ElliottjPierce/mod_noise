//! Contains logic for all kinds of gradient noise.

use core::{hint::unreachable_unchecked, ops::Add};

use bevy_math::{
    Curve, Vec2, Vec3, Vec4,
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
    G: DifferentiableGradientGenerator<P::Relative>,
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

const SQRT_2: f32 = core::f32::consts::SQRT_2;
const SQRT_3: f32 = 1.7320508;
const SQRT_4: f32 = 2.0;

/// A simple perlin noise source from uniquely random values.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeRand;

impl Noise for RuntimeRand {}

impl GradientGenerator<Vec2> for RuntimeRand {
    // The dot product can not be grater than the product of the
    // lengths, and one length is normalized and the other one is taken care of by setting
    // `NORMALIZING_FACTOR` to 2.0.
    const NORMALIZING_FACTOR: f32 = 2.0 / SQRT_2;

    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec2) -> f32 {
        let vec = Vec2::new(
            White32(seed).raw_sample(0).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(1).map_to::<UNorm>().get() * 2.0 - 1.0,
        ) * 128.0; // extra multiplication prevenst len from being Nan because of an approx zero length.
        vec.normalize().dot(offset)
    }
}

impl GradientGenerator<Vec3> for RuntimeRand {
    // See impl PerlinSource<Vec2> for RuntimeRand
    const NORMALIZING_FACTOR: f32 = 2.0 / SQRT_3;

    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3) -> f32 {
        let vec = Vec3::new(
            White32(seed).raw_sample(0).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(1).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(2).map_to::<UNorm>().get() * 2.0 - 1.0,
        ) * 128.0; // extra multiplication prevenst len from being Nan because of an approx zero length.
        vec.normalize().dot(offset)
    }
}

impl GradientGenerator<Vec4> for RuntimeRand {
    // See impl PerlinSource<Vec2> for RuntimeRand
    const NORMALIZING_FACTOR: f32 = 2.0 / SQRT_4;

    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec4) -> f32 {
        let vec = Vec4::new(
            White32(seed).raw_sample(0).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(1).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(2).map_to::<UNorm>().get() * 2.0 - 1.0,
            White32(seed).raw_sample(3).map_to::<UNorm>().get() * 2.0 - 1.0,
        ) * 128.0; // extra multiplication prevenst len from being Nan because of an approx zero length.
        vec.normalize().dot(offset)
    }
}

/// A simple perlin noise source that uses vectors with elemental values of only -1, 0, or 1.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Hashed;

impl Noise for Hashed {}

impl GradientGenerator<Vec2> for Hashed {
    // The dot product can not be grater than the product of the
    // lengths, and one length is within √d. So their product is normalized by setting
    // `NORMALIZING_FACTOR` to 1.0.
    const NORMALIZING_FACTOR: f32 = 1.0 / SQRT_2;

    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec2) -> f32 {
        let v = offset;
        match seed >> 29 {
            0 => v.x + v.y,
            1 => v.x - v.y,
            2 => -v.x + v.y,
            3 => -v.x - v.y,
            4 => v.x,
            5 => -v.x,
            6 => v.y,
            7 => -v.y,
            // SAFETY: We did >> 29 above, so there is no way for the value to be > 7.
            _ => unsafe { unreachable_unchecked() },
        }
    }
}

impl GradientGenerator<Vec3> for Hashed {
    // See impl PerlinSource<Vec2> for Cardinal.
    const NORMALIZING_FACTOR: f32 = 1.0 / SQRT_3;

    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec3) -> f32 {
        let mut result = 0.0;
        if seed & 1 > 0 {
            result += offset.x;
        }
        if seed & 2 > 0 {
            result -= offset.x;
        }
        if seed & 4 > 0 {
            result += offset.y;
        }
        if seed & 8 > 0 {
            result -= offset.y;
        }
        if seed & 16 > 0 {
            result += offset.z;
        }
        if seed & 32 > 0 {
            result -= offset.z;
        }
        result
    }
}

impl GradientGenerator<Vec4> for Hashed {
    // See impl PerlinSource<Vec2> for Cardinal.
    const NORMALIZING_FACTOR: f32 = 1.0 / SQRT_4;

    #[inline]
    fn get_gradient_dot(&self, seed: u32, offset: Vec4) -> f32 {
        let mut result = 0.0;
        if seed & 1 > 0 {
            result += offset.x;
        }
        if seed & 2 > 0 {
            result -= offset.x;
        }
        if seed & 4 > 0 {
            result += offset.y;
        }
        if seed & 8 > 0 {
            result -= offset.y;
        }
        if seed & 16 > 0 {
            result += offset.z;
        }
        if seed & 32 > 0 {
            result -= offset.z;
        }
        if seed & 64 > 0 {
            result += offset.w;
        }
        if seed & 128 > 0 {
            result -= offset.w;
        }
        result
    }
}
