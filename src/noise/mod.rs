//! Contains various noise functions

use core::marker::PhantomData;

use layering::Octave;
use periodic::ScalableNoise;
use white::SeedGenerator;

pub mod adapters;
pub mod cellular;
pub mod common_mapping;
pub mod curves;
pub mod gradient;
pub mod grid;
pub mod layering;
pub mod norm;
pub mod periodic;
pub mod value;
pub mod white;

/// Marks the type as the value inolved in noise.
pub trait NoiseValue: Send + Sync + Sized + Clone + 'static {
    /// Maps this [`NoiseValue`] to `T`.
    #[inline]
    fn map_to<T: CorolatedNoiseType<Self>>(self) -> T {
        T::map_from(self)
    }

    /// Passes this noise value through another noise.
    #[inline]
    fn and_then<T: DirectNoise<Self>>(self, noise: &T) -> T::Output {
        noise.raw_sample(self)
    }
}

/// Signifies that this type can be created from `T`.
/// This is different from [`From`] because the numerical value it represents may change.
pub trait CorolatedNoiseType<T>: NoiseValue {
    /// Constructs a valid but corolated value based on this `value`.
    fn map_from(value: T) -> Self;
}

/// Represents some noise function.
pub trait Noise: Send + Sync {
    /// Sets the seed of the noise if applicable.
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        _ = seed;
    }
}

/// Additional items for [`Noise`] that are separate to keep [`Noise`] object safe.
pub trait NoiseExt: Noise {
    /// Samples the noise.
    ///
    /// This is separate from [`raw_sample`](Noise::raw_sample) for future proofing.
    #[inline]
    fn sample<I>(&self, input: I) -> Self::Output
    where
        Self: DirectNoise<I>,
    {
        self.raw_sample(input)
    }

    /// Sets the seed of the noise if applicable.
    #[inline]
    fn with_seed(mut self, seed: &mut SeedGenerator) -> Self
    where
        Self: Sized,
    {
        self.set_seed(seed);
        self
    }

    /// Sets the [`Period`](ScalableNoise::Period) of the noise.
    #[inline]
    fn with_period<T>(mut self, period: T) -> Self
    where
        Self: Sized + ScalableNoise<T>,
    {
        self.set_scale(period);
        self
    }
}

/// Represents a noise on type `I` scalable by type `P`.
pub trait PeriodicNoise<I, P>: DirectNoise<I> + ScalableNoise<P> {}

/// Represents a noise function that samples at a point of type `I` and returns a result.
pub trait DirectNoise<I>: Noise {
    /// The result of the noise.
    type Output: NoiseValue;

    /// Samples the noise function at this `input`.
    fn raw_sample(&self, input: I) -> Self::Output;
}

/// Represents a differentiable [`Noise`].
pub trait GradientNoise<I>: DirectNoise<I> {
    /// The kind of gradient this noise has.
    type Gradient;

    /// Samples the noise at `input`, returning the gradient and the output.
    fn sample_gradient(&self, input: I) -> (Self::Gradient, Self::Output);
}

/// Represents a [`Noise`] that can change its input instead of producing an output.
pub trait WarpingNoise<I>: DirectNoise<I, Output = I> {
    /// Warps or moves around the input value.
    fn warp_domain(&self, input: &mut I);
}

impl<T: NoiseValue> CorolatedNoiseType<T> for T {
    #[inline]
    fn map_from(value: T) -> Self {
        value
    }
}

impl<I: Copy + core::ops::AddAssign, T: DirectNoise<I, Output = I>> WarpingNoise<I> for T {
    #[inline]
    fn warp_domain(&self, input: &mut I) {
        *input += self.raw_sample(*input);
    }
}

impl<I, P, T: DirectNoise<I> + ScalableNoise<P>> PeriodicNoise<I, P> for T {}

impl<T: Noise> NoiseExt for T {}

macro_rules! impl_noise_value {
    ($($name:ty),*,) => {
        $( impl NoiseValue for $name {} )*
    };
}

impl_noise_value!(
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    u128,
    i128,
    f32,
    usize,
    isize,
    bevy_math::Vec2,
    bevy_math::Vec3,
    bevy_math::Vec3A,
    bevy_math::Vec4,
    bevy_math::U8Vec2,
    bevy_math::U8Vec3,
    bevy_math::U8Vec4,
    bevy_math::I8Vec2,
    bevy_math::I8Vec3,
    bevy_math::I8Vec4,
    bevy_math::U16Vec2,
    bevy_math::U16Vec3,
    bevy_math::U16Vec4,
    bevy_math::I16Vec2,
    bevy_math::I16Vec3,
    bevy_math::I16Vec4,
    bevy_math::U64Vec2,
    bevy_math::U64Vec3,
    bevy_math::U64Vec4,
    bevy_math::I64Vec2,
    bevy_math::I64Vec3,
    bevy_math::I64Vec4,
    bevy_math::UVec2,
    bevy_math::UVec3,
    bevy_math::UVec4,
    bevy_math::IVec2,
    bevy_math::IVec3,
    bevy_math::IVec4,
);

/// Manages building a [`Noise`] `N` for an octave. If the noise type is not [`ScalableNoise`], `()` can be used for `S`.
pub trait NoiseBuilder<N, S>: NoiseBuilderBase {
    /// Constructs a [`Noise`] `N` with a seed.
    fn build(&self, seed: &mut SeedGenerator, scale: S) -> N;
}

/// Represents the root of all [`OctaveNoiseBuilder`].
pub trait NoiseBuilderBase: Send + Sync + Sized {
    /// Creates an [`Octave`] for [`Noise`] `N` and scale `S`.
    #[inline]
    fn build_octave_for<S, N>(self) -> Octave<N, S, Self>
    where
        Self: NoiseBuilder<N, S>,
    {
        Octave {
            builder: self,
            marker: PhantomData,
        }
    }
}

/// A [`OctaveNoiseBuilder`] for any noise that implements [`Default`] and [`ScalableNoise`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DefaultAndSet;

impl NoiseBuilderBase for DefaultAndSet {}

impl<S, N: Default + ScalableNoise<S>> NoiseBuilder<N, S> for DefaultAndSet {
    #[inline]
    fn build(&self, seed: &mut SeedGenerator, scale: S) -> N {
        let mut noise = N::default();
        noise.set_seed(seed);
        noise.set_scale(scale);
        noise
    }
}

/// A [`OctaveNoiseBuilder`] for any noise that implements [`Clone`] and [`ScalableNoise`].
///
/// Whatever value is here will be cloned on each sample and will then have its values set.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CloneAndSet<N>(pub N);

impl<N: Clone + Noise> NoiseBuilderBase for CloneAndSet<N> {}

impl<S, N: Clone + ScalableNoise<S>> NoiseBuilder<N, S> for CloneAndSet<N> {
    #[inline]
    fn build(&self, seed: &mut SeedGenerator, scale: S) -> N {
        let mut noise = self.0.clone();
        noise.set_seed(seed);
        noise.set_scale(scale);
        noise
    }
}

/// A [`OctaveNoiseBuilderBase`] that users can use to implement high performance [`OctaveNoiseBuilder`] for their noise.
/// Most noise types should implement [`OctaveNoiseBuilder`] for this type. If the noise type is not [`ScalableNoise`], `()` can be used for `S`.
/// This is a good default.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DirectNoiseBuilder;

impl NoiseBuilderBase for DirectNoiseBuilder {}

#[cfg(test)]
mod tests {
    use super::*;

    struct NopNoise;

    impl Noise for NopNoise {}

    impl DirectNoise<f32> for NopNoise {
        type Output = f32;

        #[inline]
        fn raw_sample(&self, input: f32) -> Self::Output {
            input
        }
    }

    impl GradientNoise<f32> for NopNoise {
        type Gradient = f32;

        #[inline]
        fn sample_gradient(&self, input: f32) -> (f32, Self::Output) {
            (0.0, input)
        }
    }

    #[test]
    fn test_traits() {
        assert_eq!(5.0, NopNoise.raw_sample(5.0));
        assert_eq!((0.0, 3.0), NopNoise.sample_gradient(3.0));
        let mut loc = 1.0;
        NopNoise.warp_domain(&mut loc);
        assert_eq!(2.0, loc);
    }
}
