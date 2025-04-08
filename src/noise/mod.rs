//! Contains various noise functions

use white::SeedGenerator;

pub mod cellular;
pub mod common_mapping;
pub mod grid;
pub mod norm;
pub mod periodic;
pub mod white;

/// Marks the type as the value inolved in noise.
pub trait NoiseValue: Sized + Clone + 'static {
    /// Maps this [`NoiseValue`] to `T`.
    #[inline]
    fn map_to<T: CorolatedNoiseType<Self>>(self) -> T {
        T::map_from(self)
    }

    /// Passes this noise value through another noise.
    #[inline]
    fn and_then<T: DirectNoise<Self>>(self, noise: T) -> T::Output {
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
pub trait Noise {
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
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        _ = seed;
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
}

/// Represents a noise function that samples at a point of type `I` and returns a result.
pub trait DirectNoise<I>: Noise {
    /// The result of the noise.
    type Output: NoiseValue;

    /// Samples the noise function at this `input`.
    fn raw_sample(&self, input: I) -> Self::Output;
}

/// Represents a differentiable [`Noise`].
pub trait GradientNoise<I>: DirectNoise<I> {
    /// Samples the noise at `input`, returning the gradient and the output.
    fn sample_gradient(&self, input: I) -> (I, Self::Output);
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
