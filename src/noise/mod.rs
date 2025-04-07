//! Contains various noise functions

pub mod norm;
pub mod white;

/// Marks the type as the value inolved in noise.
pub trait NoiseValue: Sized + Clone + 'static {}

/// Signifies that this type can be created from `T`.
/// This is different from [`From`] because the numerical value it represents may change.
pub trait CorolatedNoiseType<T>: NoiseValue {
    /// Constructs a valid but corolated value based on this `value`.
    fn map_from(value: T) -> Self;
}

/// Represents a noise function that samples at a point of type `I` and returns a result of type
/// `O`.
pub trait Noise<I, O> {
    /// Samples the noise function at this `input`.
    fn raw_sample(&self, input: I) -> O;

    /// Samples the noise, converting it to `T`.
    #[inline]
    fn sample<T: CorolatedNoiseType<O>>(&self, input: I) -> T {
        T::map_from(self.raw_sample(input))
    }
}

/// Represents a differentiable [`Noise`].
pub trait GradientNoise<I, O>: Noise<I, O> {
    /// Samples the noise at `input`, returning the gradient and the output.
    fn sample_gradient(&self, input: I) -> (I, O);
}

/// Represents a [`Noise`] that can change its input instead of producing an output.
pub trait WarpingNoise<I>: Noise<I, I> {
    /// Warps or moves around the input value.
    fn warp_domain(&self, input: &mut I);
}

impl<I: Copy + core::ops::AddAssign, T: Noise<I, I>> WarpingNoise<I> for T {
    #[inline]
    fn warp_domain(&self, input: &mut I) {
        *input += self.raw_sample(*input);
    }
}

impl<I, O, T: GradientNoise<I, O>> Noise<I, O> for T {
    #[inline]
    fn raw_sample(&self, input: I) -> O {
        self.sample_gradient(input).1
    }
}

macro_rules! impl_noise_value {
    ($($name:ty),*) => {
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
    bevy_math::Vec2,
    bevy_math::Vec3,
    bevy_math::Vec3A,
    bevy_math::Vec4
);

#[cfg(test)]
mod tests {
    use super::*;

    struct NopNoise;

    impl GradientNoise<f32, f32> for NopNoise {
        #[inline]
        fn sample_gradient(&self, input: f32) -> (f32, f32) {
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
