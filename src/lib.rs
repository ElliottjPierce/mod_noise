#![no_std]
#![warn(clippy::wildcard_imports)]
#![warn(clippy::needless_pass_by_ref_mut)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

use core::ops::AddAssign;

/// Represents a noise function that samples at a point of type `I` and returns a result of type
/// `O`.
pub trait Noise<I, O> {
    /// Samples the noise function at this `input`.
    fn sample(&self, input: I) -> O;
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

impl<I: Copy + AddAssign, T: Noise<I, I>> WarpingNoise<I> for T {
    #[inline]
    fn warp_domain(&self, input: &mut I) {
        *input += self.sample(*input);
    }
}

impl<I, O, T: GradientNoise<I, O>> Noise<I, O> for T {
    #[inline]
    fn sample(&self, input: I) -> O {
        self.sample_gradient(input).1
    }
}

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
        assert_eq!(5.0, NopNoise.sample(5.0));
        assert_eq!((0.0, 3.0), NopNoise.sample_gradient(3.0));
        let mut loc = 1.0;
        NopNoise.warp_domain(&mut loc);
        assert_eq!(2.0, loc);
    }
}
