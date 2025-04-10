//! Contains systems for layering noise ontop of eachother.

use super::{Noise, NoiseValue, periodic::ScalableNoise, white::SeedGenerator};

/// Represents the result of a series of [`NoiseLayer`]s.
pub trait LayerAccumulator<T>: Send + Sync {
    /// The final result of all the layers.
    type Result: NoiseValue;

    /// Adds this `unit_value` at this `amplitude` to the result.
    fn accumulate(&mut self, unit_value: T, amplatude: f32);
    /// Finishes the layers, returning the final result.
    fn finish(self) -> Self::Result;
}

/// A generator that specifies amplitude for a seriese of [`NoiseLayer`]s.
pub trait LayerAmplitude: Send + Sync {
    /// Gets the amplitude for the next layer.
    fn get_next_amplitude(&mut self) -> f32;
    /// Sets the amplitude for the next layer.
    fn set_next_amplitude(&mut self, amplitude: f32);
    /// Multiplies the amplitude for all future layers by this amount.
    fn multiply_amplitude(&mut self, multiplier: f32);
}

/// A generator that specifies scale for a series of [`NoiseLayer`]s.
pub trait LayerScale<S>: Send + Sync {
    /// Gets the scale for the next layer, updating it for the next one.
    fn get_next_scale(&mut self) -> S;
    /// Gets the scale for the next layer.
    fn peek_next_scale(&self) -> S;
    /// Sets the scale for the next layer.
    fn set_next_scale(&mut self, scale: S);
    /// Multiplies the scale for all future layers by this amount.
    fn multiply_scale(&mut self, multiplier: S);
}

/// Represents a layer of some [`LayeredNoise`].
/// Each layer builds on the last, producing a composition of various noises.
pub trait NoiseLayer<I, S, O>: Send + Sync {
    /// Samples this layer of noise.
    /// This should use the `input` to [`LayerAccumulator::accumulate`] into the `output`.
    /// This may also progress `seed`, `scale`, and `amplitude`.
    fn layer_sample(
        &self,
        input: &mut I,
        seed: &mut SeedGenerator,
        scale: &mut impl LayerScale<S>,
        amplitude: &mut impl LayerAmplitude,
        output: &mut impl LayerAccumulator<O>,
    );
}

macro_rules! impl_layers {
    ($($t:ident=$f:tt),+) => {
        impl<I, S, O, $($t: NoiseLayer<I, S, O>,)+> NoiseLayer<I, S, O> for ($($t,)+) {
            fn layer_sample(
                &self,
                input: &mut I,
                seed: &mut SeedGenerator,
                scale: &mut impl LayerScale<S>,
                amplitude: &mut impl LayerAmplitude,
                output: &mut impl LayerAccumulator<O>,
            ) {
                $(self.$f.layer_sample(input, seed, scale, amplitude, output);)+
            }
        }
    };
}

#[rustfmt::skip]
mod impls {
use super::*;

impl_layers!(T0 = 0);
impl_layers!(T0 = 0, T1 = 1);
impl_layers!(T0 = 0, T1 = 1, T2 = 2);
impl_layers!(T0 = 0, T1 = 1, T2 = 2, T3 = 3);
impl_layers!(T0 = 0, T1 = 1, T2 = 2, T3 = 3, T4 = 4);
impl_layers!(T0 = 0, T1 = 1, T2 = 2, T3 = 3, T4 = 4, T5 = 5);
impl_layers!(T0 = 0, T1 = 1, T2 = 2, T3 = 3, T4 = 4, T5 = 5, T6 = 6);
impl_layers!(T0 = 0, T1 = 1, T2 = 2, T3 = 3, T4 = 4, T5 = 5, T6 = 6, T7 = 7);
impl_layers!(T0 = 0, T1 = 1, T2 = 2, T3 = 3, T4 = 4, T5 = 5, T6 = 6, T7 = 7, T8 = 8);
impl_layers!(T0 = 0, T1 = 1, T2 = 2, T3 = 3, T4 = 4, T5 = 5, T6 = 6, T7 = 7, T8 = 8, T9 = 9);
impl_layers!(T0 = 0, T1 = 1, T2 = 2, T3 = 3, T4 = 4, T5 = 5, T6 = 6, T7 = 7, T8 = 8, T9 = 9, T10 = 10);
impl_layers!(T0 = 0, T1 = 1, T2 = 2, T3 = 3, T4 = 4, T5 = 5, T6 = 6, T7 = 7, T8 = 8, T9 = 9, T10 = 10, T11 = 11);
impl_layers!(T0 = 0, T1 = 1, T2 = 2, T3 = 3, T4 = 4, T5 = 5, T6 = 6, T7 = 7, T8 = 8, T9 = 9, T10 = 10, T11 = 11, T12 = 12);
impl_layers!(T0 = 0, T1 = 1, T2 = 2, T3 = 3, T4 = 4, T5 = 5, T6 = 6, T7 = 7, T8 = 8, T9 = 9, T10 = 10, T11 = 11, T12 = 12, T13 = 13);
impl_layers!(T0 = 0, T1 = 1, T2 = 2, T3 = 3, T4 = 4, T5 = 5, T6 = 6, T7 = 7, T8 = 8, T9 = 9, T10 = 10, T11 = 11, T12 = 12, T13 = 13, T14 = 14);
impl_layers!(T0 = 0, T1 = 1, T2 = 2, T3 = 3, T4 = 4, T5 = 5, T6 = 6, T7 = 7, T8 = 8, T9 = 9, T10 = 10, T11 = 11, T12 = 12, T13 = 13, T14 = 14, T15 = 15);
}

/// A [`Noise`] that operates by composing the [`NoiseLayer`]s of `L` together into some result `R`, producing a fractal appearance.
/// `S` denotes the [`LayerScale`] and `A` denotes the [`LayerAmplitude`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FractalNoise<R, S, A, L> {
    /// The scale, which defines the scale of each octave and its progression.
    /// The values of this will determine the starting point for each sample.
    ///
    /// Sometimes this is called lacunarity because it affects how sparse or dense the fractal appears.
    pub scale: S,
    /// The amplitude, which defines the controbution of each octave and its progression.
    /// The values of this will determine the starting point for each sample.
    ///
    /// Sometimes this is called gain.
    pub amplitude: A,
    /// The result of the noise. This is not the literal result but rather the starting point for its accumulation.
    /// For example, modifying this can "pre-load" some octave's results before a sample starts.
    pub result: R,
    /// These are the layers/octaves of the [`FractalNoise`].
    /// They are composed together into the result.
    pub octaves: L,
    /// The seed for [`FractalNoise`].
    pub seed: SeedGenerator,
}

impl<R, S, A, L> Noise for FractalNoise<R, S, A, L>
where
    Self: Send + Sync,
{
    #[inline]
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        self.seed = seed.branch();
    }
}

impl<T, R, S: LayerScale<T>, A, L> ScalableNoise<T> for FractalNoise<R, S, A, L>
where
    Self: Noise,
{
    fn get_scale(&self) -> T {
        self.scale.peek_next_scale()
    }

    fn set_scale(&mut self, scale: T) {
        self.scale.set_next_scale(scale);
    }
}
