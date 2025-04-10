//! Contains systems for layering noise ontop of eachother.

use core::{
    marker::PhantomData,
    ops::{AddAssign, Mul},
};

use super::{
    CorolatedNoiseType, DirectNoise, Noise, NoiseValue,
    periodic::{Frequency, Period, ScalableNoise, WholePeriod},
    white::SeedGenerator,
};

/// Represents the result of a series of [`NoiseLayer`]s.
pub trait LayerResult: Send + Sync {
    /// The final result of all the layers.
    type Result: NoiseValue;

    /// Finishes the layers, returning the final result.
    fn finish(self) -> Self::Result;
}

/// Represents the ability of a [`LayerResult`] to accumulate values of `T`.
pub trait LayerAccumulator<T>: LayerResult {
    /// Adds this `unit_value` at this `amplitude` to the result.
    fn accumulate(&mut self, unit_value: T, amplitude: f32);
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
pub trait NoiseLayer<I, S, R>: Send + Sync {
    /// Samples this layer of noise.
    /// This should use the `input` to [`LayerAccumulator::accumulate`] into the `output`.
    /// This may also progress `seed`, `scale`, and `amplitude`.
    fn layer_sample(
        &self,
        input: &mut I,
        seed: &mut SeedGenerator,
        scale: &mut S,
        amplitude: &mut impl LayerAmplitude,
        output: &mut R,
    );
}

macro_rules! impl_layers {
    ($($t:ident=$f:tt),+) => {
        impl<I, S, R, $($t: NoiseLayer<I, S, R>,)+> NoiseLayer<I, S, R> for ($($t,)+) {
            #[inline]
            fn layer_sample(
                &self,
                input: &mut I,
                seed: &mut SeedGenerator,
                scale: &mut S,
                amplitude: &mut impl LayerAmplitude,
                output: &mut R,
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

/// A [`Noise`] that operates by composing the [`NoiseLayer`]s of `L` together into some [`LayerResult`] `R`, producing a fractal appearance.
/// `S` denotes the [`LayerScale`] and `A` denotes the [`LayerAmplitude`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FractalNoise<F, R, S, A, L> {
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
    /// Describes the final operation on the result from `R`.
    pub finalizer: F,
    /// The seed for [`FractalNoise`].
    pub seed: SeedGenerator,
    /// These are the layers/octaves of the [`FractalNoise`].
    /// They are composed together into the result.
    pub octaves: L,
}

impl<F: Noise, R, S, A, L> Noise for FractalNoise<F, R, S, A, L>
where
    Self: Send + Sync,
{
    #[inline]
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        self.seed = seed.branch();
        self.finalizer.set_seed(seed);
    }
}

impl<T, F, R, S: LayerScale<T>, A, L> ScalableNoise<T> for FractalNoise<F, R, S, A, L>
where
    Self: Noise,
{
    #[inline]
    fn get_scale(&self) -> T {
        self.scale.peek_next_scale()
    }

    #[inline]
    fn set_scale(&mut self, scale: T) {
        self.scale.set_next_scale(scale);
    }
}

impl<
    I,
    F: DirectNoise<R::Result>,
    R: LayerResult + Clone,
    S: Clone,
    A: LayerAmplitude + Clone,
    L: NoiseLayer<I, S, R>,
> DirectNoise<I> for FractalNoise<F, R, S, A, L>
where
    Self: Noise,
{
    type Output = F::Output;

    #[inline]
    fn raw_sample(&self, mut input: I) -> Self::Output {
        let mut result = self.result.clone();
        self.octaves.layer_sample(
            &mut input,
            &mut self.seed.clone(),
            &mut self.scale.clone(),
            &mut self.amplitude.clone(),
            &mut result,
        );
        self.finalizer.raw_sample(result.finish())
    }
}

/// A standard [`LayerResult`] that just normalizes the result to be a direct result of the accumulated values and amplitudes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NormalizeOctavesInto<T> {
    /// The sum of all accumulated values.
    pub base: T,
    /// The total of all accumulated amplitudes.
    pub total_amplitudes: f32,
}

impl<T: NoiseValue + Mul<f32, Output = T>> LayerResult for NormalizeOctavesInto<T> {
    type Result = T;

    #[inline]
    fn finish(self) -> Self::Result {
        self.base * (1.0 / self.total_amplitudes)
    }
}

impl<I, T: NoiseValue + CorolatedNoiseType<I> + AddAssign<T> + Mul<f32, Output = T>>
    LayerAccumulator<I> for NormalizeOctavesInto<T>
{
    #[inline]
    fn accumulate(&mut self, unit_value: I, amplitude: f32) {
        let inner = T::map_from(unit_value);
        self.base += inner * amplitude;
        self.total_amplitudes += amplitude;
    }
}

/// A standard [`LayerScale`] that makes each octave a fraction of the scale of the first.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FractalScaling {
    /// The lowest frequency, the frequency of the first octave.
    pub overall: Frequency,
    /// Each octave's frequency is multiplied by this value.
    pub gain: f32,
}

impl Default for FractalScaling {
    #[inline]
    fn default() -> Self {
        Self {
            overall: Frequency::default(),
            gain: 2.0,
        }
    }
}

impl LayerScale<Frequency> for FractalScaling {
    #[inline]
    fn get_next_scale(&mut self) -> Frequency {
        let res = self.overall;
        self.overall.0 *= self.gain;
        res
    }

    #[inline]
    fn peek_next_scale(&self) -> Frequency {
        self.overall
    }

    #[inline]
    fn set_next_scale(&mut self, scale: Frequency) {
        self.overall = scale;
    }

    #[inline]
    fn multiply_scale(&mut self, multiplier: Frequency) {
        self.overall.0 *= multiplier.0;
    }
}

impl LayerScale<Period> for FractalScaling {
    #[inline]
    fn get_next_scale(&mut self) -> Period {
        LayerScale::<Frequency>::get_next_scale(self).into()
    }

    #[inline]
    fn peek_next_scale(&self) -> Period {
        LayerScale::<Frequency>::peek_next_scale(self).into()
    }

    #[inline]
    fn set_next_scale(&mut self, scale: Period) {
        LayerScale::<Frequency>::set_next_scale(self, scale.into());
    }

    #[inline]
    fn multiply_scale(&mut self, multiplier: Period) {
        LayerScale::<Frequency>::multiply_scale(self, multiplier.into());
    }
}

impl LayerScale<WholePeriod> for FractalScaling {
    #[inline]
    fn get_next_scale(&mut self) -> WholePeriod {
        LayerScale::<Frequency>::get_next_scale(self).into()
    }

    #[inline]
    fn peek_next_scale(&self) -> WholePeriod {
        LayerScale::<Frequency>::peek_next_scale(self).into()
    }

    #[inline]
    fn set_next_scale(&mut self, scale: WholePeriod) {
        LayerScale::<Frequency>::set_next_scale(self, scale.into());
    }

    #[inline]
    fn multiply_scale(&mut self, multiplier: WholePeriod) {
        LayerScale::<Frequency>::multiply_scale(self, multiplier.into());
    }
}

/// A standard [`LayerAmplitude`] that makes each octave have an amplitude proportional to the last.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProportionalAmplitude {
    /// The amplitude of the first octave
    pub base: f32,
    /// The multiple for octaves.
    pub proportion: f32,
}

impl Default for ProportionalAmplitude {
    #[inline]
    fn default() -> Self {
        Self {
            base: 1000.0,
            proportion: 0.5,
        }
    }
}

impl LayerAmplitude for ProportionalAmplitude {
    #[inline]
    fn get_next_amplitude(&mut self) -> f32 {
        let res = self.base;
        self.base *= self.proportion;
        res
    }

    #[inline]
    fn set_next_amplitude(&mut self, amplitude: f32) {
        self.base = amplitude;
    }

    #[inline]
    fn multiply_amplitude(&mut self, multiplier: f32) {
        self.base *= multiplier;
    }
}

/// Represents a direct octave of `N` noise via scale `S` and source `F`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Octave<N, S, F: Fn(&mut SeedGenerator) -> N> {
    /// The means by which `N` is created for each octave.
    pub generator: F,
    /// Marker data. `N` specifies the expected [`Noise`] type, and `S` specifies the scale to use on it.
    pub marker: PhantomData<(N, S)>,
}

impl<N, S, F: Fn(&mut SeedGenerator) -> N> Octave<N, S, F> {
    /// Creates a new [`Octave`] with the given generator.
    #[inline]
    pub fn new(generator: F) -> Self {
        Self {
            generator,
            marker: PhantomData,
        }
    }
}

impl<
    I: Copy,
    NS,
    N: DirectNoise<I> + ScalableNoise<NS>,
    R: LayerAccumulator<N::Output>,
    S: LayerScale<NS>,
    F: Fn(&mut SeedGenerator) -> N,
> NoiseLayer<I, S, R> for Octave<N, NS, F>
where
    Self: Send + Sync,
{
    #[inline]
    fn layer_sample(
        &self,
        input: &mut I,
        seed: &mut SeedGenerator,
        scale: &mut S,
        amplitude: &mut impl LayerAmplitude,
        output: &mut R,
    ) {
        let mut noise = (self.generator)(seed);
        noise.set_scale(scale.get_next_scale());
        let octave_result = noise.raw_sample(*input);
        let amplitude = amplitude.get_next_amplitude();
        output.accumulate(octave_result, amplitude);
    }
}

/// Repeats a [`NoiseLayer`] `L` some number of times.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Repeat<L> {
    /// The [`NoiseLayer`] to repeat.
    pub layer: L,
    /// The number of times to repeat it.
    pub repetitions: u32,
}

impl<L> Repeat<L> {
    /// Constructs a new [`Repeat`].
    #[inline]
    pub fn new(times: u32, layer: L) -> Self {
        Self {
            layer,
            repetitions: times,
        }
    }
}

impl<I, S, R, L: NoiseLayer<I, S, R>> NoiseLayer<I, S, R> for Repeat<L>
where
    Self: Send + Sync,
{
    #[inline]
    fn layer_sample(
        &self,
        input: &mut I,
        seed: &mut SeedGenerator,
        scale: &mut S,
        amplitude: &mut impl LayerAmplitude,
        output: &mut R,
    ) {
        for _ in 0..self.repetitions {
            self.layer
                .layer_sample(input, seed, scale, amplitude, output);
        }
    }
}
