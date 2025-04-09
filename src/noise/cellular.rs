//! Contains logic for cellular noise, including ones not based on voronoi graphs.

use super::{
    DirectNoise, Noise, NoiseValue,
    periodic::{PeriodicPoint, PeriodicSegment, ScalableNoise},
    white::SeedGenerator,
};

/// Represents cell noise over some [`DirectNoise<u32>`](DirectNoise) `N`.
/// When given a [`PeriodicSegment`], this will pass the segment's main point's seed into the noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CellNoise<N> {
    /// The inner noise used.
    pub noise: N,
    /// The seed of the cellular noise
    pub seed: u32,
}

impl<N: Noise> Noise for CellNoise<N> {
    #[inline]
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        self.seed = seed.next_seed();
        self.noise.set_seed(seed);
    }
}

impl<T: PeriodicSegment, N: DirectNoise<u32>> DirectNoise<T> for CellNoise<N> {
    type Output = N::Output;

    #[inline]
    fn raw_sample(&self, input: T) -> Self::Output {
        let input = input.get_main_point().into_relative(self.seed).seed;
        self.noise.raw_sample(input)
    }
}

/// Represents cellular noise over some for some [`ScalableNoise`] `P` and some [`DirectNoise`] `N` that acts on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CellularNoise<P, N> {
    /// The noise for making the cells.
    pub periodic: P,
    /// The noise for each cell.
    pub cell_noise: CellNoise<N>,
}

impl<P: Noise, N> Noise for CellularNoise<P, N>
where
    CellNoise<N>: Noise,
{
    #[inline]
    fn set_seed(&mut self, seed: &mut SeedGenerator) {
        self.cell_noise.set_seed(seed);
        self.periodic.set_seed(seed);
    }
}

impl<I, P: DirectNoise<I, Output: PeriodicSegment>, N> DirectNoise<I> for CellularNoise<P, N>
where
    CellNoise<N>: DirectNoise<P::Output>,
{
    type Output = <CellNoise<N> as DirectNoise<P::Output>>::Output;

    #[inline]
    fn raw_sample(&self, input: I) -> Self::Output {
        self.periodic.raw_sample(input).and_then(&self.cell_noise)
    }
}

impl<T, P: ScalableNoise<T>, N> ScalableNoise<T> for CellularNoise<P, N>
where
    Self: Noise,
{
    #[inline]
    fn get_scale(&self) -> T {
        self.periodic.get_scale()
    }

    #[inline]
    fn set_scale(&mut self, period: T) {
        self.periodic.set_scale(period);
    }
}
