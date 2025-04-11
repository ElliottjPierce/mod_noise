//! Contains logic for cellular noise, including ones not based on voronoi graphs.

use super::{
    DirectNoise, DirectNoiseBuilder, Noise, NoiseBuilder,
    periodic::{PeriodicPoint, PeriodicSegment},
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

impl<N: Noise> NoiseBuilder<CellNoise<N>, ()> for DirectNoiseBuilder
where
    Self: NoiseBuilder<N, ()>,
{
    #[inline]
    fn build(&self, seed: &mut SeedGenerator, _scale: ()) -> CellNoise<N> {
        CellNoise {
            noise: NoiseBuilder::<N, ()>::build(self, seed, ()),
            seed: seed.next_seed(),
        }
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
