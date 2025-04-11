use super::SIZE;
use bevy_math::Vec2;
use criterion::*;
use mod_noise::noise::{
    DirectNoise, DirectNoiseBuilder, NoiseBuilderBase, NoiseExt,
    adapters::Adapter,
    curves::Smoothstep,
    gradient::{QuickGradients, SegmentalGradientNoise},
    grid::OrthoGrid,
    layering::{
        FractalNoise, FractalScaling, NoiseLayerBase, NormalizeOctavesInto, ProportionalAmplitude,
    },
    norm::UNorm,
    periodic::{Frequency, Period, TilingNoise},
    white::SeedGenerator,
};

#[inline]
fn bench_2d(noise: impl DirectNoise<Vec2, Output = UNorm>) -> f32 {
    let mut res = 0.0;
    for x in 0..SIZE {
        for y in 0..SIZE {
            res += noise.raw_sample(Vec2::new(x as f32, y as f32)).get();
        }
    }
    res
}

pub fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("mod_noise");
    group.warm_up_time(core::time::Duration::from_millis(500));
    group.measurement_time(core::time::Duration::from_secs(4));

    group.bench_function("perlin", |bencher| {
        bencher.iter(|| {
            let noise = TilingNoise::<
                OrthoGrid,
                SegmentalGradientNoise<QuickGradients, Smoothstep>,
            >::default().with_period(Period(32.0));
            bench_2d(noise)
        });
    });

    group.bench_function("fbm 8 octave perlin", |bencher| {
        bencher.iter(|| {
            let noise =
                FractalNoise {
                    scale: FractalScaling {
                        overall: Period(32.0).into(),
                        gain: 2.0,
                    },
                    amplitude: ProportionalAmplitude {
                        proportion: 0.5,
                        ..Default::default()
                    },
                    result: NormalizeOctavesInto::<f32>::default(),
                    finalizer: Adapter::<UNorm>::default(),
                    seed: SeedGenerator::default(),
                    octaves:
                        DirectNoiseBuilder
                            .build_octave_for::<Frequency, TilingNoise<
                                OrthoGrid,
                                SegmentalGradientNoise<QuickGradients, Smoothstep>,
                            >>()
                            .repeat(8),
                };
            bench_2d(noise)
        });
    });
}
