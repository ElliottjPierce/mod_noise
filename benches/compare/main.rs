//! Benches this noise lib compared to others.
#![expect(
    missing_docs,
    reason = "Its a benchmark and cirterion macros don't add docs."
)]

use bevy_math::Vec2;
use criterion::*;
use fastnoise_lite::FastNoiseLite;
use libnoise::Generator as _;
use mod_noise::noise::{
    DirectNoise, NoiseExt,
    curves::Smoothstep,
    gradient::{QuickGradients, SegmentalGradientNoise},
    grid::OrthoGrid,
    norm::UNorm,
    periodic::{Period, TilingNoise},
};
use noise as noise_rs;
use noise_rs::NoiseFn;

criterion_main!(benches);
criterion_group!(benches, bench_compare);

const SIZE: u32 = 2048;

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

fn bench_compare(c: &mut Criterion) {
    let mut perlin = c.benchmark_group("perlin");
    perlin.warm_up_time(core::time::Duration::from_millis(500));
    perlin.measurement_time(core::time::Duration::from_secs(4));

    perlin.bench_function("mod_noise", |bencher| {
        bencher.iter(|| {
            let noise = TilingNoise::<
                OrthoGrid,
                SegmentalGradientNoise<QuickGradients, Smoothstep>,
            >::default().with_period(Period(32.0));
            bench_2d(noise)
        });
    });
    perlin.bench_function("fastnoise-lite", |bencher| {
        bencher.iter(|| {
            let mut noise = FastNoiseLite::new();
            noise.set_noise_type(Some(fastnoise_lite::NoiseType::Perlin));
            noise.set_fractal_type(None);
            noise.frequency = 1.0 / 32.0;
            let mut res = 0.0;
            for x in 0..SIZE {
                for y in 0..SIZE {
                    res += noise.get_noise_2d(x as f32, y as f32);
                }
            }
            res
        });
    });
    perlin.bench_function("'noise'", |bencher| {
        bencher.iter(|| {
            let noise = noise_rs::Perlin::new(noise_rs::Perlin::DEFAULT_SEED);
            let frequency = 1.0 / 32.0;
            let mut res = 0.0;
            for x in 0..SIZE {
                for y in 0..SIZE {
                    res +=
                        noise.get([(x as f32 * frequency) as f64, (y as f32 * frequency) as f64]);
                }
            }
            res
        });
    });
    perlin.bench_function("libnoise", |bencher| {
        bencher.iter(|| {
            let noise = libnoise::Perlin::<2>::new(0);
            let frequency = 1.0 / 32.0;
            let mut res = 0.0;
            for x in 0..SIZE {
                for y in 0..SIZE {
                    res += noise
                        .sample([(x as f32 * frequency) as f64, (y as f32 * frequency) as f64]);
                }
            }
            res
        });
    });
}
