//! Benches this noise lib compared to others.
#![expect(
    missing_docs,
    reason = "Its a benchmark and cirterion macros don't add docs."
)]

use bevy_math::Vec2;
use criterion::*;
use mod_noise::noise::{
    DirectNoise, NoiseExt,
    curves::Smoothstep,
    gradient::{QuickGradients, SegmentalGradientNoise},
    grid::OrthoGrid,
    norm::UNorm,
    periodic::{Period, TilingNoise},
};

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
}
