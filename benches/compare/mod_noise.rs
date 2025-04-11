use super::SIZE;
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
}
