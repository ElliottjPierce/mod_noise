use super::SIZE;
use criterion::*;
use noise as noise_rs;
use noise_rs::{NoiseFn, Perlin};

pub fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("noise");
    group.warm_up_time(core::time::Duration::from_millis(500));
    group.measurement_time(core::time::Duration::from_secs(4));

    group.bench_function("perlin", |bencher| {
        bencher.iter(|| {
            let noise = Perlin::new(Perlin::DEFAULT_SEED);
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
}
