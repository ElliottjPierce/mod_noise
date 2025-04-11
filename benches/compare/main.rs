//! Benches this noise lib compared to others.
#![expect(
    missing_docs,
    reason = "Its a benchmark and cirterion macros don't add docs."
)]

mod fastnoise_lite;
mod libnoise;
mod mod_noise;
mod noise_rs;

use criterion::*;

criterion_main!(benches);
criterion_group!(
    benches,
    noise_rs::benches,
    libnoise::benches,
    mod_noise::benches,
    fastnoise_lite::benches
);

const SIZE: u32 = 2048;
