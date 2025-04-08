//! An example for displaying noise as an image.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use mod_noise::noise::{
    DirectNoise,
    adapters::Adapter,
    cellular::{CellNoise, CellularNoise},
    grid::OrthoGrid,
    norm::UNorm,
    periodic::{Frequency, Period, PeriodicNoise},
    white::SeedGenerator,
};

/// Holds a version of the noise
pub struct NoiseOption {
    frequency: Frequency,
    seed: u64,
    noise: Box<dyn DirectNoise<Vec2, Output = UNorm>>,
}

impl NoiseOption {
    /// Displays the noise on the image.
    pub fn display_image(&self, image: &mut Image) {
        let width = image.width();
        let height = image.height();

        for x in 0..width {
            for y in 0..height {
                let loc = Vec2::new(x as f32 - (x / 2) as f32, -(y as f32 - (y / 2) as f32));
                let out = self.noise.raw_sample(loc).get();

                let color = Color::linear_rgb(out, out, out);
                if let Err(err) = image.set_color_at(x, y, color) {
                    warn!("Failed to set image color with error: {err:?}");
                }
            }
        }
    }
}

/// Holds the current noise
#[derive(Resource)]
pub struct NoiseOptions {
    options: Vec<NoiseOption>,
    selected: usize,
    image: Handle<Image>,
}

fn main() -> AppExit {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(
            Startup,
            |mut commands: Commands, mut images: ResMut<Assets<Image>>| {
                let image = Image::new_fill(
                    Extent3d {
                        width: 1920,
                        height: 1080,
                        depth_or_array_layers: 1,
                    },
                    TextureDimension::D2,
                    &[255, 255, 255, 255, 255, 255, 255, 255],
                    TextureFormat::Rgba16Unorm,
                    RenderAssetUsages::all(),
                );
                let handle = images.add(image);
                commands.spawn((
                    ImageNode {
                        image: handle.clone(),
                        ..Default::default()
                    },
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Percent(100.0),
                        ..Default::default()
                    },
                ));
                commands.spawn(Camera2d);
                commands.insert_resource(NoiseOptions {
                    options: vec![NoiseOption {
                        frequency: Period(32.0).into(),
                        seed: 0,
                        noise: Box::new(CellularNoise {
                            periodic: OrthoGrid::default(),
                            cell_noise: CellNoise {
                                noise: Adapter::<UNorm>::default(),
                                seed: 0,
                            },
                        }),
                    }],
                    selected: 0,
                    image: handle,
                });
            },
        )
        .add_systems(Update, update_system)
        .run()
}

fn update_system(
    mut noise: ResMut<NoiseOptions>,
    mut images: ResMut<Assets<Image>>,
    input: Res<ButtonInput<KeyCode>>,
) {
    let mut changed = false;

    if input.just_pressed(KeyCode::ArrowRight) {
        noise.selected = (noise.selected.wrapping_add(1)) % noise.options.len();
        changed = true;
    }
    if input.just_pressed(KeyCode::ArrowLeft) {
        noise.selected = (noise.selected.wrapping_sub(1)) % noise.options.len();
        changed = true;
    }

    let image = noise.image.id();
    let selected = noise.selected;
    let current = &mut noise.options[selected];

    if input.just_pressed(KeyCode::KeyW) {
        current.seed = current.seed.wrapping_add(1);
        changed = true;
    }
    if input.just_pressed(KeyCode::KeyS) {
        current.seed = current.seed.wrapping_sub(1);
        changed = true;
    }

    if input.just_pressed(KeyCode::KeyD) {
        current.frequency.0 *= 2.0;
        changed = true;
    }
    if input.just_pressed(KeyCode::KeyA) {
        current.frequency.0 *= 0.5;
        changed = true;
    }

    if changed {
        current
            .noise
            .set_seed(&mut SeedGenerator::new_from_u64(current.seed));
        current.display_image(images.get_mut(image).unwrap());
    }
}
