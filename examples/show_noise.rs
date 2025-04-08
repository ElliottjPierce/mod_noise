//! An example for displaying noise as an image.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use mod_noise::noise::{Noise, NoiseValue, norm::UNorm, white::White32};

fn main() -> AppExit {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(
            Startup,
            |mut commands: Commands, mut images: ResMut<Assets<Image>>| {
                let mut image = Image::new_fill(
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
                make_noise(&mut image);
                let handle = images.add(image);
                commands.spawn((
                    ImageNode {
                        image: handle,
                        ..Default::default()
                    },
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Percent(100.0),
                        ..Default::default()
                    },
                ));
                commands.spawn(Camera2d);
            },
        )
        .run()
}

fn make_noise(image: &mut Image) {
    let width = image.width();
    let height = image.height();
    let noise = White32(7239);

    for x in 0..width {
        for y in 0..height {
            // let loc = Vec2::new(x as f32 - (x / 2) as f32, -(y as f32 - (y / 2) as f32));
            let out = noise.sample([x, y]).map_to::<UNorm>().get();

            // let out = noise.sample(loc).adapt::<f32>();
            let color = Color::linear_rgb(out, out, out);
            if let Err(err) = image.set_color_at(x, y, color) {
                warn!("Failed to set image color with error: {err:?}");
            }
        }
    }
}
