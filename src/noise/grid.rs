//! Periodic noise for orthogonal grids

use bevy_math::{IVec2, UVec2, UVec3, UVec4, Vec2, Vec3, Vec3A, Vec4};

use super::{
    DirectNoise, Noise, NoiseValue,
    periodic::{PeriodicPoint, PeriodicPoints, RelativePeriodicPoint},
    white::White32,
};

/// A [`PeriodicNoise`] that produces [`GridSquare`]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrthoGrid {
    frequency: f32,
}

/// Represents a grid square.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridSquare<Z, R> {
    /// The least corner of this grid square.
    pub least_corner: Z,
    /// The positive offset from [`least_corner`](Self::least_corner) to the point in the grid square.
    pub offset_from_corner: R,
}

/// Represents a point for some [`GridSquare`]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrthoGridLattacePoint<Z, R> {
    /// Some corner of a [`GridSquare`].
    pub corner: Z,
    /// The offset from [`corner`](Self::corner) to the point in the [`GridSquare`].
    pub offset: R,
}

macro_rules! impl_grid_dimension {
    ($u:ty, $f:ty, $f_to_u:ident, $u_to_f:ident) => {
        impl PeriodicPoint<$f> for OrthoGridLattacePoint<$u, $f> {
            #[inline]
            fn into_relative(self, entropy: u32) -> RelativePeriodicPoint<$f> {
                RelativePeriodicPoint {
                    offset: self.offset,
                    seed: White32(entropy).sample(self.corner),
                }
            }
        }

        impl GridSquare<$u, $f> {
            /// Returns the [`OrthoGridLattacePoint`] for this [`GridSquare`] at `push` offset on the grid.
            #[inline]
            pub fn from_offset(&self, push: $u) -> OrthoGridLattacePoint<$u, $f> {
                OrthoGridLattacePoint {
                    corner: self.least_corner + push,
                    offset: self.offset_from_corner - push.$u_to_f(),
                }
            }
        }

        impl NoiseValue for GridSquare<$u, $f> {}

        impl DirectNoise<$f> for OrthoGrid {
            type Output = GridSquare<$u, $f>;

            #[inline]
            fn raw_sample(&self, input: $f) -> Self::Output {
                let scaled = input * self.frequency;
                GridSquare {
                    least_corner: scaled.$f_to_u(),
                    offset_from_corner: scaled.fract_gl(),
                }
            }
        }
    };
}

impl Noise for OrthoGrid {}

impl_grid_dimension!(UVec2, Vec2, as_uvec2, as_vec2);
impl_grid_dimension!(UVec3, Vec3, as_uvec3, as_vec3);
impl_grid_dimension!(UVec3, Vec3A, as_uvec3, as_vec3a);
impl_grid_dimension!(UVec4, Vec4, as_uvec4, as_vec4);

impl PeriodicPoints<Vec2> for GridSquare<UVec2, Vec2> {
    type Point = OrthoGridLattacePoint<UVec2, Vec2>;

    fn iter(&self) -> impl Iterator<Item = Self::Point> {
        [
            self.from_offset(UVec2::new(0, 0)),
            self.from_offset(UVec2::new(0, 1)),
            self.from_offset(UVec2::new(1, 0)),
            self.from_offset(UVec2::new(1, 1)),
        ]
        .into_iter()
    }
}

impl PeriodicPoints<Vec3> for GridSquare<UVec3, Vec3> {
    type Point = OrthoGridLattacePoint<UVec3, Vec3>;

    fn iter(&self) -> impl Iterator<Item = Self::Point> {
        [
            self.from_offset(UVec3::new(0, 0, 0)),
            self.from_offset(UVec3::new(0, 0, 1)),
            self.from_offset(UVec3::new(0, 1, 0)),
            self.from_offset(UVec3::new(0, 1, 1)),
            self.from_offset(UVec3::new(1, 0, 0)),
            self.from_offset(UVec3::new(1, 0, 1)),
            self.from_offset(UVec3::new(1, 1, 0)),
            self.from_offset(UVec3::new(1, 1, 1)),
        ]
        .into_iter()
    }
}

impl PeriodicPoints<Vec3A> for GridSquare<UVec3, Vec3A> {
    type Point = OrthoGridLattacePoint<UVec3, Vec3A>;

    fn iter(&self) -> impl Iterator<Item = Self::Point> {
        [
            self.from_offset(UVec3::new(0, 0, 0)),
            self.from_offset(UVec3::new(0, 0, 1)),
            self.from_offset(UVec3::new(0, 1, 0)),
            self.from_offset(UVec3::new(0, 1, 1)),
            self.from_offset(UVec3::new(1, 0, 0)),
            self.from_offset(UVec3::new(1, 0, 1)),
            self.from_offset(UVec3::new(1, 1, 0)),
            self.from_offset(UVec3::new(1, 1, 1)),
        ]
        .into_iter()
    }
}

impl PeriodicPoints<Vec4> for GridSquare<UVec4, Vec4> {
    type Point = OrthoGridLattacePoint<UVec4, Vec4>;

    fn iter(&self) -> impl Iterator<Item = Self::Point> {
        [
            self.from_offset(UVec4::new(0, 0, 0, 0)),
            self.from_offset(UVec4::new(0, 0, 0, 1)),
            self.from_offset(UVec4::new(0, 0, 1, 0)),
            self.from_offset(UVec4::new(0, 0, 1, 1)),
            self.from_offset(UVec4::new(0, 1, 0, 0)),
            self.from_offset(UVec4::new(0, 1, 0, 1)),
            self.from_offset(UVec4::new(0, 1, 1, 0)),
            self.from_offset(UVec4::new(0, 1, 1, 1)),
            self.from_offset(UVec4::new(1, 0, 0, 0)),
            self.from_offset(UVec4::new(1, 0, 0, 1)),
            self.from_offset(UVec4::new(1, 0, 1, 0)),
            self.from_offset(UVec4::new(1, 0, 1, 1)),
            self.from_offset(UVec4::new(1, 1, 0, 0)),
            self.from_offset(UVec4::new(1, 1, 0, 1)),
            self.from_offset(UVec4::new(1, 1, 1, 0)),
            self.from_offset(UVec4::new(1, 1, 1, 1)),
        ]
        .into_iter()
    }
}
