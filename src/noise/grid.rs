//! Periodic noise for orthogonal grids

use bevy_math::{IVec2, IVec3, IVec4, UVec2, UVec3, UVec4, Vec2, Vec3, Vec3A, Vec4};

use super::{
    DirectNoise, Noise, NoiseValue,
    periodic::{
        Frequency, Period, PeriodicPoint, PeriodicPoints, PeriodicSegment, PowerOf2Period,
        RelativePeriodicPoint, ScalableNoise, WholePeriod,
    },
    white::White32,
};

/// A [`ScalableNoise`] that produces [`GridSquare`] using [`PowerOf2Period`].
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct OrthoGridPowerOf2(pub PowerOf2Period);

/// A [`ScalableNoise`] that produces [`GridSquare`] using [`WholePeriod`].
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct OrthoGridInteger(pub WholePeriod);

/// A [`ScalableNoise`] that produces [`GridSquare`] using [`Frequency`]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct OrthoGrid(pub Frequency);

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
    ($u:ty, $i:ty, $f:ty, $f_to_i:ident, $u_to_f:ident, with_int) => {
        impl_grid_dimension!($u, $i, $f, $f_to_i, $u_to_f);

        impl DirectNoise<$u> for OrthoGridInteger {
            type Output = GridSquare<$u, $f>;

            #[inline]
            fn raw_sample(&self, input: $u) -> Self::Output {
                let least = input / self.0.0;
                let left = input - least;
                // x2 + 1 so we are a little off the edges.
                let offset = (left * 2 + <$u>::ONE).$u_to_f() * <$f>::splat(0.5 / self.0.0 as f32);
                GridSquare {
                    least_corner: least,
                    offset_from_corner: offset,
                }
            }
        }

        impl DirectNoise<$u> for OrthoGridPowerOf2 {
            type Output = GridSquare<$u, $f>;

            #[inline]
            fn raw_sample(&self, input: $u) -> Self::Output {
                let least = input >> self.0.0;
                let left = input - least;
                // x2 + 1 so we are a little off the edges.
                let offset =
                    (left * 2 + <$u>::ONE).$u_to_f() * <$f>::splat(0.5 / (1 << self.0.0) as f32);
                GridSquare {
                    least_corner: least,
                    offset_from_corner: offset,
                }
            }
        }

        impl DirectNoise<$i> for OrthoGridInteger {
            type Output = GridSquare<$u, $f>;

            #[inline]
            fn raw_sample(&self, input: $i) -> Self::Output {
                input.map_to::<$u>().and_then(self)
            }
        }

        impl DirectNoise<$i> for OrthoGridPowerOf2 {
            type Output = GridSquare<$u, $f>;

            #[inline]
            fn raw_sample(&self, input: $i) -> Self::Output {
                input.map_to::<$u>().and_then(self)
            }
        }
    };

    ($u:ty, $i:ty, $f:ty, $f_to_i:ident, $u_to_f:ident) => {
        impl PeriodicPoint for OrthoGridLattacePoint<$u, $f> {
            type Relative = $f;

            #[inline]
            fn into_relative(self, entropy: u32) -> RelativePeriodicPoint<$f> {
                RelativePeriodicPoint {
                    offset: self.offset,
                    seed: White32(entropy).raw_sample(self.corner),
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

        impl PeriodicSegment for GridSquare<$u, $f> {
            type Point = OrthoGridLattacePoint<$u, $f>;

            type Points = Self;

            #[inline]
            fn get_main_point(&self) -> Self::Point {
                OrthoGridLattacePoint {
                    corner: self.least_corner,
                    offset: self.offset_from_corner,
                }
            }

            #[inline]
            fn get_points(self) -> Self::Points {
                self
            }
        }

        impl DirectNoise<$f> for OrthoGrid {
            type Output = GridSquare<$u, $f>;

            #[inline]
            fn raw_sample(&self, input: $f) -> Self::Output {
                let scaled = input * self.0.0;
                GridSquare {
                    least_corner: scaled.$f_to_i().map_to::<$u>(),
                    offset_from_corner: scaled.fract_gl(),
                }
            }
        }

        impl PeriodicPoints for GridSquare<$u, $f> {
            type Point = OrthoGridLattacePoint<$u, $f>;

            #[inline]
            fn iter(&self) -> impl Iterator<Item = Self::Point> {
                self.corners().into_iter()
            }
        }
    };
}

impl Noise for OrthoGrid {}
impl Noise for OrthoGridInteger {}
impl Noise for OrthoGridPowerOf2 {}

impl_grid_dimension!(UVec2, IVec2, Vec2, as_ivec2, as_vec2, with_int);
impl_grid_dimension!(UVec3, IVec3, Vec3, as_ivec3, as_vec3, with_int);
impl_grid_dimension!(UVec3, IVec3, Vec3A, as_ivec3, as_vec3a);
impl_grid_dimension!(UVec4, IVec4, Vec4, as_ivec4, as_vec4, with_int);

impl GridSquare<UVec2, Vec2> {
    #[inline]
    fn corners(&self) -> [OrthoGridLattacePoint<UVec2, Vec2>; 4] {
        [
            self.from_offset(UVec2::new(0, 0)),
            self.from_offset(UVec2::new(0, 1)),
            self.from_offset(UVec2::new(1, 0)),
            self.from_offset(UVec2::new(1, 1)),
        ]
    }
}

impl GridSquare<UVec3, Vec3> {
    #[inline]
    fn corners(&self) -> [OrthoGridLattacePoint<UVec3, Vec3>; 8] {
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
    }
}

impl GridSquare<UVec3, Vec3A> {
    #[inline]
    fn corners(&self) -> [OrthoGridLattacePoint<UVec3, Vec3A>; 8] {
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
    }
}

impl GridSquare<UVec4, Vec4> {
    #[inline]
    fn corners(&self) -> [OrthoGridLattacePoint<UVec4, Vec4>; 16] {
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
    }
}

impl ScalableNoise<Frequency> for OrthoGrid {
    #[inline]
    fn get_scale(&self) -> Frequency {
        self.0
    }

    #[inline]
    fn set_scale(&mut self, period: Frequency) {
        self.0 = period;
    }
}

impl ScalableNoise<Period> for OrthoGrid {
    #[inline]
    fn get_scale(&self) -> Period {
        self.0.into()
    }

    #[inline]
    fn set_scale(&mut self, period: Period) {
        self.0 = period.into();
    }
}

impl ScalableNoise<WholePeriod> for OrthoGridInteger {
    #[inline]
    fn get_scale(&self) -> WholePeriod {
        self.0
    }

    #[inline]
    fn set_scale(&mut self, period: WholePeriod) {
        self.0 = period;
    }
}

impl ScalableNoise<PowerOf2Period> for OrthoGridPowerOf2 {
    #[inline]
    fn get_scale(&self) -> PowerOf2Period {
        self.0
    }

    #[inline]
    fn set_scale(&mut self, period: PowerOf2Period) {
        self.0 = period;
    }
}
