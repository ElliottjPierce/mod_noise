//! Periodic noise for orthogonal grids

use bevy_math::{
    Curve, HasTangent, IVec2, IVec3, IVec4, UVec2, UVec3, UVec4, Vec2, Vec3, Vec3A, Vec4,
    curve::derivatives::SampleDerivative,
};

use super::{
    DirectNoise, Noise, NoiseValue,
    curves::Lerpable,
    periodic::{
        DiferentiablePeriodicPoints, Frequency, Period, PeriodicPoint, PeriodicPoints,
        PeriodicSegment, PowerOf2Period, RelativePeriodicPoint, SamplablePeriodicPoints,
        ScalableNoise, WholePeriod,
    },
    white::White32,
};

/// A [`ScalableNoise`] that produces [`GridSquare`] using [`PowerOf2Period`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrthoGridPowerOf2(pub PowerOf2Period);

impl Default for OrthoGridPowerOf2 {
    #[inline]
    fn default() -> Self {
        Self(Default::default())
    }
}

/// A [`ScalableNoise`] that produces [`GridSquare`] using [`WholePeriod`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrthoGridInteger(pub WholePeriod);

impl Default for OrthoGridInteger {
    #[inline]
    fn default() -> Self {
        Self(Default::default())
    }
}

/// A [`ScalableNoise`] that produces [`GridSquare`] using [`Frequency`]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrthoGrid(pub Frequency);

impl Default for OrthoGrid {
    #[inline]
    fn default() -> Self {
        Self(Default::default())
    }
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
                    // needing floor is annoying, but fract_gl calls it anyway, and it prevents some artifacts.
                    least_corner: scaled.floor().$f_to_i().map_to::<$u>(),
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
        self.corners_map(|x| x)
    }

    #[inline]
    fn corners_map<T>(&self, mut f: impl FnMut(OrthoGridLattacePoint<UVec2, Vec2>) -> T) -> [T; 4] {
        [
            f(self.from_offset(UVec2::new(0, 0))),
            f(self.from_offset(UVec2::new(0, 1))),
            f(self.from_offset(UVec2::new(1, 0))),
            f(self.from_offset(UVec2::new(1, 1))),
        ]
    }
}

impl SamplablePeriodicPoints for GridSquare<UVec2, Vec2> {
    #[inline]
    fn sample_smooth<T: Lerpable>(
        &self,
        f: impl FnMut(Self::Point) -> T,
        curve: &impl Curve<f32>,
    ) -> T {
        // points
        let [ld, lu, rd, ru] = self.corners_map(f);
        let mix = self.offset_from_corner.map(|t| curve.sample_unchecked(t));

        // lerp
        let l = ld.lerp_to(lu).sample_unchecked(mix.y);
        let r = rd.lerp_to(ru).sample_unchecked(mix.y);
        l.lerp_to(r).sample_unchecked(mix.x)
    }
}

impl DiferentiablePeriodicPoints for GridSquare<UVec2, Vec2> {
    type Gradient<D> = [D; 2];

    #[inline]
    fn sample_gradient_smooth<T: HasTangent<Tangent: Lerpable>>(
        &self,
        f: impl FnMut(Self::Point) -> T,
        difference: impl Fn(&T, &T) -> T::Tangent,
        curve: &impl SampleDerivative<f32>,
    ) -> Self::Gradient<T::Tangent> {
        // points
        let [ld, lu, rd, ru] = self.corners_map(f);
        let [mix_x, mix_y] = self
            .offset_from_corner
            .to_array()
            .map(|t| curve.sample_with_derivative_unchecked(t));

        // derivatives
        let ld_lu = difference(&ld, &lu);
        let rd_ru = difference(&rd, &ru);
        let ld_rd = difference(&ld, &rd);
        let lu_ru = difference(&lu, &ru);

        // lerp
        let dx = ld_rd.lerp_to(lu_ru).sample_unchecked(mix_y.value) * mix_x.derivative;
        let dy = ld_lu.lerp_to(rd_ru).sample_unchecked(mix_x.value) * mix_y.derivative;
        [dx, dy]
    }
}

impl GridSquare<UVec3, Vec3> {
    #[inline]
    fn corners(&self) -> [OrthoGridLattacePoint<UVec3, Vec3>; 8] {
        self.corners_map(|x| x)
    }

    #[inline]
    fn corners_map<T>(&self, mut f: impl FnMut(OrthoGridLattacePoint<UVec3, Vec3>) -> T) -> [T; 8] {
        [
            f(self.from_offset(UVec3::new(0, 0, 0))),
            f(self.from_offset(UVec3::new(0, 0, 1))),
            f(self.from_offset(UVec3::new(0, 1, 0))),
            f(self.from_offset(UVec3::new(0, 1, 1))),
            f(self.from_offset(UVec3::new(1, 0, 0))),
            f(self.from_offset(UVec3::new(1, 0, 1))),
            f(self.from_offset(UVec3::new(1, 1, 0))),
            f(self.from_offset(UVec3::new(1, 1, 1))),
        ]
    }
}

impl SamplablePeriodicPoints for GridSquare<UVec3, Vec3> {
    #[inline]
    fn sample_smooth<T: Lerpable>(
        &self,
        f: impl FnMut(Self::Point) -> T,
        curve: &impl Curve<f32>,
    ) -> T {
        // points
        let [ldb, ldf, lub, luf, rdb, rdf, rub, ruf] = self.corners_map(f);
        let mix = self.offset_from_corner.map(|t| curve.sample_unchecked(t));

        // lerp
        let ld = ldb.lerp_to(ldf).sample_unchecked(mix.z);
        let lu = lub.lerp_to(luf).sample_unchecked(mix.z);
        let rd = rdb.lerp_to(rdf).sample_unchecked(mix.z);
        let ru = rub.lerp_to(ruf).sample_unchecked(mix.z);
        let l = ld.lerp_to(lu).sample_unchecked(mix.y);
        let r = rd.lerp_to(ru).sample_unchecked(mix.y);
        l.lerp_to(r).sample_unchecked(mix.x)
    }
}

impl DiferentiablePeriodicPoints for GridSquare<UVec3, Vec3> {
    type Gradient<D> = [D; 3];

    #[inline]
    fn sample_gradient_smooth<T: HasTangent<Tangent: Lerpable>>(
        &self,
        f: impl FnMut(Self::Point) -> T,
        difference: impl Fn(&T, &T) -> T::Tangent,
        curve: &impl SampleDerivative<f32>,
    ) -> Self::Gradient<T::Tangent> {
        // points// points
        let [ldb, ldf, lub, luf, rdb, rdf, rub, ruf] = self.corners_map(f);
        let [mix_x, mix_y, mix_z] = self
            .offset_from_corner
            .to_array()
            .map(|t| curve.sample_with_derivative_unchecked(t));

        // derivatives
        let ldb_ldf = difference(&ldb, &ldf);
        let lub_luf = difference(&lub, &luf);
        let rdb_rdf = difference(&rdb, &rdf);
        let rub_ruf = difference(&rub, &ruf);

        let ldb_lub = difference(&ldb, &lub);
        let ldf_luf = difference(&ldf, &luf);
        let rdb_rub = difference(&rdb, &rub);
        let rdf_ruf = difference(&rdf, &ruf);

        let ldb_rdb = difference(&ldb, &rdb);
        let ldf_rdf = difference(&ldf, &rdf);
        let lub_rub = difference(&lub, &rub);
        let luf_ruf = difference(&luf, &ruf);

        // lerp
        let dx = {
            let d = ldb_rdb.lerp_to(ldf_rdf).sample_unchecked(mix_z.value);
            let u = lub_rub.lerp_to(luf_ruf).sample_unchecked(mix_z.value);
            d.lerp_to(u).sample_unchecked(mix_y.value)
        } * mix_x.derivative;
        let dy = {
            let l = ldb_lub.lerp_to(ldf_luf).sample_unchecked(mix_z.value);
            let r = rdb_rub.lerp_to(rdf_ruf).sample_unchecked(mix_z.value);
            l.lerp_to(r).sample_unchecked(mix_x.value)
        } * mix_y.derivative;
        let dz = {
            let l = ldb_ldf.lerp_to(lub_luf).sample_unchecked(mix_y.value);
            let r = rdb_rdf.lerp_to(rub_ruf).sample_unchecked(mix_y.value);
            l.lerp_to(r).sample_unchecked(mix_x.value)
        } * mix_z.derivative;

        [dx, dy, dz]
    }
}

impl GridSquare<UVec3, Vec3A> {
    #[inline]
    fn corners(&self) -> [OrthoGridLattacePoint<UVec3, Vec3A>; 8] {
        self.corners_map(|x| x)
    }

    #[inline]
    fn corners_map<T>(
        &self,
        mut f: impl FnMut(OrthoGridLattacePoint<UVec3, Vec3A>) -> T,
    ) -> [T; 8] {
        [
            f(self.from_offset(UVec3::new(0, 0, 0))),
            f(self.from_offset(UVec3::new(0, 0, 1))),
            f(self.from_offset(UVec3::new(0, 1, 0))),
            f(self.from_offset(UVec3::new(0, 1, 1))),
            f(self.from_offset(UVec3::new(1, 0, 0))),
            f(self.from_offset(UVec3::new(1, 0, 1))),
            f(self.from_offset(UVec3::new(1, 1, 0))),
            f(self.from_offset(UVec3::new(1, 1, 1))),
        ]
    }
}

impl SamplablePeriodicPoints for GridSquare<UVec3, Vec3A> {
    #[inline]
    fn sample_smooth<T: Lerpable>(
        &self,
        f: impl FnMut(Self::Point) -> T,
        curve: &impl Curve<f32>,
    ) -> T {
        // points
        let [ldb, ldf, lub, luf, rdb, rdf, rub, ruf] = self.corners_map(f);
        let mix = self.offset_from_corner.map(|t| curve.sample_unchecked(t));

        // lerp
        let ld = ldb.lerp_to(ldf).sample_unchecked(mix.z);
        let lu = lub.lerp_to(luf).sample_unchecked(mix.z);
        let rd = rdb.lerp_to(rdf).sample_unchecked(mix.z);
        let ru = rub.lerp_to(ruf).sample_unchecked(mix.z);
        let l = ld.lerp_to(lu).sample_unchecked(mix.y);
        let r = rd.lerp_to(ru).sample_unchecked(mix.y);
        l.lerp_to(r).sample_unchecked(mix.x)
    }
}

impl DiferentiablePeriodicPoints for GridSquare<UVec3, Vec3A> {
    type Gradient<D> = [D; 3];

    #[inline]
    fn sample_gradient_smooth<T: HasTangent<Tangent: Lerpable>>(
        &self,
        f: impl FnMut(Self::Point) -> T,
        difference: impl Fn(&T, &T) -> T::Tangent,
        curve: &impl SampleDerivative<f32>,
    ) -> Self::Gradient<T::Tangent> {
        // points// points
        let [ldb, ldf, lub, luf, rdb, rdf, rub, ruf] = self.corners_map(f);
        let [mix_x, mix_y, mix_z] = self
            .offset_from_corner
            .to_array()
            .map(|t| curve.sample_with_derivative_unchecked(t));

        // derivatives
        let ldb_ldf = difference(&ldb, &ldf);
        let lub_luf = difference(&lub, &luf);
        let rdb_rdf = difference(&rdb, &rdf);
        let rub_ruf = difference(&rub, &ruf);

        let ldb_lub = difference(&ldb, &lub);
        let ldf_luf = difference(&ldf, &luf);
        let rdb_rub = difference(&rdb, &rub);
        let rdf_ruf = difference(&rdf, &ruf);

        let ldb_rdb = difference(&ldb, &rdb);
        let ldf_rdf = difference(&ldf, &rdf);
        let lub_rub = difference(&lub, &rub);
        let luf_ruf = difference(&luf, &ruf);

        // lerp
        let dx = {
            let d = ldb_rdb.lerp_to(ldf_rdf).sample_unchecked(mix_z.value);
            let u = lub_rub.lerp_to(luf_ruf).sample_unchecked(mix_z.value);
            d.lerp_to(u).sample_unchecked(mix_y.value)
        } * mix_x.derivative;
        let dy = {
            let l = ldb_lub.lerp_to(ldf_luf).sample_unchecked(mix_z.value);
            let r = rdb_rub.lerp_to(rdf_ruf).sample_unchecked(mix_z.value);
            l.lerp_to(r).sample_unchecked(mix_x.value)
        } * mix_y.derivative;
        let dz = {
            let l = ldb_ldf.lerp_to(lub_luf).sample_unchecked(mix_y.value);
            let r = rdb_rdf.lerp_to(rub_ruf).sample_unchecked(mix_y.value);
            l.lerp_to(r).sample_unchecked(mix_x.value)
        } * mix_z.derivative;

        [dx, dy, dz]
    }
}

impl GridSquare<UVec4, Vec4> {
    #[inline]
    fn corners(&self) -> [OrthoGridLattacePoint<UVec4, Vec4>; 16] {
        self.corners_map(|x| x)
    }

    #[inline]
    fn corners_map<T>(
        &self,
        mut f: impl FnMut(OrthoGridLattacePoint<UVec4, Vec4>) -> T,
    ) -> [T; 16] {
        [
            f(self.from_offset(UVec4::new(0, 0, 0, 0))),
            f(self.from_offset(UVec4::new(0, 0, 0, 1))),
            f(self.from_offset(UVec4::new(0, 0, 1, 0))),
            f(self.from_offset(UVec4::new(0, 0, 1, 1))),
            f(self.from_offset(UVec4::new(0, 1, 0, 0))),
            f(self.from_offset(UVec4::new(0, 1, 0, 1))),
            f(self.from_offset(UVec4::new(0, 1, 1, 0))),
            f(self.from_offset(UVec4::new(0, 1, 1, 1))),
            f(self.from_offset(UVec4::new(1, 0, 0, 0))),
            f(self.from_offset(UVec4::new(1, 0, 0, 1))),
            f(self.from_offset(UVec4::new(1, 0, 1, 0))),
            f(self.from_offset(UVec4::new(1, 0, 1, 1))),
            f(self.from_offset(UVec4::new(1, 1, 0, 0))),
            f(self.from_offset(UVec4::new(1, 1, 0, 1))),
            f(self.from_offset(UVec4::new(1, 1, 1, 0))),
            f(self.from_offset(UVec4::new(1, 1, 1, 1))),
        ]
    }
}

impl SamplablePeriodicPoints for GridSquare<UVec4, Vec4> {
    #[inline]
    fn sample_smooth<T: Lerpable>(
        &self,
        f: impl FnMut(Self::Point) -> T,
        curve: &impl Curve<f32>,
    ) -> T {
        // points
        let [
            ldbp,
            ldbq,
            ldfp,
            ldfq,
            lubp,
            lubq,
            lufp,
            lufq,
            rdbp,
            rdbq,
            rdfp,
            rdfq,
            rubp,
            rubq,
            rufp,
            rufq,
        ] = self.corners_map(f);
        let mix = self.offset_from_corner.map(|t| curve.sample_unchecked(t));

        // lerp
        let ldb = ldbp.lerp_to(ldbq).sample_unchecked(mix.w);
        let ldf = ldfp.lerp_to(ldfq).sample_unchecked(mix.w);
        let lub = lubp.lerp_to(lubq).sample_unchecked(mix.w);
        let luf = lufp.lerp_to(lufq).sample_unchecked(mix.w);
        let rdb = rdbp.lerp_to(rdbq).sample_unchecked(mix.w);
        let rdf = rdfp.lerp_to(rdfq).sample_unchecked(mix.w);
        let rub = rubp.lerp_to(rubq).sample_unchecked(mix.w);
        let ruf = rufp.lerp_to(rufq).sample_unchecked(mix.w);
        let ld = ldb.lerp_to(ldf).sample_unchecked(mix.z);
        let lu = lub.lerp_to(luf).sample_unchecked(mix.z);
        let rd = rdb.lerp_to(rdf).sample_unchecked(mix.z);
        let ru = rub.lerp_to(ruf).sample_unchecked(mix.z);
        let l = ld.lerp_to(lu).sample_unchecked(mix.y);
        let r = rd.lerp_to(ru).sample_unchecked(mix.y);
        l.lerp_to(r).sample_unchecked(mix.x)
    }
}

impl DiferentiablePeriodicPoints for GridSquare<UVec4, Vec4> {
    type Gradient<D> = [D; 4];

    #[inline]
    fn sample_gradient_smooth<T: HasTangent<Tangent: Lerpable>>(
        &self,
        f: impl FnMut(Self::Point) -> T,
        difference: impl Fn(&T, &T) -> T::Tangent,
        curve: &impl SampleDerivative<f32>,
    ) -> Self::Gradient<T::Tangent> {
        // points// points
        let [
            ldbp,
            ldbq,
            ldfp,
            ldfq,
            lubp,
            lubq,
            lufp,
            lufq,
            rdbp,
            rdbq,
            rdfp,
            rdfq,
            rubp,
            rubq,
            rufp,
            rufq,
        ] = self.corners_map(f);
        let [mix_x, mix_y, mix_z, mix_w] = self
            .offset_from_corner
            .to_array()
            .map(|t| curve.sample_with_derivative_unchecked(t));

        // derivatives
        let ldbp_ldbq = difference(&ldbp, &ldbq);
        let ldfp_ldfq = difference(&ldfp, &ldfq);
        let lubp_lubq = difference(&lubp, &lubq);
        let lufp_lufq = difference(&lufp, &lufq);
        let rdbp_rdbq = difference(&rdbp, &rdbq);
        let rdfp_rdfq = difference(&rdfp, &rdfq);
        let rubp_rubq = difference(&rubp, &rubq);
        let rufp_rufq = difference(&rufp, &rufq);

        let ldbp_ldfp = difference(&ldbp, &ldfp);
        let lubp_lufp = difference(&lubp, &lufp);
        let rdbp_rdfp = difference(&rdbp, &rdfp);
        let rubp_rufp = difference(&rubp, &rufp);
        let ldbq_ldfq = difference(&ldbq, &ldfq);
        let lubq_lufq = difference(&lubq, &lufq);
        let rdbq_rdfq = difference(&rdbq, &rdfq);
        let rubq_rufq = difference(&rubq, &rufq);

        let ldbp_lubp = difference(&ldbp, &lubp);
        let ldfp_lufp = difference(&ldfp, &lufp);
        let rdbp_rubp = difference(&rdbp, &rubp);
        let rdfp_rufp = difference(&rdfp, &rufp);
        let ldbq_lubq = difference(&ldbq, &lubq);
        let ldfq_lufq = difference(&ldfq, &lufq);
        let rdbq_rubq = difference(&rdbq, &rubq);
        let rdfq_rufq = difference(&rdfq, &rufq);

        let ldbp_rdbp = difference(&ldbp, &rdbp);
        let ldfp_rdfp = difference(&ldfp, &rdfp);
        let lubp_rubp = difference(&lubp, &rubp);
        let lufp_rufp = difference(&lufp, &rufp);
        let ldbq_rdbq = difference(&ldbq, &rdbq);
        let ldfq_rdfq = difference(&ldfq, &rdfq);
        let lubq_rubq = difference(&lubq, &rubq);
        let lufq_rufq = difference(&lufq, &rufq);

        // lerp
        let dx = {
            let db = ldbp_rdbp.lerp_to(ldbq_rdbq).sample_unchecked(mix_w.value);
            let df = ldfp_rdfp.lerp_to(ldfq_rdfq).sample_unchecked(mix_w.value);
            let ub = lubp_rubp.lerp_to(lubq_rubq).sample_unchecked(mix_w.value);
            let uf = lufp_rufp.lerp_to(lufq_rufq).sample_unchecked(mix_w.value);
            let d = db.lerp_to(df).sample_unchecked(mix_z.value);
            let u = ub.lerp_to(uf).sample_unchecked(mix_z.value);
            d.lerp_to(u).sample_unchecked(mix_y.value)
        } * mix_x.derivative;
        let dy = {
            let lb = ldbp_lubp.lerp_to(ldbq_lubq).sample_unchecked(mix_w.value);
            let lf = ldfp_lufp.lerp_to(ldfq_lufq).sample_unchecked(mix_w.value);
            let rb = rdbp_rubp.lerp_to(rdbq_rubq).sample_unchecked(mix_w.value);
            let rf = rdfp_rufp.lerp_to(rdfq_rufq).sample_unchecked(mix_w.value);
            let l = lb.lerp_to(lf).sample_unchecked(mix_z.value);
            let r = rb.lerp_to(rf).sample_unchecked(mix_z.value);
            l.lerp_to(r).sample_unchecked(mix_x.value)
        } * mix_y.derivative;
        let dz = {
            let ld = ldbp_ldfp.lerp_to(ldbq_ldfq).sample_unchecked(mix_w.value);
            let lu = lubp_lufp.lerp_to(lubq_lufq).sample_unchecked(mix_w.value);
            let rd = rdbp_rdfp.lerp_to(rdbq_rdfq).sample_unchecked(mix_w.value);
            let ru = rubp_rufp.lerp_to(rubq_rufq).sample_unchecked(mix_w.value);
            let d = ld.lerp_to(rd).sample_unchecked(mix_x.value);
            let u = lu.lerp_to(ru).sample_unchecked(mix_x.value);
            d.lerp_to(u).sample_unchecked(mix_y.value)
        } * mix_z.derivative;
        let dw = {
            let ld = ldbp_ldbq.lerp_to(ldfp_ldfq).sample_unchecked(mix_z.value);
            let lu = lubp_lubq.lerp_to(lufp_lufq).sample_unchecked(mix_z.value);
            let rd = rdbp_rdbq.lerp_to(rdfp_rdfq).sample_unchecked(mix_z.value);
            let ru = rubp_rubq.lerp_to(rufp_rufq).sample_unchecked(mix_z.value);
            let d = ld.lerp_to(rd).sample_unchecked(mix_x.value);
            let u = lu.lerp_to(ru).sample_unchecked(mix_x.value);
            d.lerp_to(u).sample_unchecked(mix_y.value)
        } * mix_w.derivative;
        [dx, dy, dz, dw]
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
