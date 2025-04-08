//! Contains common mapping.

use super::{CorolatedNoiseType, NoiseValue};

/// easily implement mapping for integers
macro_rules! impl_mapper {
    ($s:ty, $u:ty) => {
        impl CorolatedNoiseType<$s> for $u {
            #[inline]
            fn map_from(value: $s) -> Self {
                (value as $u) ^ (1 << (<$u>::BITS - 1))
            }
        }
    };
}

impl_mapper!(i8, u8);
impl_mapper!(i16, u16);
impl_mapper!(i32, u32);
impl_mapper!(i64, u64);
impl_mapper!(i128, u128);

/// easily implement mapping for integer vecs
macro_rules! impl_mapper_vec {
    ($s:ty, $u:ty) => {
        impl CorolatedNoiseType<$s> for $u {
            #[inline]
            fn map_from(value: $s) -> Self {
                Self::from_array(value.to_array().map(|v| v.map_to()))
            }
        }
    };
}

impl_mapper_vec!(bevy_math::I8Vec2, bevy_math::U8Vec2);
impl_mapper_vec!(bevy_math::I8Vec3, bevy_math::U8Vec3);
impl_mapper_vec!(bevy_math::I8Vec4, bevy_math::U8Vec4);
impl_mapper_vec!(bevy_math::I16Vec2, bevy_math::U16Vec2);
impl_mapper_vec!(bevy_math::I16Vec3, bevy_math::U16Vec3);
impl_mapper_vec!(bevy_math::I16Vec4, bevy_math::U16Vec4);
impl_mapper_vec!(bevy_math::IVec2, bevy_math::UVec2);
impl_mapper_vec!(bevy_math::IVec3, bevy_math::UVec3);
impl_mapper_vec!(bevy_math::IVec4, bevy_math::UVec4);
impl_mapper_vec!(bevy_math::I64Vec2, bevy_math::U64Vec2);
impl_mapper_vec!(bevy_math::I64Vec3, bevy_math::U64Vec3);
impl_mapper_vec!(bevy_math::I64Vec4, bevy_math::U64Vec4);

#[cfg(test)]
mod tests {
    use crate::noise::NoiseValue;

    #[test]
    fn check_mapping() {
        assert_eq!(u32::MIN, i32::MIN.map_to::<u32>());
        assert_eq!(u32::MAX / 2 + 1, 0i32.map_to::<u32>());
        assert_eq!(u32::MAX, i32::MAX.map_to::<u32>());
    }
}
