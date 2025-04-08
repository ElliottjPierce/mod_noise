//! Contains noise types for converting one [`NoiseValue`] to another.

use core::marker::PhantomData;

use super::{CorolatedNoiseType, DirectNoise, Noise, NoiseValue};

/// A [`DirectNoise`] that converts one [`NoiseValue`] to another.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Adapter<T>(pub PhantomData<T>);

impl<T> Default for Adapter<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: Send + Sync> Noise for Adapter<T> {}

macro_rules! impl_adapter {
    ($l:ident->$o:ident, $($p:ident->$t:ident),*) => {

        #[expect(clippy::allow_attributes, reason = "Only needed for one macro type")]
        #[allow(unused_parens, reason = "Makes the macro easier.")]
        impl<I: NoiseValue, $($t: CorolatedNoiseType<$p>,)* $o: CorolatedNoiseType<$l>> DirectNoise<I> for Adapter<( $($t,)* $o )> {
            type Output = $o;

            #[inline]
            fn raw_sample(&self, input: I) -> Self::Output {
                input
                    $(.map_to::<$t>())*
                    .map_to::<$o>()
            }
        }

    };
}

impl_adapter!(I->O,);
impl_adapter!(T0->O, I->T0);
impl_adapter!(T1->O, I->T0, T0->T1);
impl_adapter!(T2->O, I->T0, T0->T1, T1->T2);
impl_adapter!(T3->O, I->T0, T0->T1, T1->T2, T2->T3);
impl_adapter!(T4->O, I->T0, T0->T1, T1->T2, T2->T3, T3->T4);
impl_adapter!(T5->O, I->T0, T0->T1, T1->T2, T2->T3, T3->T4, T4->T5);
impl_adapter!(T6->O, I->T0, T0->T1, T1->T2, T2->T3, T3->T4, T4->T5, T5->T6);
impl_adapter!(T7->O, I->T0, T0->T1, T1->T2, T2->T3, T3->T4, T4->T5, T5->T6, T6->T7);
impl_adapter!(T8->O, I->T0, T0->T1, T1->T2, T2->T3, T3->T4, T4->T5, T5->T6, T6->T7, T7->T8);
impl_adapter!(T9->O, I->T0, T0->T1, T1->T2, T2->T3, T3->T4, T4->T5, T5->T6, T6->T7, T7->T8, T8->T9);
