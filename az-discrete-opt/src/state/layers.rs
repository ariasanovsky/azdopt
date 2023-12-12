use ringbuffer::ConstGenericRingBuffer;

use crate::log::ShortForm;

#[derive(Clone, Debug)]
pub struct Layers<S, const L: usize> {
    buffer: ConstGenericRingBuffer<S, L>
}

impl<S, const L: usize> Layers<S, L> {
    pub fn buffer(&self) -> &ConstGenericRingBuffer<S, L> {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut ConstGenericRingBuffer<S, L> {
        &mut self.buffer
    }
}

impl<S: ShortForm, const L: usize> ShortForm for Layers<S, L> {
    fn short_form(&self) -> String {
        self.buffer()[0].short_form()
    }
}
