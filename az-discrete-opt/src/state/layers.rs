use ringbuffer::{ConstGenericRingBuffer, RingBuffer};

use crate::log::ShortForm;

#[derive(Clone, Debug)]
pub struct Layers<S, const L: usize> {
    buffer: ConstGenericRingBuffer<S, L>
}

impl<S, const L: usize> Layers<S, L> {
    pub fn new(s: S) -> Self {
        Self {
            buffer: ConstGenericRingBuffer::from_iter(core::iter::once(s)),
        }
    }

    pub fn back(&self) -> &S {
        unsafe { self.buffer.back().unwrap_unchecked() }
    }

    pub fn push(&mut self, s: S) {
        self.buffer.push(s);
    }

    pub fn push_op(&mut self, op: impl Fn(&S) -> S) {
        self.push(op(self.back()))
    }

    pub fn buffer(&self) -> &ConstGenericRingBuffer<S, L> {
        &self.buffer
    }
}

impl<S: ShortForm, const L: usize> ShortForm for Layers<S, L> {
    fn short_form(&self) -> String {
        self.back().short_form()
    }
}


#[cfg(test)]
mod tests {
    use ringbuffer::RingBuffer;
    use super::Layers;
    
    #[test]
    fn layer_of_3_integers_only_keeps_the_last_3() {
        let mut layers: Layers<i32, 3> = Layers::new(0);
        layers.push(1);
        layers.push(2);
        layers.push(3);
        layers.push(4);
        layers.push(5);
        assert_eq!(layers.buffer().iter().copied().collect::<Vec<_>>(), vec![3, 4, 5]);
    }
}