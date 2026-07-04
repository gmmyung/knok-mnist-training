pub fn step(params: &mut [f32], grads: &[f32], learning_rate: f32) {
    assert_eq!(params.len(), grads.len());
    for (param, grad) in params.iter_mut().zip(grads) {
        *param -= learning_rate * grad;
    }
}

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let other = (self.next_u64() as usize) % (index + 1);
            values.swap(index, other);
        }
    }

    pub fn index(&mut self, upper: usize) -> usize {
        assert!(upper > 0);
        (self.next_u64() as usize) % upper
    }

    fn next_f32(&mut self) -> f32 {
        let value = self.next_u64() >> 40;
        value as f32 / (1_u32 << 24) as f32
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }
}
