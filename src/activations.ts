/**
 * Activation functions add non-linearity. Without them, any stack of layers
 * collapses into a single linear function and the network could never learn
 * something like XOR.
 *
 * `derivative` receives the pre-activation value `z` (the weighted sum), which
 * is what backpropagation needs to apply the chain rule.
 */
export interface Activation {
  fn: (z: number) => number;
  derivative: (z: number) => number;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export const activations = {
  /** Squashes to (0, 1). Good for the output of a yes/no classifier. */
  sigmoid: {
    fn: sigmoid,
    derivative: (z) => {
      const s = sigmoid(z);
      return s * (1 - s);
    },
  },
  /** Squashes to (-1, 1). Zero-centered, usually trains hidden layers faster than sigmoid. */
  tanh: {
    fn: Math.tanh,
    derivative: (z) => 1 - Math.tanh(z) ** 2,
  },
  /** max(0, z). The default in modern deep networks: cheap and does not saturate for z > 0. */
  relu: {
    fn: (z) => Math.max(0, z),
    derivative: (z) => (z > 0 ? 1 : 0),
  },
  /** Identity. Used for regression outputs. */
  linear: {
    fn: (z) => z,
    derivative: () => 1,
  },
} satisfies Record<string, Activation>;

export type ActivationName = keyof typeof activations;
