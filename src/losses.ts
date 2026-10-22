/**
 * A loss function turns "how wrong is the prediction" into a single number.
 * Training is nothing more than nudging the weights to make this number smaller.
 *
 * `derivative` returns dLoss/dPrediction for each output, the starting point
 * of backpropagation.
 */
export interface Loss {
  fn: (predicted: number[], expected: number[]) => number;
  derivative: (predicted: number[], expected: number[]) => number[];
}

// Keeps log() and divisions away from 0 and 1, where they explode.
const EPSILON = 1e-12;
const clip = (p: number) => Math.min(1 - EPSILON, Math.max(EPSILON, p));

/** Mean squared error. The classic choice for regression. */
export const mse: Loss = {
  fn: (predicted, expected) =>
    predicted.reduce((sum, p, i) => sum + (p - expected[i]!) ** 2, 0) / predicted.length,
  derivative: (predicted, expected) =>
    predicted.map((p, i) => (2 * (p - expected[i]!)) / predicted.length),
};

/**
 * Binary cross-entropy. The right choice for yes/no outputs coming from a sigmoid:
 * it punishes confident wrong answers much harder than MSE does.
 */
export const binaryCrossEntropy: Loss = {
  fn: (predicted, expected) =>
    -predicted.reduce((sum, raw, i) => {
      const p = clip(raw);
      const y = expected[i]!;
      return sum + y * Math.log(p) + (1 - y) * Math.log(1 - p);
    }, 0) / predicted.length,
  derivative: (predicted, expected) =>
    predicted.map((raw, i) => {
      const p = clip(raw);
      return (p - expected[i]!) / (p * (1 - p)) / predicted.length;
    }),
};
