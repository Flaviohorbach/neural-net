import { describe, expect, it } from "vitest";
import {
  NeuralNetwork,
  Rng,
  accuracy,
  binaryCrossEntropy,
  circle,
  mse,
  trainValidationSplit,
  xor,
  type Loss,
  type Sample,
} from "../src";

/**
 * Gradient check: compares backpropagation with a brute-force numerical
 * derivative, (loss(w + h) - loss(w - h)) / 2h, for every single weight.
 * If they agree, the calculus in backprop is right.
 */
function maxGradientError(network: NeuralNetwork, sample: Sample, loss: Loss): number {
  const analytic = network.gradients(sample, loss);
  const h = 1e-5;
  let maxError = 0;

  const check = (get: () => number, set: (v: number) => void, expected: number) => {
    const original = get();
    set(original + h);
    const plus = loss.fn(network.predict(sample.x), sample.y);
    set(original - h);
    const minus = loss.fn(network.predict(sample.x), sample.y);
    set(original);
    const numeric = (plus - minus) / (2 * h);
    const error = Math.abs(numeric - expected) / Math.max(1e-8, Math.abs(numeric) + Math.abs(expected));
    maxError = Math.max(maxError, error);
  };

  network.layers.forEach((layer, l) => {
    layer.weights.forEach((row, j) => {
      row.forEach((_, i) => check(() => row[i]!, (v) => (row[i] = v), analytic.weights[l]![j]![i]!));
      check(() => layer.biases[j]!, (v) => (layer.biases[j] = v), analytic.biases[l]![j]!);
    });
  });

  return maxError;
}

describe("backpropagation", () => {
  it.each([
    ["binary cross-entropy", binaryCrossEntropy],
    ["mean squared error", mse],
  ] as const)("matches numerical gradients with %s", (_, loss) => {
    const network = new NeuralNetwork(3, [
      { size: 5, activation: "tanh" },
      { size: 4, activation: "sigmoid" },
      { size: 2, activation: "sigmoid" },
    ], new Rng(1));

    const error = maxGradientError(network, { x: [0.3, -0.8, 0.5], y: [1, 0] }, loss);
    expect(error).toBeLessThan(1e-6);
  });
});

describe("training", () => {
  it("learns XOR", () => {
    const network = new NeuralNetwork(2, [
      { size: 4, activation: "tanh" },
      { size: 1, activation: "sigmoid" },
    ], new Rng(7));

    const history = network.train(xor, binaryCrossEntropy, { epochs: 2000, learningRate: 0.5, rng: new Rng(7) });

    expect(history.at(-1)!).toBeLessThan(history[0]! / 10);
    expect(accuracy(network, xor)).toBe(1);
  });

  it("generalizes to unseen data on the circle problem", () => {
    const rng = new Rng(42);
    const { train, validation } = trainValidationSplit(circle(400, rng), 0.25, rng);
    const network = new NeuralNetwork(2, [
      { size: 8, activation: "tanh" },
      { size: 1, activation: "sigmoid" },
    ], rng);

    network.train(train, binaryCrossEntropy, { epochs: 200, learningRate: 0.3, batchSize: 16, rng });

    expect(accuracy(network, validation)).toBeGreaterThan(0.9);
  });
});

describe("Rng", () => {
  it("is reproducible for the same seed", () => {
    const a = new Rng(123);
    const b = new Rng(123);
    expect(Array.from({ length: 5 }, () => a.next())).toEqual(Array.from({ length: 5 }, () => b.next()));
  });
});
