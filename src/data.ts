import type { NeuralNetwork, Sample } from "./network";
import { Rng } from "./random";

/** XOR: the classic problem a single neuron cannot solve, but two layers can. */
export const xor: Sample[] = [
  { x: [0, 0], y: [0] },
  { x: [0, 1], y: [1] },
  { x: [1, 0], y: [1] },
  { x: [1, 1], y: [0] },
];

/**
 * Random points in the square [-1, 1]². Label 1 if the point is inside a
 * circle of the given radius. A non-linear boundary that a straight line
 * cannot separate. `noise` flips a fraction of labels to mimic messy real data.
 */
export function circle(count: number, rng = new Rng(), radius = 0.6, noise = 0): Sample[] {
  return Array.from({ length: count }, () => {
    const x = [rng.uniform(-1, 1), rng.uniform(-1, 1)];
    let inside = Math.hypot(x[0]!, x[1]!) < radius;
    if (rng.next() < noise) inside = !inside;
    return { x, y: [inside ? 1 : 0] };
  });
}

/**
 * Points in [-1, 1]² labeled 1 when x and y have opposite signs: XOR spread
 * over a plane. Needs a hidden layer, like the 4-sample XOR.
 */
export function quadrants(count: number, rng = new Rng(), noise = 0): Sample[] {
  return Array.from({ length: count }, () => {
    const x = [rng.uniform(-1, 1), rng.uniform(-1, 1)];
    let positive = x[0]! * x[1]! < 0;
    if (rng.next() < noise) positive = !positive;
    return { x, y: [positive ? 1 : 0] };
  });
}

/**
 * Two interleaved spiral arms, one per class. The hardest of the three:
 * small networks fail here, which makes it a good test for layer sizes.
 * `jitter` moves points randomly around their arm.
 */
export function spiral(count: number, rng = new Rng(), jitter = 0): Sample[] {
  return Array.from({ length: count }, (_, i) => {
    const label = i % 2;
    const t = rng.uniform(0.05, 1);
    const angle = t * 3 * Math.PI + label * Math.PI;
    const radius = t * 0.9;
    const x = [
      radius * Math.cos(angle) + rng.normal() * jitter,
      radius * Math.sin(angle) + rng.normal() * jitter,
    ];
    return { x, y: [label] };
  });
}

/**
 * Splits the data so we can measure the model on samples it never trained on.
 * Good training accuracy with bad validation accuracy means overfitting.
 */
export function trainValidationSplit(data: Sample[], validationRatio: number, rng = new Rng()) {
  const shuffled = rng.shuffle(data);
  const cut = Math.round(shuffled.length * (1 - validationRatio));
  return { train: shuffled.slice(0, cut), validation: shuffled.slice(cut) };
}

/** Share of samples where the rounded output matches the label (binary classification). */
export function accuracy(network: NeuralNetwork, data: Sample[], threshold = 0.5): number {
  const hits = data.filter((s) => (network.predict(s.x)[0]! >= threshold ? 1 : 0) === s.y[0]).length;
  return hits / data.length;
}
