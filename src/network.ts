import { activations, type Activation, type ActivationName } from "./activations";
import type { Loss } from "./losses";
import { Rng } from "./random";

export interface Sample {
  x: number[];
  y: number[];
}

export interface LayerSpec {
  size: number;
  activation: ActivationName;
}

interface Layer {
  /** weights[j][i] connects input neuron i to output neuron j. */
  weights: number[][];
  biases: number[];
  activation: Activation;
}

/** Gradients with the same shape as the layers, one entry per weight and bias. */
export interface Gradients {
  weights: number[][][];
  biases: number[][];
}

export interface TrainOptions {
  epochs: number;
  learningRate: number;
  batchSize?: number;
  rng?: Rng;
  /** Called after every epoch with the average training loss. */
  onEpoch?: (epoch: number, loss: number) => void;
}

const dot = (a: number[], b: number[]) => a.reduce((sum, v, i) => sum + v * b[i]!, 0);

/**
 * A fully connected feed-forward neural network (multilayer perceptron).
 *
 * The whole learning cycle lives here:
 *   forward()   → turns an input into a prediction
 *   gradients() → backpropagation: how much each weight contributed to the error
 *   train()     → gradient descent: move every weight a little against its gradient
 */
export class NeuralNetwork {
  readonly layers: Layer[];

  constructor(inputSize: number, specs: LayerSpec[], rng = new Rng()) {
    let fanIn = inputSize;
    this.layers = specs.map((spec) => {
      // Xavier/Glorot initialization: random weights scaled by layer size so the
      // signal neither vanishes nor explodes as it passes through the layers.
      const scale = Math.sqrt(2 / (fanIn + spec.size));
      const layer: Layer = {
        weights: Array.from({ length: spec.size }, () =>
          Array.from({ length: fanIn }, () => rng.normal() * scale),
        ),
        biases: new Array<number>(spec.size).fill(0),
        activation: activations[spec.activation],
      };
      fanIn = spec.size;
      return layer;
    });
  }

  /** Inference: run an input through the network. */
  predict(x: number[]): number[] {
    return this.forward(x).output;
  }

  /** Average loss over a dataset. */
  evaluate(data: Sample[], loss: Loss): number {
    return data.reduce((sum, s) => sum + loss.fn(this.predict(s.x), s.y), 0) / data.length;
  }

  /**
   * Forward pass. Besides the output, it keeps every layer's input and
   * pre-activation (z), because backpropagation needs them.
   */
  private forward(x: number[]) {
    const inputs: number[][] = [];
    const zs: number[][] = [];
    let a = x;
    for (const layer of this.layers) {
      inputs.push(a);
      const z = layer.weights.map((row, j) => dot(row, a) + layer.biases[j]!);
      zs.push(z);
      a = z.map(layer.activation.fn);
    }
    return { output: a, inputs, zs };
  }

  /**
   * Backpropagation for a single sample.
   *
   * `delta` is dLoss/dz for the current layer. It starts at the output and is
   * pushed backwards one layer at a time using the chain rule:
   *   delta_prev[i] = sum_j( W[j][i] * delta[j] ) * activation'(z_prev[i])
   */
  gradients(sample: Sample, loss: Loss): Gradients {
    const { output, inputs, zs } = this.forward(sample.x);
    const last = this.layers.length - 1;

    const dLoss = loss.derivative(output, sample.y);
    let delta = dLoss.map((g, j) => g * this.layers[last]!.activation.derivative(zs[last]![j]!));

    const weights: number[][][] = new Array(this.layers.length);
    const biases: number[][] = new Array(this.layers.length);

    for (let l = last; l >= 0; l--) {
      const layer = this.layers[l]!;
      const input = inputs[l]!;
      // dLoss/dW[j][i] = delta[j] * input[i]   and   dLoss/db[j] = delta[j]
      weights[l] = delta.map((d) => input.map((a) => d * a));
      biases[l] = [...delta];

      if (l > 0) {
        const prev = this.layers[l - 1]!;
        delta = input.map((_, i) => {
          const back = delta.reduce((sum, d, j) => sum + layer.weights[j]![i]! * d, 0);
          return back * prev.activation.derivative(zs[l - 1]![i]!);
        });
      }
    }

    return { weights, biases };
  }

  /**
   * Mini-batch gradient descent. Each epoch shuffles the data, splits it into
   * batches, averages the gradients of each batch and takes one step.
   * Returns the training loss after every epoch (the "loss curve").
   */
  train(data: Sample[], loss: Loss, options: TrainOptions): number[] {
    const { epochs, learningRate, batchSize = data.length, rng = new Rng(), onEpoch } = options;
    const history: number[] = [];

    for (let epoch = 1; epoch <= epochs; epoch++) {
      const shuffled = rng.shuffle(data);
      for (let start = 0; start < shuffled.length; start += batchSize) {
        this.step(shuffled.slice(start, start + batchSize), loss, learningRate);
      }
      const epochLoss = this.evaluate(data, loss);
      history.push(epochLoss);
      onEpoch?.(epoch, epochLoss);
    }

    return history;
  }

  private step(batch: Sample[], loss: Loss, learningRate: number) {
    const sum = this.zeroGradients();
    for (const sample of batch) {
      const g = this.gradients(sample, loss);
      this.layers.forEach((layer, l) => {
        layer.weights.forEach((row, j) => {
          row.forEach((_, i) => (sum.weights[l]![j]![i]! += g.weights[l]![j]![i]!));
          sum.biases[l]![j]! += g.biases[l]![j]!;
        });
      });
    }

    // Move against the gradient: downhill on the loss surface.
    const rate = learningRate / batch.length;
    this.layers.forEach((layer, l) => {
      layer.weights.forEach((row, j) => {
        row.forEach((_, i) => (row[i]! -= rate * sum.weights[l]![j]![i]!));
        layer.biases[j]! -= rate * sum.biases[l]![j]!;
      });
    });
  }

  private zeroGradients(): Gradients {
    return {
      weights: this.layers.map((layer) => layer.weights.map((row) => row.map(() => 0))),
      biases: this.layers.map((layer) => layer.biases.map(() => 0)),
    };
  }
}
