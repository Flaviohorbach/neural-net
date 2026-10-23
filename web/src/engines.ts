import { NeuralNetwork, Rng, binaryCrossEntropy, type ActivationName, type Sample } from "../../src";

export type EngineName = "scratch" | "tfjs";

export interface EngineConfig {
  hidden: number[];
  activation: Extract<ActivationName, "tanh" | "relu">;
  learningRate: number;
  batchSize: number;
}

/**
 * The same model behind two implementations, so the playground can swap them.
 * Both build input(2) → hidden layers → 1 sigmoid output and train with
 * plain SGD on binary cross-entropy, which keeps the comparison fair.
 */
export interface Engine {
  /** Runs one epoch and returns the training loss. */
  trainEpoch(): Promise<number>;
  /** Probability of class 1 for each point. */
  predict(points: number[][]): Promise<number[]>;
  dispose(): void;
}

export async function createEngine(name: EngineName, train: Sample[], config: EngineConfig, seed: number): Promise<Engine> {
  return name === "scratch" ? scratchEngine(train, config, seed) : tfjsEngine(train, config);
}

function scratchEngine(train: Sample[], config: EngineConfig, seed: number): Engine {
  const rng = new Rng(seed);
  const network = new NeuralNetwork(2, [
    ...config.hidden.map((size) => ({ size, activation: config.activation })),
    { size: 1, activation: "sigmoid" as const },
  ], rng);

  return {
    async trainEpoch() {
      const [loss] = network.train(train, binaryCrossEntropy, {
        epochs: 1,
        learningRate: config.learningRate,
        batchSize: config.batchSize,
        rng,
      });
      return loss!;
    },
    async predict(points) {
      return points.map((p) => network.predict(p)[0]!);
    },
    dispose() {},
  };
}

async function tfjsEngine(train: Sample[], config: EngineConfig): Promise<Engine> {
  // Loaded on demand: TensorFlow.js is ~1 MB and the default engine does not need it.
  const tf = await import("@tensorflow/tfjs");

  const model = tf.sequential();
  config.hidden.forEach((units, i) => {
    model.add(tf.layers.dense({ units, activation: config.activation, ...(i === 0 ? { inputShape: [2] } : {}) }));
  });
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  model.compile({ optimizer: tf.train.sgd(config.learningRate), loss: "binaryCrossentropy" });

  const xs = tf.tensor2d(train.map((s) => s.x));
  const ys = tf.tensor2d(train.map((s) => s.y));

  return {
    async trainEpoch() {
      const history = await model.fit(xs, ys, { epochs: 1, batchSize: config.batchSize, shuffle: true, verbose: 0 });
      return history.history.loss![0] as number;
    },
    async predict(points) {
      const output = tf.tidy(() => model.predict(tf.tensor2d(points)) as import("@tensorflow/tfjs").Tensor);
      const values = await output.data();
      output.dispose();
      return Array.from(values);
    },
    dispose() {
      xs.dispose();
      ys.dispose();
      model.dispose();
    },
  };
}
