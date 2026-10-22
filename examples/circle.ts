import {
  NeuralNetwork,
  Rng,
  accuracy,
  binaryCrossEntropy,
  circle,
  decisionMap,
  sparkline,
  trainValidationSplit,
} from "../src";

// The full cycle: data → training → validation → inference.
const rng = new Rng(42);

// 1. Data: 600 points, 5% with flipped labels (real data is never clean).
const data = circle(600, rng, 0.6, 0.05);
const { train, validation } = trainValidationSplit(data, 0.2, rng);
console.log(`Data: ${train.length} training samples, ${validation.length} validation samples\n`);

// 2. Training
const network = new NeuralNetwork(2, [
  { size: 8, activation: "tanh" },
  { size: 8, activation: "tanh" },
  { size: 1, activation: "sigmoid" },
], rng);

const history = network.train(train, binaryCrossEntropy, {
  epochs: 300,
  learningRate: 0.3,
  batchSize: 16,
  rng,
  onEpoch: (epoch, loss) => {
    if (epoch === 1 || epoch % 50 === 0) {
      const valLoss = network.evaluate(validation, binaryCrossEntropy);
      console.log(`epoch ${String(epoch).padStart(3)}  train loss ${loss.toFixed(4)}  validation loss ${valLoss.toFixed(4)}`);
    }
  },
});
console.log(`\nLoss curve: ${sparkline(history)}`);

// 3. Validation: accuracy on data the network never saw.
console.log(`\nTraining accuracy:   ${(accuracy(network, train) * 100).toFixed(1)}%`);
console.log(`Validation accuracy: ${(accuracy(network, validation) * 100).toFixed(1)}%`);

// 4. Inference: what the network learned, drawn over the whole plane.
console.log("\nDecision map ('#' = inside the circle):\n");
console.log(decisionMap(network));

for (const point of [[0, 0], [0.5, 0.5], [0.9, -0.1]]) {
  const p = network.predict(point)[0]!;
  console.log(`\n(${point.join(", ")}) → ${(p * 100).toFixed(1)}% inside`);
}
