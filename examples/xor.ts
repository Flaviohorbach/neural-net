import { NeuralNetwork, Rng, binaryCrossEntropy, sparkline, xor } from "../src";

// 2 inputs → 4 hidden neurons (tanh) → 1 output (sigmoid, a probability between 0 and 1)
const rng = new Rng(7);
const network = new NeuralNetwork(2, [
  { size: 4, activation: "tanh" },
  { size: 1, activation: "sigmoid" },
], rng);

console.log("Before training (random weights):");
for (const s of xor) console.log(`  ${s.x.join(" XOR ")} → ${network.predict(s.x)[0]!.toFixed(3)}  (expected ${s.y[0]})`);

const history = network.train(xor, binaryCrossEntropy, {
  epochs: 2000,
  learningRate: 0.5,
  rng,
  onEpoch: (epoch, loss) => {
    if (epoch === 1 || epoch % 400 === 0) console.log(`epoch ${String(epoch).padStart(4)}  loss ${loss.toFixed(5)}`);
  },
});

console.log(`\nLoss curve: ${sparkline(history)}`);

console.log("\nAfter training:");
for (const s of xor) console.log(`  ${s.x.join(" XOR ")} → ${network.predict(s.x)[0]!.toFixed(3)}  (expected ${s.y[0]})`);
