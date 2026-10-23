export { NeuralNetwork, type Sample, type LayerSpec, type Gradients, type TrainOptions } from "./network";
export { activations, type Activation, type ActivationName } from "./activations";
export { mse, binaryCrossEntropy, type Loss } from "./losses";
export { xor, circle, quadrants, spiral, trainValidationSplit, accuracy } from "./data";
export { sparkline, decisionMap } from "./plot";
export { Rng } from "./random";
