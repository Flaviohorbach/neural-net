import type { NeuralNetwork } from "./network";

const BARS = "▁▂▃▄▅▆▇█";

/** One-line chart of a series, e.g. the loss curve. */
export function sparkline(values: number[], width = 60): string {
  const step = Math.max(1, Math.floor(values.length / width));
  const sampled = values.filter((_, i) => i % step === 0);
  const min = Math.min(...sampled);
  const max = Math.max(...sampled);
  const range = max - min || 1;
  return sampled
    .map((v) => BARS[Math.round(((v - min) / range) * (BARS.length - 1))])
    .join("");
}

/**
 * Draws what the network "thinks" over the square [-1, 1]²:
 * '#' where it predicts 1, '·' where it predicts 0.
 */
export function decisionMap(network: NeuralNetwork, cols = 48, rows = 20): string {
  const lines: string[] = [];
  for (let r = 0; r < rows; r++) {
    const y = 1 - (2 * r) / (rows - 1);
    let line = "";
    for (let c = 0; c < cols; c++) {
      const x = -1 + (2 * c) / (cols - 1);
      line += network.predict([x, y])[0]! >= 0.5 ? "#" : "·";
    }
    lines.push(line);
  }
  return lines.join("\n");
}
