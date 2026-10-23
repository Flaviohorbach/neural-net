import { Rng, binaryCrossEntropy, circle, quadrants, spiral, trainValidationSplit, type Sample } from "../../src";
import { drawDecisionMap, drawLossChart, gridPoints } from "./draw";
import { createEngine, type Engine, type EngineConfig, type EngineName } from "./engines";

const SEED = 42;
const RESOLUTION = 48;
const MAX_EPOCHS = 1500;
const GRID = gridPoints(RESOLUTION);

const datasets: Record<string, (rng: Rng) => Sample[]> = {
  circle: (rng) => circle(300, rng, 0.6, 0.03),
  quadrants: (rng) => quadrants(300, rng, 0.03),
  spiral: (rng) => spiral(400, rng, 0.03),
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/** Validation loss over the last 50 epochs is clearly above its best value so far. */
function isOverfitting(valHistory: number[]): boolean {
  if (valHistory.length < 150) return false;
  const recent = valHistory.slice(-50);
  const best = Math.min(...valHistory);
  return recent.reduce((a, b) => a + b, 0) / recent.length > best * 1.3;
}

export function initPlayground() {
  const form = $<HTMLFormElement>("pg-controls");
  const playButton = $<HTMLButtonElement>("pg-play");
  const map = $<HTMLCanvasElement>("pg-map");
  const chart = $<HTMLCanvasElement>("pg-chart");
  const hint = $<HTMLParagraphElement>("pg-hint");
  const select = (id: string) => $<HTMLSelectElement>(id).value;

  let train: Sample[] = [];
  let validation: Sample[] = [];
  let engine: Engine | null = null;
  let running = false;
  let epoch = 0;
  let trainHistory: number[] = [];
  let valHistory: number[] = [];
  // Bumped on every reset so an epoch still in flight from the old model is ignored.
  let generation = 0;

  async function reset() {
    generation++;
    const current = generation;
    running = false;
    playButton.textContent = "Train";
    engine?.dispose();
    engine = null;

    const rng = new Rng(SEED);
    ({ train, validation } = trainValidationSplit(datasets[select("pg-dataset")]!(rng), 0.2, rng));
    epoch = 0;
    trainHistory = [];
    valHistory = [];

    const config: EngineConfig = {
      hidden: select("pg-hidden").split(",").map(Number),
      activation: select("pg-activation") as EngineConfig["activation"],
      learningRate: Number(select("pg-lr")),
      batchSize: 16,
    };
    const name = select("pg-engine") as EngineName;

    $("pg-engine-note").textContent = name === "tfjs"
      ? "Slower here on purpose: for a network this small, moving data to the GPU costs more than the math. TensorFlow.js pays off with large models."
      : "The network in src/network.ts: plain arrays and hand-written backpropagation.";
    hint.textContent = name === "tfjs" ? "Loading TensorFlow.js…" : "";
    const created = await createEngine(name, train, config, SEED);
    if (current !== generation) return created.dispose();
    engine = created;
    hint.textContent = "";
    await render();
  }

  /** Redraws everything. `recordEpoch` appends the validation loss to the curve (only after training). */
  async function render(recordEpoch = false) {
    if (!engine) return;
    const predictions = await engine.predict([...GRID, ...validation.map((s) => s.x)]);
    const valPredictions = predictions.slice(GRID.length);

    drawDecisionMap(map, predictions.slice(0, GRID.length), RESOLUTION, train, validation);

    const valLoss = validation.reduce((sum, s, i) => sum + binaryCrossEntropy.fn([valPredictions[i]!], s.y), 0) / validation.length;
    const valAcc = validation.filter((s, i) => (valPredictions[i]! >= 0.5 ? 1 : 0) === s.y[0]).length / validation.length;
    if (recordEpoch) valHistory.push(valLoss);

    $("pg-epoch").textContent = String(epoch);
    $("pg-train-loss").textContent = trainHistory.length ? trainHistory.at(-1)!.toFixed(4) : "–";
    $("pg-val-loss").textContent = valLoss.toFixed(4);
    $("pg-val-acc").textContent = `${(valAcc * 100).toFixed(1)}%`;
    drawLossChart(chart, trainHistory, valHistory);
  }

  async function loop() {
    const current = generation;
    if (!running || !engine) return;
    const loss = await engine.trainEpoch();
    if (current !== generation) return;
    epoch++;
    if (!Number.isFinite(loss)) {
      stop("The loss exploded to infinity: the learning rate is too high. Lower it and press Reset.");
      return;
    }
    trainHistory.push(loss);
    await render(true);
    if (!hint.textContent && isOverfitting(valHistory)) {
      hint.textContent =
        "Overfitting: training loss keeps falling but validation loss is rising. The network is memorizing the noisy points instead of the shape.";
    }
    if (epoch >= MAX_EPOCHS) return stop(`Stopped after ${MAX_EPOCHS} epochs.`);
    requestAnimationFrame(loop);
  }

  function stop(message = "") {
    running = false;
    playButton.textContent = "Train";
    hint.textContent = message;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!engine) return;
    running = !running;
    playButton.textContent = running ? "Pause" : "Train";
    if (running) {
      hint.textContent = "";
      requestAnimationFrame(loop);
    }
  });
  $("pg-reset").addEventListener("click", () => void reset());
  form.querySelectorAll("select").forEach((el) => el.addEventListener("change", () => void reset()));
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => void render());
  window.addEventListener("resize", () => void render());

  void reset();
}
