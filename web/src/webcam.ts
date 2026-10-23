import type * as TF from "@tensorflow/tfjs";
import type { MobileNet } from "@tensorflow-models/mobilenet";

const CLASS_NAMES = ["Thumbs up", "Thumbs down", "Nothing"];
const MIN_EXAMPLES = 10;
const CAPTURE_INTERVAL_MS = 100;

interface ClassRow {
  name: HTMLInputElement;
  button: HTMLButtonElement;
  bar: HTMLSpanElement;
  label: HTMLElement;
  /** Embeddings captured for this class: one 1D tensor per frame. */
  examples: TF.Tensor1D[];
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/**
 * Transfer learning in three steps:
 *  1. MobileNet (pre-trained on ImageNet) turns a frame into an embedding vector.
 *  2. We collect embeddings for each class while the user holds a button.
 *  3. A small dense network learns to map embeddings → classes, in seconds.
 */
export function initWebcam() {
  const video = $<HTMLVideoElement>("cam-video");
  const status = $<HTMLParagraphElement>("cam-status");
  const startButton = $<HTMLButtonElement>("cam-start");
  const trainButton = $<HTMLButtonElement>("cam-train");
  const trainStatus = $<HTMLParagraphElement>("cam-train-status");

  let tf: typeof TF;
  let mobilenet: MobileNet;
  let head: TF.Sequential | null = null;
  let predicting = false;

  const rows: ClassRow[] = CLASS_NAMES.map((name) => {
    const el = document.createElement("div");
    el.className = "cam-class";
    el.innerHTML = `
      <input aria-label="Class name" value="${name}" />
      <button type="button" disabled>Hold to record</button>
      <div class="bar"><span></span><b>0 examples</b></div>`;
    $("cam-classes").append(el);
    return {
      name: el.querySelector("input")!,
      button: el.querySelector("button")!,
      bar: el.querySelector(".bar span")!,
      label: el.querySelector(".bar b")!,
      examples: [],
    };
  });

  /** One frame → MobileNet embedding (the layer before ImageNet's 1,000 classes). */
  const embed = () => tf.tidy(() => (mobilenet.infer(video, true) as TF.Tensor2D).squeeze<TF.Tensor1D>());

  startButton.addEventListener("click", async () => {
    startButton.disabled = true;
    try {
      status.textContent = "Asking for camera permission…";
      video.srcObject = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 224, height: 224 },
        audio: false,
      });
      await video.play();

      status.textContent = "Loading MobileNet (about 5 MB, only the first time)…";
      tf = await import("@tensorflow/tfjs");
      const { load } = await import("@tensorflow-models/mobilenet");
      mobilenet = await load({ version: 2, alpha: 0.5 });
      embed().dispose(); // warm-up: the first inference compiles the GPU shaders

      status.textContent = "Ready. Hold a button to record examples of each class.";
      startButton.hidden = true;
      rows.forEach((row) => (row.button.disabled = false));
    } catch (error) {
      startButton.disabled = false;
      status.textContent =
        error instanceof DOMException && error.name === "NotAllowedError"
          ? "Camera permission was denied. Allow it in the browser and try again."
          : `Could not start: ${(error as Error).message}`;
    }
  });

  rows.forEach((row) => {
    let timer: number | undefined;
    const startRecording = (event: PointerEvent) => {
      row.button.setPointerCapture(event.pointerId);
      row.button.classList.add("recording");
      timer = window.setInterval(() => {
        row.examples.push(embed());
        updateCounts();
      }, CAPTURE_INTERVAL_MS);
    };
    const stopRecording = () => {
      window.clearInterval(timer);
      row.button.classList.remove("recording");
    };
    row.button.addEventListener("pointerdown", startRecording);
    row.button.addEventListener("pointerup", stopRecording);
    row.button.addEventListener("pointercancel", stopRecording);
  });

  function updateCounts() {
    rows.forEach((row) => {
      if (!predicting) row.label.textContent = `${row.examples.length} examples`;
    });
    const ready = rows.filter((row) => row.examples.length >= MIN_EXAMPLES).length >= 2;
    trainButton.disabled = !ready;
  }

  trainButton.addEventListener("click", async () => {
    const used = rows.filter((row) => row.examples.length > 0);
    trainButton.disabled = true;
    predicting = false;
    head?.dispose();

    const xs = tf.stack(used.flatMap((row) => row.examples));
    const labels = used.flatMap((row, classIndex) => row.examples.map(() => classIndex));
    const ys = tf.oneHot(tf.tensor1d(labels, "int32"), used.length);

    head = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [xs.shape[1]!], units: 64, activation: "relu" }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: used.length, activation: "softmax" }),
      ],
    });
    head.compile({ optimizer: tf.train.adam(0.001), loss: "categoricalCrossentropy", metrics: ["accuracy"] });

    await head.fit(xs, ys, {
      epochs: 25,
      batchSize: 16,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          trainStatus.textContent = `Training… epoch ${epoch + 1}/25, accuracy ${((logs?.acc ?? 0) * 100).toFixed(0)}%`;
        },
      },
    });
    xs.dispose();
    ys.dispose();

    trainStatus.textContent = "Trained. Show something to the camera. Record more examples and retrain to improve it.";
    trainButton.disabled = false;
    trainButton.textContent = "Retrain";
    predicting = true;
    predictLoop(used);
  });

  async function predictLoop(used: ClassRow[]) {
    while (predicting && head) {
      const probabilities = tf.tidy(() => (head!.predict(embed().expandDims(0)) as TF.Tensor2D).squeeze<TF.Tensor1D>());
      const values = await probabilities.data();
      probabilities.dispose();
      used.forEach((row, i) => {
        const pct = values[i]! * 100;
        row.bar.style.width = `${pct}%`;
        row.label.textContent = `${row.name.value || "Class"} ${pct.toFixed(0)}%`;
      });
      await tf.nextFrame();
    }
  }
}
