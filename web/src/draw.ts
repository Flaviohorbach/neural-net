import type { Sample } from "../../src";

/** Reads a color token from the CSS so the canvas follows light/dark mode. */
export const cssColor = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** "#rrggbb" → [r, g, b]. The color tokens in style.css are all 6-digit hex. */
const rgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

/** Grid of points covering [-1, 1]², used to ask the network about the whole plane. */
export function gridPoints(resolution: number): number[][] {
  const points: number[][] = [];
  for (let row = 0; row < resolution; row++) {
    for (let col = 0; col < resolution; col++) {
      points.push([-1 + (2 * (col + 0.5)) / resolution, 1 - (2 * (row + 0.5)) / resolution]);
    }
  }
  return points;
}

/** Sizes a canvas backing store for crisp drawing on high-DPI screens. */
function prepare(canvas: HTMLCanvasElement) {
  const ratio = window.devicePixelRatio || 1;
  const { width, height } = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(width * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

export function drawDecisionMap(
  canvas: HTMLCanvasElement,
  predictions: number[] | null,
  resolution: number,
  train: Sample[],
  validation: Sample[],
) {
  const { ctx, width, height } = prepare(canvas);
  const c0 = cssColor("--class0");
  const c1 = cssColor("--class1");
  ctx.fillStyle = cssColor("--surface");
  ctx.fillRect(0, 0, width, height);

  // Background: one pixel per grid cell, tinted by the predicted class and stronger
  // when the network is confident, then scaled up with smoothing into a soft gradient.
  if (predictions) {
    const bg = rgb(cssColor("--surface"));
    const [rgb0, rgb1] = [rgb(c0), rgb(c1)];
    const image = new ImageData(resolution, resolution);
    predictions.forEach((p, i) => {
      const tint = p >= 0.5 ? rgb1 : rgb0;
      const strength = Math.abs(p - 0.5) * 1.6;
      for (let k = 0; k < 3; k++) image.data[i * 4 + k] = bg[k]! + (tint[k]! - bg[k]!) * strength;
      image.data[i * 4 + 3] = 255;
    });
    const small = new OffscreenCanvas(resolution, resolution);
    small.getContext("2d")!.putImageData(image, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(small, 0, 0, width, height);
  }

  const toPx = (x: number[]) => [((x[0]! + 1) / 2) * width, ((1 - x[1]!) / 2) * height] as const;

  for (const s of train) {
    const [px, py] = toPx(s.x);
    ctx.beginPath();
    ctx.arc(px, py, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = s.y[0] === 1 ? c1 : c0;
    ctx.fill();
  }

  ctx.lineWidth = 2;
  ctx.strokeStyle = cssColor("--fg");
  for (const s of validation) {
    const [px, py] = toPx(s.x);
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = s.y[0] === 1 ? c1 : c0;
    ctx.fill();
    ctx.stroke();
  }
}

export function drawLossChart(canvas: HTMLCanvasElement, train: number[], validation: number[]) {
  const { ctx, width, height } = prepare(canvas);
  ctx.fillStyle = cssColor("--surface");
  ctx.fillRect(0, 0, width, height);
  if (train.length < 2) return;

  const pad = 8;
  const max = Math.max(...train.slice(0, 5), ...validation.slice(0, 5), 0.1);
  const x = (i: number) => pad + (i / (train.length - 1)) * (width - pad * 2);
  const y = (v: number) => height - pad - (Math.min(v, max) / max) * (height - pad * 2);

  ctx.strokeStyle = cssColor("--line");
  ctx.lineWidth = 1;
  for (const frac of [0.25, 0.5, 0.75]) {
    ctx.beginPath();
    ctx.moveTo(pad, y(max * frac));
    ctx.lineTo(width - pad, y(max * frac));
    ctx.stroke();
  }

  const line = (series: number[], color: string) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    series.forEach((v, i) => (i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v))));
    ctx.stroke();
  };
  line(validation, cssColor("--val-line"));
  line(train, cssColor("--accent"));
}
