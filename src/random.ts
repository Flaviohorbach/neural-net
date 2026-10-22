/**
 * Seeded pseudo-random number generator (mulberry32).
 *
 * Training uses randomness in three places: initial weights, shuffling the
 * data and generating synthetic datasets. A fixed seed makes every run
 * reproducible, which is essential when you want to compare experiments.
 */
export class Rng {
  private state: number;

  constructor(seed = 42) {
    this.state = seed >>> 0;
  }

  /** Uniform number in [0, 1). */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform number in [min, max). */
  uniform(min: number, max: number): number {
    return min + (max - min) * this.next();
  }

  /** Standard normal sample (mean 0, std 1) using the Box-Muller transform. */
  normal(): number {
    const u = 1 - this.next(); // avoid log(0)
    const v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Returns a shuffled copy (Fisher-Yates). */
  shuffle<T>(items: readonly T[]): T[] {
    const copy = [...items];
    for (let i = copy.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      [copy[i], copy[j]] = [copy[j]!, copy[i]!];
    }
    return copy;
  }
}
