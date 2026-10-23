import { defineConfig } from "vite";

// The demo lives in web/ and imports the network from src/.
// Relative base so the build works under GitHub Pages' /neural-net/ path.
export default defineConfig({
  root: "web",
  base: "./",
  server: { fs: { allow: [".."] } },
  // TensorFlow.js is a ~1.3 MB chunk, but it is loaded on demand, never on first paint.
  build: { outDir: "../dist", emptyOutDir: true, chunkSizeWarningLimit: 1500 },
});
