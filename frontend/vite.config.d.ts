/**
 * Development proxy: the browser only ever talks to the Vite origin, and Vite
 * forwards /api to the Flask backend. That keeps the frontend free of any absolute
 * host, so the same build works locally, behind the preview proxy and in production
 * where Flask serves the built assets itself.
 */
declare const _default: import("vite").UserConfig;
export default _default;
