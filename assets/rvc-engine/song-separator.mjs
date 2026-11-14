// On-device song separation for the offline cover fallback.
//
// Runs a UVR MDX-Net vocal model (Kim_Vocal_2, MIT) through onnxruntime-web.
// The DSP below (periodic Hann STFT, n_fft 7680, hop 1024, 3072 bins, 256
// frames, trim n_fft/2, compensation 1.009) is a line-for-line port of the
// Python reference used to validate it (docs/rvc-offline-fallback.md), so the
// browser produces the same stems as the reference within float32 rounding.
// Nothing here talks to the network: the caller passes model bytes in.

export const MDX_CONFIG = Object.freeze({
  nFft: 7680,
  hop: 1024,
  dimF: 3072,
  dimT: 256,
  compensate: 1.009,
  sampleRate: 44100,
});

// ---------------------------------------------------------------- FFT ----
// Mixed-radix (2/3/5) complex FFT with a real-input wrapper. 7680 is not a
// power of two, so the usual radix-2 code cannot be used.
function factorize(n) {
  const out = [];
  for (const p of [4, 2, 3, 5]) {
    while (n % p === 0) {
      out.push(p);
      n /= p;
    }
  }
  if (n !== 1) throw new Error(`FFT size has unsupported factor ${n}`);
  return out;
}

export function createComplexFFT(n) {
  const factors = factorize(n);
  const cos = new Float64Array(n);
  const sin = new Float64Array(n);
  for (let k = 0; k < n; k += 1) {
    cos[k] = Math.cos((2 * Math.PI * k) / n);
    sin[k] = -Math.sin((2 * Math.PI * k) / n);
  }
  const tRe = new Float64Array(8);
  const tIm = new Float64Array(8);

  const inputOrder = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let remaining = i;
    let size = n;
    let stride = 1;
    for (const radix of factors) {
      size /= radix;
      inputOrder[i] += Math.floor(remaining / size) * stride;
      remaining %= size;
      stride *= radix;
    }
  }
  // Each stage combines disjoint blocks. Keep the original butterfly
  // arithmetic and summation order while removing per-frame recursion.
  const stages = [];
  let size = 1;
  for (const p of [...factors].reverse()) {
    const m = size;
    size *= p;
    const roots = Uint32Array.from({ length: p * p }, (_, i) =>
      ((i % p) * Math.floor(i / p) * (n / p)) % n);
    stages.push({ p, m, size, step: n / size, roots });
  }

  return {
    size: n,
    forward(inRe, inIm, outRe, outIm) {
      for (let i = 0; i < n; i++) {
        const source = inputOrder[i];
        outRe[i] = inRe[source];
        outIm[i] = inIm[source];
      }
      for (const { p, m, size, step, roots } of stages) {
        for (let block = 0; block < n; block += size) {
          for (let k = 0; k < m; k++) {
            const base = block + k;
            const phase = k * step;
            for (let q = 0; q < p; q++) {
              const idx = base + q * m;
              const tw = q * phase;
              const a = outRe[idx], b = outIm[idx];
              tRe[q] = a * cos[tw] - b * sin[tw];
              tIm[q] = a * sin[tw] + b * cos[tw];
            }
            for (let u = 0; u < p; u++) {
              let sr = 0, si = 0;
              for (let q = 0; q < p; q++) {
                const tw = roots[u * p + q];
                sr += tRe[q] * cos[tw] - tIm[q] * sin[tw];
                si += tRe[q] * sin[tw] + tIm[q] * cos[tw];
              }
              outRe[base + u * m] = sr;
              outIm[base + u * m] = si;
            }
          }
        }
      }
    },
  };
}

// Real FFT of even length N via one complex FFT of N/2.
export function createRealFFT(N) {
  const M = N / 2;
  const cfft = createComplexFFT(M);
  const zRe = new Float64Array(M);
  const zIm = new Float64Array(M);
  const ZRe = new Float64Array(M);
  const ZIm = new Float64Array(M);
  const wc = new Float64Array(M + 1);
  const ws = new Float64Array(M + 1);
  for (let k = 0; k <= M; k += 1) {
    wc[k] = Math.cos((2 * Math.PI * k) / N);
    ws[k] = Math.sin((2 * Math.PI * k) / N);
  }
  return {
    // x: length N (Float64Array). Writes bins [0, nBins) into re/im.
    forward(x, re, im, nBins) {
      for (let i = 0; i < M; i += 1) {
        zRe[i] = x[2 * i];
        zIm[i] = x[2 * i + 1];
      }
      cfft.forward(zRe, zIm, ZRe, ZIm);
      for (let k = 0; k < nBins; k += 1) {
        const k1 = k % M;
        const k2 = (M - k) % M;
        const ar = ZRe[k1];
        const ai = ZIm[k1];
        const br = ZRe[k2];
        const bi = -ZIm[k2];
        const er = 0.5 * (ar + br);
        const ei = 0.5 * (ai + bi);
        const dr = 0.5 * (ar - br);
        const di = 0.5 * (ai - bi);
        // odd = -i * e^{-2πik/N} * d
        const c = wc[k];
        const s = ws[k];
        const odR = di * c - dr * s;
        const odI = -dr * c - di * s;
        re[k] = er + odR;
        im[k] = ei + odI;
      }
    },
    // re/im: bins [0, nBins), bins >= nBins are zero. Writes N samples to out.
    inverse(re, im, nBins, out) {
      for (let k = 0; k < M; k += 1) {
        const xr = k < nBins ? re[k] : 0;
        const xi = k < nBins ? im[k] : 0;
        const yr = M - k < nBins ? re[M - k] : 0;
        const yi = -(M - k < nBins ? im[M - k] : 0);
        const fer = 0.5 * (xr + yr);
        const fei = 0.5 * (xi + yi);
        const dr = 0.5 * (xr - yr);
        const di = 0.5 * (xi - yi);
        const c = wc[k];
        const s = ws[k];
        const for_ = dr * c - di * s; // d * e^{+2πik/N}
        const foi = dr * s + di * c;
        // Z = Fe + i*Fo, inverse via conj(FFT(conj(Z)))/M
        zRe[k] = fer - foi;
        zIm[k] = -(fei + for_);
      }
      cfft.forward(zRe, zIm, ZRe, ZIm);
      for (let i = 0; i < M; i += 1) {
        out[2 * i] = ZRe[i] / M;
        out[2 * i + 1] = -ZIm[i] / M;
      }
    },
  };
}

// --------------------------------------------------------------- STFT ----
export function createMdxDsp(config = MDX_CONFIG) {
  const { nFft, hop, dimF, dimT } = config;
  const chunk = hop * (dimT - 1);
  const half = nFft / 2;
  const win = new Float64Array(nFft);
  for (let i = 0; i < nFft; i += 1) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nFft);
  const rfft = createRealFFT(nFft);
  const frame = new Float64Array(nFft);
  const re = new Float64Array(dimF);
  const im = new Float64Array(dimF);
  const padded = new Float64Array(chunk + nFft);
  const olaLen = nFft + hop * (dimT - 1);
  const ola = new Float64Array(olaLen);
  const wsum = new Float64Array(olaLen);
  for (let t = 0; t < dimT; t += 1) {
    for (let i = 0; i < nFft; i += 1) wsum[t * hop + i] += win[i] * win[i];
  }

  // channels: array of Float32Array/Float64Array, each `chunk` long.
  // Returns Float32Array [1, 2C, dimF, dimT] in [L re, L im, R re, R im] order.
  function stft(channels) {
    const C = channels.length;
    const spec = new Float32Array(2 * C * dimF * dimT);
    for (let c = 0; c < C; c += 1) {
      const x = channels[c];
      // center=True with reflect padding
      for (let i = 0; i < chunk; i += 1) padded[half + i] = x[i];
      for (let i = 1; i <= half; i += 1) {
        padded[half - i] = x[i];
        padded[half + chunk - 1 + i] = x[chunk - 1 - i];
      }
      const baseRe = 2 * c * dimF * dimT;
      const baseIm = (2 * c + 1) * dimF * dimT;
      for (let t = 0; t < dimT; t += 1) {
        const off = t * hop;
        for (let i = 0; i < nFft; i += 1) frame[i] = padded[off + i] * win[i];
        rfft.forward(frame, re, im, dimF);
        for (let f = 0; f < dimF; f += 1) {
          spec[baseRe + f * dimT + t] = re[f];
          spec[baseIm + f * dimT + t] = im[f];
        }
      }
    }
    return spec;
  }

  // spec: Float32Array [2C, dimF, dimT]; returns C Float64Arrays of `chunk`.
  function istft(spec, C) {
    const outs = [];
    for (let c = 0; c < C; c += 1) {
      ola.fill(0);
      const baseRe = 2 * c * dimF * dimT;
      const baseIm = (2 * c + 1) * dimF * dimT;
      for (let t = 0; t < dimT; t += 1) {
        for (let f = 0; f < dimF; f += 1) {
          re[f] = spec[baseRe + f * dimT + t];
          im[f] = spec[baseIm + f * dimT + t];
        }
        rfft.inverse(re, im, dimF, frame);
        const off = t * hop;
        for (let i = 0; i < nFft; i += 1) ola[off + i] += frame[i] * win[i];
      }
      const y = new Float64Array(chunk);
      for (let i = 0; i < chunk; i += 1) {
        const w = wsum[half + i];
        y[i] = ola[half + i] / (w > 1e-8 ? w : 1);
      }
      outs.push(y);
    }
    return outs;
  }

  return { stft, istft, chunk, trim: half, gen: chunk - nFft };
}

// ---------------------------------------------------------- separation ----
// runModel(Float32Array spec, dims) -> Promise<Float32Array> (same shape).
// left/right: Float32Array at 44.1 kHz. Returns { vocals: [L, R], instrumental: [L, R] }.
// Memory-lean: no padded copy of the song is made (phones). With
// `inPlaceInstrumental` the input arrays are overwritten with the
// instrumental (mix - vocals) instead of allocating two more tracks.
export async function separateStereo(left, right, runModel, {
  config = MDX_CONFIG, onProgress, shouldCancel, inPlaceInstrumental = false, overlapSeconds = 0,
} = {}) {
  const dsp = createMdxDsp(config);
  const { chunk, trim, gen } = dsp;
  const n = left.length;
  const R = right || left;
  // Virtual padded signal: TRIM zeros, the song, then zeros.
  const segL = new Float64Array(chunk);
  const segR = new Float64Array(chunk);
  const fill = (dst, srcArr, start) => {
    const from = start - trim;
    const begin = Math.max(0, from);
    const end = Math.min(n, from + chunk);
    dst.fill(0);
    dst.set(srcArr.subarray(begin, end), begin - from);
  };
  const vocals = [new Float32Array(n), new Float32Array(n)];
  const overlap = Math.max(0, Math.min(gen - 1, Math.round(overlapSeconds * config.sampleRate)));
  const stride = gen - overlap;
  const steps = Math.max(1, 1 + Math.ceil((n - gen) / stride));
  const dims = [1, 4, config.dimF, config.dimT];
  let written = 0;
  for (let step = 0; step < steps; step += 1) {
    const i = step * stride;
    if (shouldCancel?.()) throw new Error("SEPARATION_CANCELLED");
    fill(segL, left, i);
    fill(segR, R, i);
    const seg = [segL, segR];
    const spec = dsp.stft(seg);
    const pred = await runModel(spec, dims);
    const [yl, yr] = dsp.istft(pred, 2);
    const take = Math.min(gen, n - i);
    const shared = Math.max(0, written - i);
    for (let k = 0; k < take; k += 1) {
      const blend = k < shared ? 0.5 - 0.5 * Math.cos(Math.PI * (k + 1) / (shared + 1)) : 1;
      vocals[0][i + k] = vocals[0][i + k] * (1 - blend) + yl[trim + k] * config.compensate * blend;
      vocals[1][i + k] = vocals[1][i + k] * (1 - blend) + yr[trim + k] * config.compensate * blend;
    }
    written = i + take;
    onProgress?.(step + 1, steps);
  }
  const sameInput = !right || right === left;
  const instrumental = inPlaceInstrumental && !sameInput
    ? [left, right]
    : [new Float32Array(n), new Float32Array(n)];
  for (let k = 0; k < n; k += 1) {
    const l = left[k];
    const r = R[k];
    instrumental[0][k] = l - vocals[0][k];
    instrumental[1][k] = r - vocals[1][k];
  }
  return { vocals, instrumental };
}
