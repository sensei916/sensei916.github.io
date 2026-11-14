import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync, statSync } from "node:fs";
import { createHash } from "node:crypto";
import { COVER_RATE, openPcmWave, renderCoverWindows } from "../assets/rvc-engine/device-cover.mjs";
import {
  createRealFFT,
  createMdxDsp,
  separateStereo,
  MDX_CONFIG,
} from "../assets/rvc-engine/song-separator.mjs";

const root = new URL("../", import.meta.url);
const read = (path) => readFileSync(new URL(path, root), "utf8");

test("mixed-radix real FFT matches a direct DFT for n_fft 7680", () => {
  const N = MDX_CONFIG.nFft;
  const rfft = createRealFFT(N);
  const x = new Float64Array(N);
  for (let i = 0; i < N; i += 1) x[i] = Math.sin(i * 0.013) + 0.3 * Math.cos(i * 0.71) + ((i * 7919) % 101) / 101 - 0.5;
  const re = new Float64Array(N / 2 + 1);
  const im = new Float64Array(N / 2 + 1);
  rfft.forward(x, re, im, N / 2 + 1);
  for (const k of [0, 1, 7, 1234, 3071, 3840]) {
    let sr = 0;
    let si = 0;
    for (let n = 0; n < N; n += 1) {
      sr += x[n] * Math.cos((2 * Math.PI * k * n) / N);
      si -= x[n] * Math.sin((2 * Math.PI * k * n) / N);
    }
    assert.ok(Math.abs(sr - re[k]) < 1e-6 && Math.abs(si - im[k]) < 1e-6, `bin ${k}`);
  }
  const back = new Float64Array(N);
  rfft.inverse(re, im, N / 2 + 1, back);
  for (let i = 0; i < N; i += 97) assert.ok(Math.abs(back[i] - x[i]) < 1e-9);
});

test("separation loop reconstructs the mix through an identity model", async () => {
  const n = 44100; // one second -> a single MDX chunk
  const left = new Float32Array(n);
  const right = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    left[i] = 0.4 * Math.sin((2 * Math.PI * 220 * i) / 44100);
    right[i] = 0.3 * Math.sin((2 * Math.PI * 330 * i) / 44100);
  }
  const mixL = left.slice();
  let calls = 0;
  const { vocals, instrumental } = await separateStereo(left, right, async (spec, dims) => {
    calls += 1;
    assert.deepEqual(dims, [1, 4, MDX_CONFIG.dimF, MDX_CONFIG.dimT]);
    return spec;
  });
  assert.equal(calls, 1);
  assert.equal(vocals[0].length, n);
  for (let i = 1000; i < n - 1000; i += 101) {
    assert.ok(Math.abs(vocals[0][i] - mixL[i] * MDX_CONFIG.compensate) < 1e-4);
    assert.ok(Math.abs(instrumental[0][i] + vocals[0][i] - mixL[i]) < 1e-6);
  }
  const dsp = createMdxDsp();
  assert.equal(dsp.chunk, 261120);
  assert.equal(dsp.gen, 253440);
});

test("separator model chunks are published with integrity hashes", () => {
  const manifest = JSON.parse(read("assets/rvc-models.json"));
  const separator = manifest.baseModels.songSeparator;
  assert.ok(separator, "baseModels.songSeparator");
  assert.equal(separator.chunks.length, separator.chunkSha256.length);
  assert.match(separator.sha256, /^[0-9a-f]{64}$/u);
  assert.equal(separator.license, "MIT");
  const whole = createHash("sha256");
  let present = 0;
  separator.chunks.forEach((path, index) => {
    const url = new URL(path, root);
    if (!existsSync(url)) return; // sparse checkouts may omit models/
    present += 1;
    const bytes = readFileSync(url);
    assert.ok(statSync(url).size < 20 * 1024 * 1024, "chunk under jsDelivr limit");
    assert.equal(createHash("sha256").update(bytes).digest("hex"), separator.chunkSha256[index]);
    whole.update(bytes);
  });
  if (present === separator.chunks.length) assert.equal(whole.digest("hex"), separator.sha256);
});

test("song covers can be selected directly on-device, including files past cloud limits", () => {
  const source = read("assets/rvc.js");
  assert.match(source, /async function runDeviceSongCover\(/u);
  assert.match(source, /OFFLINE_SONG_FALLBACK_KEY = "rvcOfflineSongFallback"/u);
  assert.match(source, /window\.POSTPREP_RVC_OFFLINE_SONG === false/u);
  const routing = source.slice(source.indexOf("async function runRvcInference()"));
  const probe = routing.indexOf('state.audioMode === "song" && state.engineReady !== true && canRunDeviceSongCover');
  const cloud = routing.indexOf("runOfficialRvcInference({ allowDeviceFallback");
  assert.ok(probe > 0 && cloud > probe, "offline probe precedes the cloud call");
  assert.match(routing, /if \(cloudReady !== true\) return runDeviceSongCover\(\);/u);
  assert.match(routing, /cloudResult\?\.fallback && state\.audioMode === "song" && canRunDeviceSongCover/u);
  assert.match(routing, /if \(state\.inferenceMode === "local"\) \{\s+if \(state\.audioMode === "song"\) return runDeviceSongCover\(\);/u);
  assert.doesNotMatch(source, /DEVICE_SONG_MAX_AUDIO_SECONDS/u);
});

test("cover windows keep exact stereo length and crossfade overlapping samples", async () => {
  const frames = Math.round(COVER_RATE * .35);
  const windows = [];
  const wav = await renderCoverWindows(frames, (from, to) => {
    const level = .2 * windows.length;
    return [new Float32Array(to - from).fill(level), new Float32Array(to - from).fill(-level)];
  }, { windowSeconds: .1, onWindow: (index, total) => windows.push([index, total]) });
  assert.deepEqual(windows, [[1, 4], [2, 4], [3, 4], [4, 4]]);
  assert.equal(wav.size, 44 + frames * 4);
  const reader = await openPcmWave(wav);
  assert.equal(reader.length, frames);
  const join = COVER_RATE / 10;
  const { left, right } = await reader.read(join - 1, join + 3000);
  assert.ok(Math.abs(left[0] - left[1]) < 1e-4);
  assert.ok(Math.abs(left[2500] - .4) < 1e-4);
  assert.ok(Math.abs(left[1200] + right[1200]) < 1e-4);
});

test("overlapped MDX windows maintain length and the sum of the two stems", async () => {
  const config = { ...MDX_CONFIG, nFft: 32, hop: 8, dimF: 17, dimT: 9, sampleRate: 100 };
  const left = Float32Array.from({ length: 128 }, (_, i) => .5 * Math.sin(i / 5));
  const right = Float32Array.from(left, value => value * .6);
  let calls = 0;
  const { vocals, instrumental } = await separateStereo(left, right, async spec => {
    calls++;
    return spec;
  }, { config, overlapSeconds: .08 });
  assert.ok(calls > 2);
  assert.equal(vocals[0].length, left.length);
  for (let i = 0; i < left.length; i++) {
    assert.ok(Math.abs(vocals[0][i] - left[i] * config.compensate) < 1e-5);
    assert.ok(Math.abs(vocals[1][i] + instrumental[1][i] - right[i]) < 1e-6);
  }
});

test("separator worker uses the bundled onnxruntime build and verifies WebGPU output", () => {
  const worker = read("assets/rvc-engine/separator.worker.js");
  assert.match(worker, /from "\.\/ort126\/ort\.min\.mjs"/u);
  assert.match(worker, /relativeError\(out, ref\)/u);
  assert.match(worker, /inPlaceInstrumental: true/u);
});
