// Cover windows keep the separator, RVC and mastering working sets bounded.
export const COVER_RATE = 44100;
export const COVER_WINDOW_SECONDS = 60;
const WAV_DATA_MAX = 0xffffffff - 36;

function tag(view, offset) {
  return String.fromCharCode(...new Uint8Array(view.buffer, offset, 4));
}

// PCM WAV can be read in bounded file slices, including very long recordings.
// Compressed and uncommon WAV encodings use the browser's normal decoder.
export async function openPcmWave(file) {
  if (file.size < 44) return null;
  const header = new DataView(await file.slice(0, 12).arrayBuffer());
  const container = tag(header, 0);
  if ((container !== "RIFF" && container !== "RF64") || tag(header, 8) !== "WAVE") return null;
  let format;
  let dataSize64;
  for (let offset = 12; offset + 8 <= file.size;) {
    const chunk = new DataView(await file.slice(offset, offset + 8).arrayBuffer());
    const name = tag(chunk, 0);
    const size32 = chunk.getUint32(4, true);
    const size = name === "data" && container === "RF64" && size32 === 0xffffffff ? dataSize64 : size32;
    const start = offset + 8;
    if (!Number.isSafeInteger(size) || start + size > file.size) return null;
    if (name === "ds64" && container === "RF64" && size >= 28) {
      const sizes = new DataView(await file.slice(start, start + 28).arrayBuffer());
      dataSize64 = Number(sizes.getBigUint64(8, true));
    } else if (name === "fmt " && size >= 16) {
      format = new DataView(await file.slice(start, start + Math.min(40, size)).arrayBuffer());
    } else if (name === "data" && format) {
      let encoding = format.getUint16(0, true);
      if (encoding === 0xfffe) {
        if (format.byteLength < 40 || format.getUint16(16, true) < 22 ||
            format.getUint32(28, true) !== 0x00100000 ||
            format.getBigUint64(32, false) !== 0x800000aa00389b71n) return null;
        encoding = format.getUint32(24, true);
      }
      const channels = format.getUint16(2, true);
      const rate = format.getUint32(4, true);
      const align = format.getUint16(12, true);
      const bits = format.getUint16(14, true);
      if (channels < 1 || channels > 2 || rate < 8000 || rate > 192000 || align !== channels * bits / 8 ||
          !(encoding === 1 && [16, 24, 32].includes(bits) || encoding === 3 && bits === 32)) return null;
      const samples = Math.floor(size / align);
      const length = Math.round(samples * COVER_RATE / rate);
      const read = encoding === 3 ? (data, i) => data.getFloat32(i, true)
        : bits === 16 ? (data, i) => data.getInt16(i, true) / 32768
          : bits === 24 ? (data, i) => (data.getUint16(i, true) | data.getInt8(i + 2) << 16) / 8388608
            : (data, i) => data.getInt32(i, true) / 2147483648;
      return {
        length,
        async read(from, to) {
          const first = Math.round(from * rate / COVER_RATE);
          const last = Math.min(samples, Math.round(to * rate / COVER_RATE));
          const data = new DataView(await file.slice(start + first * align, start + last * align).arrayBuffer());
          const left = new Float32Array(last - first);
          const right = new Float32Array(last - first);
          for (let i = 0; i < left.length; i++) {
            const position = i * align;
            const l = read(data, position);
            const r = channels === 2 ? read(data, position + bits / 8) : l;
            left[i] = Number.isFinite(l) ? l : 0;
            right[i] = Number.isFinite(r) ? r : 0;
          }
          return { left, right, rate };
        },
      };
    }
    offset = start + size + (size & 1);
  }
  return null;
}

function pcm16(left, right, from, to) {
  const view = new DataView(new ArrayBuffer((to - from) * 4));
  const sample = value => {
    const limited = Math.max(-1, Math.min(1, value));
    return limited * (limited < 0 ? 0x8000 : 0x7fff);
  };
  for (let i = from, offset = 0; i < to; i++, offset += 4) {
    view.setInt16(offset, sample(left[i]), true);
    view.setInt16(offset + 2, sample(right[i]), true);
  }
  return new Blob([view.buffer]);
}

export async function renderCoverWindows(length, process, { signal, onWindow, windowSeconds = COVER_WINDOW_SECONDS } = {}) {
  if (!Number.isSafeInteger(length) || length < 1) throw new Error("Empty or invalid song");
  const core = Math.round(windowSeconds * COVER_RATE);
  const fade = Math.min(Math.round(COVER_RATE * 0.12), Math.floor(core / 2));
  const context = COVER_RATE;
  const count = Math.ceil(length / core);
  const blocks = [];
  let tail = null;
  for (let start = 0, index = 0; start < length; start += core, index++) {
    signal?.throwIfAborted();
    onWindow?.(index + 1, count);
    const next = Math.min(length, start + core);
    const end = Math.min(length, next + fade);
    const from = Math.max(0, start - context);
    const to = Math.min(length, end + context);
    const [left, right] = await process(from, to);
    signal?.throwIfAborted();
    const offset = start - from;
    if (left.length < end - from || right.length < end - from) throw new Error("Cover window is shorter than expected");
    if (tail) {
      for (let i = 0; i < tail[0].length; i++) {
        const weight = (1 - Math.cos(Math.PI * (i + 1) / (tail[0].length + 1))) / 2;
        left[offset + i] = tail[0][i] * (1 - weight) + left[offset + i] * weight;
        right[offset + i] = tail[1][i] * (1 - weight) + right[offset + i] * weight;
      }
    }
    blocks.push(pcm16(left, right, offset, next - from));
    tail = [left.slice(next - from, end - from), right.slice(next - from, end - from)];
  }
  signal?.throwIfAborted();
  const bytes = length * 4;
  const rf64 = bytes > WAV_DATA_MAX;
  const header = new DataView(new ArrayBuffer(rf64 ? 80 : 44));
  const writeTag = (offset, value) => {
    for (let i = 0; i < value.length; i++) header.setUint8(offset + i, value.charCodeAt(i));
  };
  writeTag(0, rf64 ? "RF64" : "RIFF");
  header.setUint32(4, rf64 ? 0xffffffff : 36 + bytes, true);
  writeTag(8, "WAVE");
  const fmt = rf64 ? 48 : 12;
  if (rf64) {
    writeTag(12, "ds64");
    header.setUint32(16, 28, true);
    header.setBigUint64(20, BigInt(header.byteLength - 8 + bytes), true);
    header.setBigUint64(28, BigInt(bytes), true);
    header.setBigUint64(36, BigInt(length), true);
  }
  writeTag(fmt, "fmt ");
  header.setUint32(fmt + 4, 16, true);
  header.setUint16(fmt + 8, 1, true);
  header.setUint16(fmt + 10, 2, true);
  header.setUint32(fmt + 12, COVER_RATE, true);
  header.setUint32(fmt + 16, COVER_RATE * 4, true);
  header.setUint16(fmt + 20, 4, true);
  header.setUint16(fmt + 22, 16, true);
  writeTag(fmt + 24, "data");
  header.setUint32(fmt + 28, rf64 ? 0xffffffff : bytes, true);
  return new Blob([header.buffer, ...blocks], { type: "audio/wav" });
}
