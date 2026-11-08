import { useState } from 'react'
import { useGpu } from '../../lib/gpu'
import { decodeAudio, useCaseMedia } from '../../lib/media'
import { usePlayback } from '../../lib/playback'
import { mmss } from '../../lib/format'
import { Icon } from '../ui/ui'

const MODELS = [
  { id: 'onnx-community/whisper-tiny', label: 'Whisper tiny (≈40 MB, fastest)' },
  { id: 'onnx-community/whisper-base', label: 'Whisper base (≈80 MB, better)' },
]

interface Chunk {
  timestamp: [number, number | null]
  text: string
}

/** Resample mono samples to 16 kHz (what Whisper expects) with the browser's audio engine. */
async function to16k(samples: Float32Array, rate: number): Promise<Float32Array> {
  if (rate === 16000) return samples
  const length = Math.ceil((samples.length * 16000) / rate)
  const ctx = new OfflineAudioContext(1, length, 16000)
  const buffer = ctx.createBuffer(1, samples.length, rate)
  buffer.copyToChannel(samples as Float32Array<ArrayBuffer>, 0)
  const src = ctx.createBufferSource()
  src.buffer = buffer
  src.connect(ctx.destination)
  src.start()
  return (await ctx.startRendering()).getChannelData(0)
}

/**
 * Speech to text on the viewer's own GPU (Whisper via transformers.js + WebGPU): nothing is sent
 * anywhere, and it works even when the server has no AI key. Experimental.
 */
export function BrowserTranscript() {
  const gpuState = useGpu()
  const media = useCaseMedia()
  const pb = usePlayback()
  const [model, setModel] = useState(MODELS[0].id)
  const [status, setStatus] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<{ text: string; chunks: Chunk[]; seconds: number; model: string } | null>(null)

  if (gpuState === 'loading' || !gpuState || !media.mediaUrl) return null

  const run = async () => {
    setBusy(true)
    setError(null)
    setResult(null)
    const started = performance.now()
    try {
      setStatus('Decoding the soundtrack…')
      const audio = await decodeAudio(media)
      const pcm = await to16k(audio.samples, audio.rate)
      setStatus('Loading the speech model (first time: a download, then cached)…')
      // Loaded on demand: transformers.js is large and most visits never need it.
      const { pipeline } = await import('@huggingface/transformers')
      const files: Record<string, number> = {}
      const asr = await pipeline('automatic-speech-recognition', model, {
        device: 'webgpu',
        dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
        progress_callback: (p: { status: string; file?: string; loaded?: number; total?: number }) => {
          if (p.status === 'progress' && p.file && p.total) {
            files[p.file] = (p.loaded ?? 0) / p.total
            const done = Object.values(files)
            setStatus(`Downloading the model… ${Math.round((done.reduce((a, b) => a + b, 0) / done.length) * 100)}%`)
          }
        },
      })
      setStatus(`Transcribing ${mmss(pcm.length / 16000)} of audio on your GPU…`)
      const out = (await asr(pcm, { chunk_length_s: 30, stride_length_s: 5, return_timestamps: true })) as { text: string; chunks?: Chunk[] }
      setResult({ text: out.text.trim(), chunks: out.chunks ?? [], seconds: (performance.now() - started) / 1000, model })
      setStatus(null)
    } catch (e) {
      setError(String((e as Error)?.message ?? e))
      setStatus(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="frame-scan browser-ai">
      <div className="frame-scan-head">
        <b>
          <Icon name="microphone-lines" /> Transcript on your device
        </b>
        <span className="exp-tag">experimental</span>
        <span className="faint small">Whisper runs on your GPU in this browser: the audio never leaves your machine.</span>
        <span className="grow" />
        <select className="input ctl-select" value={model} onChange={(e) => setModel(e.target.value)} disabled={busy} aria-label="Speech model">
          {MODELS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
        <button className="btn btn-blue sm" onClick={run} disabled={busy}>
          <Icon name="play" /> {busy ? 'Working…' : result ? 'Run again' : 'Transcribe'}
        </button>
      </div>
      {status && <p className="faint small">{status}</p>}
      {error && <p className="chart-foot">Could not transcribe: {error}</p>}
      {result && (
        <div className="browser-ai-out">
          {result.chunks.length ? (
            <ol className="browser-ai-chunks">
              {result.chunks.map((c, i) => (
                <li key={i}>
                  <button className="linkish mono" onClick={() => pb.seek(c.timestamp[0])}>
                    {mmss(c.timestamp[0])}
                  </button>{' '}
                  {c.text.trim()}
                </li>
              ))}
            </ol>
          ) : (
            <p>{result.text || '(no speech recognised)'}</p>
          )}
          <p className="faint small">
            {result.model.split('/')[1]} · {result.seconds.toFixed(1)} s including loading · machine transcription: check against the audio.
          </p>
        </div>
      )}
    </div>
  )
}
