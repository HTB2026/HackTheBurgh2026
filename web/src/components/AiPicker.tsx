import type { AiProvider, Health } from '../lib/types'

export const AI_LABEL: Record<AiProvider, string> = { openai: 'OpenAI', gemini: 'Gemini', ollama: 'Gemma' }

/** One-line hint per provider, shown on hover. */
export const AI_HINT: Record<AiProvider, string> = {
  openai: 'OpenAI (cloud): two speech models cross-check the transcript.',
  gemini: 'Google Gemini (cloud): two Gemini models cross-check the transcript. The free tier allows only a few cases a day.',
  ollama: 'Gemma 4 on your own server (Ollama): nothing leaves your infrastructure. Slower, and the transcript has a single reading.',
}

/** The AI providers a case can pick on this server (only worth showing when there are two or more). */
export const aiChoices = (health: Health | null): AiProvider[] => health?.ai?.providers ?? []

/** Choose which AI provider (OpenAI or Gemini) a case uses. Renders nothing unless there is a choice. */
export function AiPicker({
  health,
  value,
  onChange,
  compact,
}: {
  health: Health | null
  value: AiProvider | null
  onChange: (p: AiProvider) => void
  compact?: boolean
}) {
  const choices = aiChoices(health)
  if (choices.length < 2) return null
  const current = value && choices.includes(value) ? value : (health?.ai?.default ?? choices[0])
  return (
    <select
      className={`input ai-picker${compact ? ' sm' : ''}`}
      value={current}
      onChange={(e) => onChange(e.target.value as AiProvider)}
      aria-label="AI model"
      data-tip={compact ? 'AI model used when this case is run again' : undefined}
    >
      {choices.map((p) => (
        <option key={p} value={p} title={AI_HINT[p]}>
          {AI_LABEL[p] ?? p}
        </option>
      ))}
    </select>
  )
}
