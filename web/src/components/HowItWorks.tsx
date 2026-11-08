import { useEffect } from 'react'
import { useHealth } from '../lib/health'
import { modulePlain, VERDICT } from '../lib/plain'
import { Alert, Icon, Redacted, SectionTitle } from './ui/ui'

export function HowItWorks({ section }: { section?: string }) {
  const { health } = useHealth()
  useEffect(() => {
    if (section !== 'limits') return
    const frame = requestAnimationFrame(() => document.getElementById('limits')?.scrollIntoView({ block: 'start' }))
    return () => cancelAnimationFrame(frame)
  }, [section, health])
  return (
    <>
      <section className="banner">
        <div className="wrap">
          <div>
            <h1>How it works</h1>
            <p>{health ? `${health.modules.length} independent witnesses` : 'Independent witnesses'}, questioned one by one.</p>
          </div>
        </div>
      </section>
      <main className="wrap how">
        <Alert title="Leads, not proof">
          EARWITNESS measures things a person cannot see or hear. Each finding says how it was measured and how sure it is. Use it to decide where to
          look, then confirm with the original source.
        </Alert>

        <SectionTitle icon="list-check" accent="teal" small="Every check runs on every video">
          The checks for video
        </SectionTitle>
        <div className="how-grid">
          {(health?.modules ?? []).map((m) => {
            const p = modulePlain(m.name, m.title)
            return (
              <article key={m.name} className={`panel how-card c-${p.accent}`}>
                <header>
                  <Icon name={p.icon} />
                  <h3>{p.name}</h3>
                </header>
                <p>{p.does}</p>
                {p.reveals && (
                  <p>
                    <b>Can reveal:</b> <span className="muted">{p.reveals}</span>
                  </p>
                )}
                {p.limits && (
                  <p>
                    <b>Limits:</b> <span className="muted">{p.limits}</span>
                  </p>
                )}
              </article>
            )
          })}
          {!health && <Redacted lines={6} />}
        </div>

        <SectionTitle icon="image" accent="purple" small="A photograph is one exposure at full resolution: different questions, different checks">
          The checks for photographs
        </SectionTitle>
        <div className="how-grid">
          {(health?.image_modules ?? []).map((m) => {
            const p = modulePlain(m.name, m.title)
            return (
              <article key={`img:${m.name}`} className={`panel how-card c-${p.accent}`}>
                <header>
                  <Icon name={p.icon} />
                  <h3>{p.name}</h3>
                </header>
                <p>{p.does}</p>
                {p.reveals && (
                  <p>
                    <b>Can reveal:</b> <span className="muted">{p.reveals}</span>
                  </p>
                )}
                {p.limits && (
                  <p>
                    <b>Limits:</b> <span className="muted">{p.limits}</span>
                  </p>
                )}
              </article>
            )
          })}
          {!health && <Redacted lines={6} />}
        </div>

        <SectionTitle icon="scale-balanced" accent="red" id="verdict" small="The verdict is the strongest claim the evidence supports">
          Reading the verdict
        </SectionTitle>
        <div className="panel pad">
          <dl className="stats">
            {(['edited', 'possible', 'clean', 'synthetic'] as const).map((v) => (
              <div key={v}>
                <dt>
                  <span className={`pill inline v-${v}`}>
                    <Icon name={VERDICT[v].icon} /> {VERDICT[v].word}
                  </span>
                </dt>
                <dd className="left">{VERDICT[v].line}</dd>
              </div>
            ))}
          </dl>
        </div>

        <SectionTitle icon="ban" accent="grey" id="limits" small="Be honest about what a file can and cannot tell you">
          What it cannot prove
        </SectionTitle>
        <div className="panel pad prose">
          <p>
            <b>No cuts found is not the same as authentic.</b> A video can be staged, or an AI-generated clip can be recorded off a screen in one take.
          </p>
          <p>
            <b>A photo with no signs of editing can still be generated.</b> Only a label the generator wrote (Content Credentials, a recipe, a name)
            proves AI origin; pixel statistics merely raise suspicion, and a generated picture re-saved through a phone loses even those.
          </p>
          <p>
            <b>Labels inside a file can be edited.</b> Dates, device names and GPS are treated as weak evidence unless another check (the power grid,
            the sun, the picture) agrees.
          </p>
          <p>
            <b>AI results are labelled AI.</b> They are useful leads, especially for location, but can be confidently wrong.
          </p>
          <p>
            <b>Re-uploads destroy clues.</b> Messaging apps and social networks strip metadata and recompress picture and sound. Ask for the original
            file whenever you can.
          </p>
        </div>
      </main>
    </>
  )
}
