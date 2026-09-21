import { ArrowDown } from 'lucide-react'
import { VerdictScope } from './VerdictScope'

export function Hero() {
  return (
    <section id="top" className="mx-auto max-w-6xl px-6 pb-20 pt-16 md:pb-28 md:pt-24">
      <div className="grid items-center gap-14 md:grid-cols-2 md:gap-10">
        <div>
          <p className="mb-5 text-sm text-text-muted">AI-powered news credibility analysis</p>
          <h1 className="font-display text-5xl leading-[1.08] text-text-primary sm:text-6xl">
            Is this news
            <br />
            real, or fake?
          </h1>
          <p className="mt-6 max-w-md text-lg leading-relaxed text-text-secondary">
            TruthLens analyzes a news article using a machine learning model trained to
            recognize the linguistic patterns that separate genuine reporting from fabricated
            claims.
          </p>
          <div className="mt-9 flex flex-wrap items-center gap-4">
            <a
              href="#detector"
              className="inline-flex items-center gap-2 rounded-xl bg-signal-real px-5 py-3 text-sm font-medium text-ink-950 transition-transform hover:scale-[1.02] active:scale-[0.99]"
            >
              Analyze an article
              <ArrowDown className="h-4 w-4" aria-hidden="true" />
            </a>
            <a
              href="#how-it-works"
              className="text-sm text-text-secondary underline decoration-ink-border underline-offset-4 transition-colors hover:text-text-primary hover:decoration-text-secondary"
            >
              How it works
            </a>
          </div>
        </div>

        <div className="relative mx-auto aspect-square w-full max-w-md">
          <div className="glass-panel absolute inset-0 rounded-[28px] border border-ink-border shadow-card" />
          <div className="relative flex h-full w-full items-center justify-center p-6">
            <VerdictScope />
          </div>
        </div>
      </div>
    </section>
  )
}
