import { Loader2, Search, Sparkles } from 'lucide-react'
import type { AnalyzerStatus } from '@/hooks/useAnalyzer'
import { EXAMPLE_ARTICLES } from '@/lib/constants'
import { cn } from '@/lib/cn'

interface Props {
  title: string
  text: string
  status: AnalyzerStatus
  fieldError: string | null
  onTitleChange: (value: string) => void
  onTextChange: (value: string) => void
  onAnalyze: () => void
  onLoadExample: (id: string) => void
  maxChars: number
}

const isBusy = (status: AnalyzerStatus) => status === 'loading' || status === 'waking'

export function AnalyzerCard({
  title,
  text,
  status,
  fieldError,
  onTitleChange,
  onTextChange,
  onAnalyze,
  onLoadExample,
  maxChars,
}: Props) {
  const wordCount = text.trim().length ? text.trim().split(/\s+/).length : 0
  const charCount = text.length
  const busy = isBusy(status)

  return (
    <div className="glass-panel rounded-2xl border border-ink-border p-6 shadow-card sm:p-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="font-display text-2xl text-text-primary">Analyze an article</h2>
          <p className="mt-1.5 text-sm text-text-secondary">
            Provide a headline and the article body to get a credibility reading.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {EXAMPLE_ARTICLES.map((ex) => (
            <button
              key={ex.id}
              type="button"
              onClick={() => onLoadExample(ex.id)}
              disabled={busy}
              className="rounded-lg border border-ink-border px-3 py-1.5 text-xs text-text-secondary transition-colors hover:border-ink-borderHover hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-40"
            >
              {ex.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-6 space-y-5">
        <div>
          <label htmlFor="article-title" className="mb-2 block text-sm text-text-secondary">
            Article headline <span className="text-text-muted">(optional)</span>
          </label>
          <input
            id="article-title"
            type="text"
            value={title}
            onChange={(e) => onTitleChange(e.target.value)}
            disabled={busy}
            maxLength={500}
            placeholder="e.g. Fed holds rates steady, signals cautious approach"
            className="w-full rounded-xl border border-ink-border bg-ink-800/60 px-4 py-3 text-sm text-text-primary placeholder:text-text-muted transition-colors focus:border-brand/50 disabled:opacity-50"
          />
        </div>

        <div>
          <div className="mb-2 flex items-center justify-between">
            <label htmlFor="article-text" className="block text-sm text-text-secondary">
              Article text
            </label>
            <span
              className={cn(
                'font-mono text-xs',
                charCount > maxChars ? 'text-signal-fake' : 'text-text-muted'
              )}
            >
              {wordCount.toLocaleString()} words · {charCount.toLocaleString()}/{maxChars.toLocaleString()} chars
            </span>
          </div>
          <textarea
            id="article-text"
            value={text}
            onChange={(e) => onTextChange(e.target.value)}
            disabled={busy}
            rows={9}
            maxLength={maxChars}
            placeholder="Paste the full article body here…"
            aria-describedby={fieldError ? 'article-text-error' : undefined}
            aria-invalid={Boolean(fieldError)}
            className={cn(
              'w-full resize-y rounded-xl border bg-ink-800/60 px-4 py-3 text-sm leading-relaxed text-text-primary placeholder:text-text-muted transition-colors focus:border-brand/50 disabled:opacity-50',
              fieldError ? 'border-signal-fake/50' : 'border-ink-border'
            )}
          />
          {fieldError && (
            <p id="article-text-error" className="mt-2 text-xs text-signal-fake">
              {fieldError}
            </p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-3 pt-1">
          <button
            type="button"
            onClick={onAnalyze}
            disabled={busy}
            className="inline-flex min-w-[168px] items-center justify-center gap-2 rounded-xl bg-signal-real px-6 py-3 text-sm font-medium text-ink-950 transition-transform hover:scale-[1.01] active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:scale-100"
          >
            {busy ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                {status === 'waking' ? 'Starting up…' : 'Analyzing…'}
              </>
            ) : (
              <>
                <Search className="h-4 w-4" aria-hidden="true" />
                Analyze article
              </>
            )}
          </button>
          {status === 'waking' && (
            <span className="inline-flex items-center gap-1.5 text-xs text-amber-400">
              <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
              The analysis service is starting up — first requests can take up to a minute.
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
