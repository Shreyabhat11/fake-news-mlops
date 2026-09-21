import { useState } from 'react'
import { ChevronDown, Layers, Loader2 } from 'lucide-react'
import { batchPredict } from '@/services/api'
import type { ApiError, PredictResponse } from '@/types/api'
import { cn } from '@/lib/cn'
import { InlineAlert } from './StatusIndicator'

const MAX_ARTICLES = 50

export function BatchAnalysis() {
  const [open, setOpen] = useState(false)
  const [raw, setRaw] = useState('')
  const [status, setStatus] = useState<'idle' | 'loading' | 'waking' | 'error' | 'success'>('idle')
  const [results, setResults] = useState<PredictResponse[]>([])
  const [error, setError] = useState<ApiError | null>(null)

  const articles = raw
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)

  async function handleSubmit() {
    if (articles.length === 0) return
    if (articles.length > MAX_ARTICLES) {
      setError({ kind: 'validation', message: `Batch requests are limited to ${MAX_ARTICLES} articles.` })
      setStatus('error')
      return
    }
    setStatus('loading')
    setError(null)

    const res = await batchPredict({ texts: articles }, () => setStatus('waking'))

    if (res.ok) {
      setResults(res.data.predictions)
      setStatus('success')
    } else {
      setError(res.error)
      setStatus('error')
    }
  }

  const busy = status === 'loading' || status === 'waking'

  return (
    <section className="border-t border-ink-border">
      <div className="mx-auto max-w-6xl px-6 py-14">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="flex w-full items-center justify-between gap-4 text-left"
        >
          <span className="flex items-center gap-2.5">
            <Layers className="h-4 w-4 text-text-muted" aria-hidden="true" />
            <span className="font-display text-xl text-text-primary">Batch analysis</span>
            <span className="text-xs text-text-muted">optional</span>
          </span>
          <ChevronDown
            className={cn('h-4 w-4 text-text-muted transition-transform', open && 'rotate-180')}
            aria-hidden="true"
          />
        </button>

        {open && (
          <div className="mt-6">
            <p className="mb-4 max-w-xl text-sm text-text-secondary">
              Enter up to {MAX_ARTICLES} articles, one per line, to classify them together.
            </p>
            <textarea
              value={raw}
              onChange={(e) => setRaw(e.target.value)}
              disabled={busy}
              rows={5}
              placeholder={'Article one text…\nArticle two text…\nArticle three text…'}
              className="w-full resize-y rounded-xl border border-ink-border bg-ink-800/60 px-4 py-3 text-sm text-text-primary placeholder:text-text-muted focus:border-brand/50 disabled:opacity-50"
            />
            <div className="mt-4 flex items-center gap-3">
              <button
                type="button"
                onClick={handleSubmit}
                disabled={busy || articles.length === 0}
                className="inline-flex items-center gap-2 rounded-xl border border-ink-border px-5 py-2.5 text-sm text-text-primary transition-colors hover:border-ink-borderHover disabled:cursor-not-allowed disabled:opacity-50"
              >
                {busy && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}
                {status === 'waking' ? 'Starting up…' : busy ? 'Classifying…' : `Classify ${articles.length || ''} article${articles.length === 1 ? '' : 's'}`}
              </button>
            </div>

            {status === 'error' && error && (
              <div className="mt-5">
                <InlineAlert tone="error">{error.message}</InlineAlert>
              </div>
            )}

            {status === 'success' && results.length > 0 && (
              <div className="mt-6 overflow-x-auto rounded-xl border border-ink-border">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-ink-border text-xs text-text-muted">
                      <th scope="col" className="px-4 py-3 font-normal">
                        Article
                      </th>
                      <th scope="col" className="px-4 py-3 font-normal">
                        Prediction
                      </th>
                      <th scope="col" className="px-4 py-3 font-normal">
                        Confidence
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r, i) => (
                      <tr key={r.request_id} className="border-b border-ink-border last:border-b-0">
                        <td className="max-w-xs truncate px-4 py-3 text-text-secondary">{articles[i]}</td>
                        <td className="px-4 py-3">
                          <span
                            className={cn(
                              'font-mono text-xs',
                              r.prediction === 'REAL' ? 'text-signal-real' : 'text-signal-fake'
                            )}
                          >
                            {r.prediction}
                          </span>
                        </td>
                        <td className="px-4 py-3 font-mono text-text-primary">
                          {(r.confidence * 100).toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  )
}
