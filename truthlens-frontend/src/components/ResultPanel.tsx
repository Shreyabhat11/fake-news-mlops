import { CheckCircle2, Clock3, Cpu, RefreshCcw, ShieldAlert, XCircle, Zap } from 'lucide-react'
import type { AnalyzerStatus } from '@/hooks/useAnalyzer'
import type { ApiError, PredictResponse } from '@/types/api'
import { ConfidenceGauge } from './ConfidenceGauge'
import { ProbabilityBar } from './ProbabilityBar'
import { cn } from '@/lib/cn'

interface Props {
  status: AnalyzerStatus
  result: PredictResponse | null
  error: ApiError | null
  onRetry: () => void
}

function errorCopy(error: ApiError): string {
  switch (error.kind) {
    case 'timeout':
      return 'The analysis service is taking longer than expected. It may still be waking up from idle — try again in a moment.'
    case 'network':
      return 'Unable to reach the analysis service. Check your connection and try again.'
    case 'rate_limited':
      return 'Too many requests right now. Please wait a moment before analyzing again.'
    case 'validation':
      return error.message
    case 'server':
      return error.message
    default:
      return 'Something went wrong while analyzing this article. Please try again.'
  }
}

export function ResultPanel({ status, result, error, onRetry }: Props) {
  if (status === 'loading' || status === 'waking') {
    return (
      <div className="glass-panel rounded-2xl border border-ink-border p-8 shadow-card">
        <div className="flex flex-col items-center gap-3 py-6 text-center">
          <div className="h-10 w-10 animate-spin rounded-full border-2 border-ink-border border-t-signal-real" />
          <p className="text-sm text-text-secondary">
            {status === 'waking'
              ? 'The analysis service is starting up. This can take up to a minute on the first request.'
              : 'Analyzing article…'}
          </p>
        </div>
      </div>
    )
  }

  if (status === 'error' && error) {
    return (
      <div className="glass-panel rounded-2xl border border-signal-fake/25 p-8 shadow-card">
        <div className="flex flex-col items-center gap-3 py-2 text-center">
          <ShieldAlert className="h-8 w-8 text-signal-fake" aria-hidden="true" />
          <p className="max-w-sm text-sm text-text-secondary">{errorCopy(error)}</p>
          <button
            type="button"
            onClick={onRetry}
            className="mt-2 inline-flex items-center gap-2 rounded-lg border border-ink-border px-4 py-2 text-sm text-text-primary transition-colors hover:border-ink-borderHover"
          >
            <RefreshCcw className="h-3.5 w-3.5" aria-hidden="true" />
            Try again
          </button>
        </div>
      </div>
    )
  }

  if (status !== 'success' || !result) {
    return null
  }

  const isReal = result.prediction === 'REAL'
  const tone: 'real' | 'fake' = isReal ? 'real' : 'fake'

  return (
    <div
      className={cn(
        'glass-panel rounded-2xl border p-6 shadow-card sm:p-8',
        isReal ? 'border-signal-real/25' : 'border-signal-fake/25'
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-6">
        <div className="flex items-start gap-3">
          {isReal ? (
            <CheckCircle2 className="mt-1 h-7 w-7 shrink-0 text-signal-real" aria-hidden="true" />
          ) : (
            <XCircle className="mt-1 h-7 w-7 shrink-0 text-signal-fake" aria-hidden="true" />
          )}
          <div>
            <h3 className="font-display text-2xl text-text-primary">
              {isReal ? 'Likely Real' : 'Likely Fake'}
            </h3>
            <p className="mt-1 text-sm text-text-secondary">
              {isReal
                ? 'This article shows patterns consistent with genuine reporting.'
                : 'This article shows patterns commonly associated with fabricated content.'}
            </p>
          </div>
        </div>
        <ConfidenceGauge confidence={result.confidence} tone={tone} />
      </div>

      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        <ProbabilityBar label="Real probability" value={result.real_probability} tone="real" />
        <ProbabilityBar label="Fake probability" value={result.fake_probability} tone="fake" />
      </div>

      <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-3 border-t border-ink-border pt-5 text-xs text-text-muted">
        <span className="inline-flex items-center gap-1.5 font-mono">
          <Cpu className="h-3.5 w-3.5" aria-hidden="true" />
          Model {result.model_version}
        </span>
        <span className="inline-flex items-center gap-1.5 font-mono">
          <Clock3 className="h-3.5 w-3.5" aria-hidden="true" />
          {result.processing_time_ms.toFixed(0)} ms
        </span>
        {result.cached && (
          <span className="inline-flex items-center gap-1.5 font-mono text-brand">
            <Zap className="h-3.5 w-3.5" aria-hidden="true" />
            Cached result
          </span>
        )}
        {result.drift_status !== 'normal' && (
          <span
            className={cn(
              'inline-flex items-center gap-1.5 font-mono',
              result.drift_status === 'warning' ? 'text-amber-400' : 'text-signal-fake'
            )}
          >
            <ShieldAlert className="h-3.5 w-3.5" aria-hidden="true" />
            Drift: {result.drift_status}
          </span>
        )}
      </div>

      <p className="mt-5 text-xs leading-relaxed text-text-muted">
        This result is a machine-learning prediction, not a determination of factual truth.
        Verify important claims using reliable sources.
      </p>
    </div>
  )
}
