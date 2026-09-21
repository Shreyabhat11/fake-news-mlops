import type { ReactNode } from 'react'
import { AlertTriangle, Loader2, WifiOff } from 'lucide-react'
import type { HealthState } from '@/hooks/useHealth'
import { cn } from '@/lib/cn'

interface Props {
  health: HealthState
  onRetry: () => void
}

export function StatusIndicator({ health, onRetry }: Props) {
  if (health.status === 'checking') {
    return (
      <div className="flex items-center gap-2 text-xs text-text-muted">
        <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
        <span>Checking service…</span>
      </div>
    )
  }

  if (health.status === 'online') {
    const { model_loaded, embedding_service_ready } = health.data
    const allReady = model_loaded && embedding_service_ready
    return (
      <div
        className="flex items-center gap-2 text-xs text-text-secondary"
        role="status"
        aria-label={allReady ? 'API online, model and embeddings ready' : 'API online, some services degraded'}
      >
        <span className="relative flex h-2 w-2">
          <span
            className={cn(
              'absolute inline-flex h-full w-full rounded-full opacity-60 animate-pulseSoft',
              allReady ? 'bg-signal-real' : 'bg-amber-400'
            )}
          />
          <span
            className={cn('relative inline-flex h-2 w-2 rounded-full', allReady ? 'bg-signal-real' : 'bg-amber-400')}
          />
        </span>
        <span>API online</span>
        <span className="text-ink-border">·</span>
        <span className={model_loaded ? 'text-text-secondary' : 'text-amber-400'}>
          Model {model_loaded ? 'ready' : 'loading'}
        </span>
        <span className="text-ink-border">·</span>
        <span className={embedding_service_ready ? 'text-text-secondary' : 'text-amber-400'}>
          Embeddings {embedding_service_ready ? 'ready' : 'loading'}
        </span>
      </div>
    )
  }

  if (health.status === 'waking') {
    return (
      <div className="flex items-center gap-2 text-xs text-amber-400" role="status">
        <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
        <span>Waking the analysis service — this can take up to a minute on first load.</span>
      </div>
    )
  }

  return (
    <button
      type="button"
      onClick={onRetry}
      className="flex items-center gap-2 text-xs text-signal-fake hover:text-signal-fake/80 transition-colors"
    >
      <WifiOff className="h-3.5 w-3.5" aria-hidden="true" />
      <span>Unable to reach the API — tap to retry</span>
    </button>
  )
}

export function InlineAlert({
  tone = 'warning',
  children,
}: {
  tone?: 'warning' | 'error'
  children: ReactNode
}) {
  return (
    <div
      role="alert"
      className={cn(
        'flex items-start gap-2.5 rounded-xl border px-4 py-3 text-sm',
        tone === 'warning'
          ? 'border-amber-400/25 bg-amber-400/10 text-amber-200'
          : 'border-signal-fake/30 bg-signal-fake/10 text-red-200'
      )}
    >
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
      <span>{children}</span>
    </div>
  )
}
