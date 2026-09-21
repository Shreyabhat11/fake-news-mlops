import { cn } from '@/lib/cn'

interface Props {
  label: string
  value: number // 0..1
  tone: 'real' | 'fake'
}

export function ProbabilityBar({ label, value, tone }: Props) {
  const pct = Math.round(value * 1000) / 10
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between text-sm">
        <span className="text-text-secondary">{label}</span>
        <span className="font-mono text-text-primary">{pct.toFixed(1)}%</span>
      </div>
      <div
        className="prob-bar-track"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${label}: ${pct.toFixed(1)} percent`}
      >
        <div
          className={cn('prob-bar-fill', tone === 'real' ? 'bg-signal-real' : 'bg-signal-fake')}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
