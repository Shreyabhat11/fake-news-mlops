interface Props {
  confidence: number // 0..1
  tone: 'real' | 'fake'
}

const SIZE = 148
const STROKE = 10
const RADIUS = (SIZE - STROKE) / 2
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

export function ConfidenceGauge({ confidence, tone }: Props) {
  const pct = Math.max(0, Math.min(1, confidence))
  const offset = CIRCUMFERENCE * (1 - pct)
  const color = tone === 'real' ? '#3DDC97' : '#F2665A'

  return (
    <div className="relative flex h-[148px] w-[148px] items-center justify-center">
      <svg width={SIZE} height={SIZE} className="-rotate-90">
        <circle
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={RADIUS}
          fill="none"
          stroke="rgba(232,234,237,0.08)"
          strokeWidth={STROKE}
        />
        <circle
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={RADIUS}
          fill="none"
          stroke={color}
          strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 0.9s cubic-bezier(0.16, 1, 0.3, 1)' }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="font-display text-3xl text-text-primary">{(pct * 100).toFixed(1)}%</span>
        <span className="mt-0.5 text-xs text-text-muted">Confidence</span>
      </div>
    </div>
  )
}
