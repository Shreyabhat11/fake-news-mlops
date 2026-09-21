import { Github, ScanEye } from 'lucide-react'
import type { HealthState } from '@/hooks/useHealth'
import { StatusIndicator } from './StatusIndicator'

const GITHUB_URL = 'https://github.com/Shreyabhat11/fake-news-mlops'

interface Props {
  health: HealthState
  onRetryHealth: () => void
}

export function Navbar({ health, onRetryHealth }: Props) {
  return (
    <header className="sticky top-0 z-50 border-b border-ink-border bg-ink-900/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <a href="#top" className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-signal-real/10 text-signal-real">
            <ScanEye className="h-4 w-4" aria-hidden="true" />
          </span>
          <span className="font-display text-lg tracking-tight text-text-primary">TruthLens</span>
        </a>

        <nav className="hidden items-center gap-8 text-sm text-text-secondary md:flex" aria-label="Primary">
          <a href="#detector" className="transition-colors hover:text-text-primary">
            Detector
          </a>
          <a href="#how-it-works" className="transition-colors hover:text-text-primary">
            How it works
          </a>
          <a href="#model" className="transition-colors hover:text-text-primary">
            Model
          </a>
        </nav>

        <div className="flex items-center gap-4">
          <div className="hidden lg:block">
            <StatusIndicator health={health} onRetry={onRetryHealth} />
          </div>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            aria-label="View source on GitHub"
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-ink-border text-text-secondary transition-colors hover:border-ink-borderHover hover:text-text-primary"
          >
            <Github className="h-4 w-4" aria-hidden="true" />
          </a>
        </div>
      </div>
      <div className="border-t border-ink-border px-6 py-2 lg:hidden">
        <StatusIndicator health={health} onRetry={onRetryHealth} />
      </div>
    </header>
  )
}
