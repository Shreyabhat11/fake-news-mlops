import { Github } from 'lucide-react'

const GITHUB_URL = 'https://github.com/Shreyabhat11/fake-news-mlops'

export function Footer() {
  return (
    <footer className="border-t border-ink-border">
      <div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-4 px-6 py-10 text-sm text-text-muted sm:flex-row sm:items-center">
        <p>TruthLens — a machine learning demo. Predictions are not fact-checks.</p>
        <a
          href={GITHUB_URL}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 transition-colors hover:text-text-secondary"
        >
          <Github className="h-4 w-4" aria-hidden="true" />
          Source on GitHub
        </a>
      </div>
    </footer>
  )
}
