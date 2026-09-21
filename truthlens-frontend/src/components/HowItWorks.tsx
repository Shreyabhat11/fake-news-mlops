const STEPS = [
  {
    number: '01',
    title: 'Article input',
    body: 'You provide a headline and the article body through the form above.',
  },
  {
    number: '02',
    title: 'Semantic analysis',
    body: 'The deployed embedding service (Sentence Transformers) converts the article into a 384-dimensional semantic vector.',
  },
  {
    number: '03',
    title: 'ML classification',
    body: 'A trained classifier scores that vector and returns a REAL/FAKE prediction with a probability estimate.',
  },
]

export function HowItWorks() {
  return (
    <section id="how-it-works" className="mx-auto max-w-6xl px-6 py-20 md:py-28">
      <div className="max-w-xl">
        <h2 className="font-display text-3xl text-text-primary sm:text-4xl">How it works</h2>
        <p className="mt-4 text-text-secondary">
          Three steps run on every request, entirely on the deployed FastAPI backend. The
          system evaluates linguistic patterns — it does not fact-check claims against the
          internet.
        </p>
      </div>

      <div className="mt-12 grid gap-8 sm:grid-cols-3 sm:gap-6">
        {STEPS.map((step) => (
          <div key={step.number} className="border-t border-ink-border pt-5">
            <span className="font-mono text-sm text-text-muted">{step.number}</span>
            <h3 className="mt-3 font-display text-xl text-text-primary">{step.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-text-secondary">{step.body}</p>
          </div>
        ))}
      </div>
    </section>
  )
}
