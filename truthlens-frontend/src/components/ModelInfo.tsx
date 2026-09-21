import { useModelInfo } from '@/hooks/useModelInfo'
import { MODEL_FACTS } from '@/lib/constants'

interface Stat {
  label: string
  value: string
}

export function ModelInfo() {
  const { data } = useModelInfo()

  const classifier = data?.classifier || MODEL_FACTS.fallbackClassifier
  const version = data?.model_version || MODEL_FACTS.fallbackVersion
  const testF1 = data?.test_f1 ?? MODEL_FACTS.testF1

  const stats: Stat[] = [
    { label: 'Model', value: classifier },
    { label: 'Embeddings', value: MODEL_FACTS.embeddingModel },
    { label: 'Embedding dimension', value: String(MODEL_FACTS.embeddingDimension) },
    { label: 'Model version', value: version },
    { label: 'Dataset', value: MODEL_FACTS.datasetSize },
    { label: 'Test F1', value: testF1.toFixed(3) },
    { label: 'Test ROC-AUC', value: MODEL_FACTS.testRocAuc.toFixed(3) },
  ]

  return (
    <section id="model" className="border-t border-ink-border">
      <div className="mx-auto max-w-6xl px-6 py-20 md:py-28">
        <div className="max-w-xl">
          <h2 className="font-display text-3xl text-text-primary sm:text-4xl">Model</h2>
          <p className="mt-4 text-text-secondary">
            Inference runs on a classifier trained over sentence embeddings — the deployed API
            performs both the embedding step and the classification step for every request.
          </p>
        </div>

        <dl className="mt-12 grid grid-cols-1 gap-px overflow-hidden rounded-2xl border border-ink-border bg-ink-border sm:grid-cols-2 lg:grid-cols-4">
          {stats.map((stat) => (
            <div key={stat.label} className="bg-ink-900 p-5">
              <dt className="text-xs text-text-muted">{stat.label}</dt>
              <dd className="mt-2 font-mono text-sm text-text-primary">{stat.value}</dd>
            </div>
          ))}
        </dl>

        {data?.canary_mode && (
          <p className="mt-4 text-xs text-amber-400">Canary mode active — a subset of traffic is routed to a challenger model.</p>
        )}
        {data?.shadow_mode && (
          <p className="mt-1 text-xs text-brand">Shadow mode active — a challenger model logs comparison predictions.</p>
        )}
      </div>
    </section>
  )
}
