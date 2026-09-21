import { useHealth } from '@/hooks/useHealth'
import { useAnalyzer } from '@/hooks/useAnalyzer'
import { EXAMPLE_ARTICLES } from '@/lib/constants'
import { Navbar } from '@/components/Navbar'
import { Hero } from '@/components/Hero'
import { AnalyzerCard } from '@/components/AnalyzerCard'
import { ResultPanel } from '@/components/ResultPanel'
import { HowItWorks } from '@/components/HowItWorks'
import { ModelInfo } from '@/components/ModelInfo'
import { BatchAnalysis } from '@/components/BatchAnalysis'
import { Footer } from '@/components/Footer'

function App() {
  const { state: health, refresh: refreshHealth } = useHealth()
  const analyzer = useAnalyzer()

  function handleLoadExample(id: string) {
    const example = EXAMPLE_ARTICLES.find((e) => e.id === id)
    if (example) analyzer.loadExample(example)
  }

  return (
    <div className="min-h-screen">
      <Navbar health={health} onRetryHealth={refreshHealth} />

      <main>
        <Hero />

        <section id="detector" className="mx-auto max-w-3xl px-6 pb-24">
          <div className="space-y-6">
            <AnalyzerCard
              title={analyzer.title}
              text={analyzer.text}
              status={analyzer.status}
              fieldError={analyzer.fieldError}
              onTitleChange={analyzer.setTitle}
              onTextChange={(v) => {
                analyzer.setText(v)
                if (analyzer.fieldError) analyzer.setFieldError(null)
              }}
              onAnalyze={analyzer.analyze}
              onLoadExample={handleLoadExample}
              maxChars={analyzer.limits.MAX_CHARS}
            />

            <ResultPanel
              status={analyzer.status}
              result={analyzer.result}
              error={analyzer.error}
              onRetry={analyzer.analyze}
            />
          </div>
        </section>

        <HowItWorks />
        <ModelInfo />
        <BatchAnalysis />
      </main>

      <Footer />
    </div>
  )
}

export default App
