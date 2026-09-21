import { useCallback, useRef, useState } from 'react'
import { predict } from '@/services/api'
import type { ApiError, PredictResponse } from '@/types/api'
import type { ExampleArticle } from '@/lib/constants'

export type AnalyzerStatus = 'idle' | 'validating' | 'loading' | 'waking' | 'success' | 'error'

const MIN_WORDS = 5
const MIN_CHARS = 10
const MAX_CHARS = 10000

export function validateArticleText(text: string): string | null {
  const trimmed = text.trim()
  if (trimmed.length === 0) {
    return 'Paste an article body to analyze.'
  }
  if (trimmed.length < MIN_CHARS) {
    return `Article text must be at least ${MIN_CHARS} characters.`
  }
  if (trimmed.length > MAX_CHARS) {
    return `Article text must be under ${MAX_CHARS.toLocaleString()} characters.`
  }
  if (trimmed.split(/\s+/).filter(Boolean).length < MIN_WORDS) {
    return `Article text must contain at least ${MIN_WORDS} words.`
  }
  return null
}

export function useAnalyzer() {
  const [title, setTitle] = useState('')
  const [text, setText] = useState('')
  const [status, setStatus] = useState<AnalyzerStatus>('idle')
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [fieldError, setFieldError] = useState<string | null>(null)
  const requestSeq = useRef(0)

  const loadExample = useCallback((example: ExampleArticle) => {
    setTitle(example.title)
    setText(example.text)
    setFieldError(null)
    setError(null)
    setStatus('idle')
    setResult(null)
  }, [])

  const reset = useCallback(() => {
    setTitle('')
    setText('')
    setFieldError(null)
    setError(null)
    setStatus('idle')
    setResult(null)
  }, [])

  const analyze = useCallback(async () => {
    const validation = validateArticleText(text)
    if (validation) {
      setFieldError(validation)
      setStatus('idle')
      return
    }
    setFieldError(null)
    setError(null)
    setResult(null)
    setStatus('loading')

    const seq = ++requestSeq.current
    const onColdStart = () => {
      if (requestSeq.current === seq) setStatus('waking')
    }

    const res = await predict({ text: text.trim(), title: title.trim() || undefined }, onColdStart)

    if (requestSeq.current !== seq) return // a newer request superseded this one

    if (res.ok) {
      setResult(res.data)
      setStatus('success')
    } else {
      setError(res.error)
      setStatus('error')
    }
  }, [text, title])

  return {
    title,
    setTitle,
    text,
    setText,
    status,
    result,
    error,
    fieldError,
    setFieldError,
    loadExample,
    analyze,
    reset,
    limits: { MIN_WORDS, MIN_CHARS, MAX_CHARS },
  }
}
