import { useCallback, useEffect, useRef, useState } from 'react'
import { checkHealth } from '@/services/api'
import type { HealthResponse } from '@/types/api'

export type HealthState =
  | { status: 'checking' }
  | { status: 'online'; data: HealthResponse }
  | { status: 'waking' } // first check failed/timed out — likely a cold start
  | { status: 'offline' }

const POLL_INTERVAL_MS = 20_000

export function useHealth() {
  const [state, setState] = useState<HealthState>({ status: 'checking' })
  const attemptRef = useRef(0)

  const runCheck = useCallback(async () => {
    const controller = new AbortController()
    const result = await checkHealth(controller.signal)
    attemptRef.current += 1

    if (result.ok) {
      setState({ status: 'online', data: result.data })
    } else if (attemptRef.current === 1) {
      // First failure is treated as a likely cold start rather than a hard failure.
      setState({ status: 'waking' })
    } else {
      setState({ status: 'offline' })
    }
  }, [])

  useEffect(() => {
    runCheck()
    const interval = setInterval(runCheck, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [runCheck])

  return { state, refresh: runCheck }
}
