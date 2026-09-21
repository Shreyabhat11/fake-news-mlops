import { useEffect, useState } from 'react'
import { getModelInfo } from '@/services/api'
import type { ModelInfoResponse } from '@/types/api'

export function useModelInfo() {
  const [data, setData] = useState<ModelInfoResponse | null>(null)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    getModelInfo(controller.signal).then((res) => {
      if (res.ok) setData(res.data)
      setLoaded(true)
    })
    return () => controller.abort()
  }, [])

  return { data, loaded }
}
