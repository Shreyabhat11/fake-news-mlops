import type {
  ApiError,
  ApiResult,
  BatchPredictRequest,
  BatchPredictResponse,
  HealthResponse,
  ModelInfoResponse,
  PredictRequest,
  PredictResponse,
} from '@/types/api'

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/+$/, '') ||
  'https://fake-news-mlops.onrender.com'

/** Render free-tier instances can take 30–60s to wake from an idle spin-down. */
const COLD_START_TIMEOUT_MS = 60_000
/** Fires an onColdStart callback if a request has been pending this long. */
const COLD_START_THRESHOLD_MS = 2_500
/** Health checks should be quick — don't block the UI waiting on a cold API. */
const HEALTH_TIMEOUT_MS = 8_000

interface FetchOptions {
  signal?: AbortSignal
  onColdStart?: () => void
  timeoutMs?: number
}

async function request<T>(
  path: string,
  init: RequestInit,
  options: FetchOptions = {}
): Promise<ApiResult<T>> {
  const { onColdStart, timeoutMs = COLD_START_TIMEOUT_MS } = options
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  const coldStartTimer = onColdStart
    ? setTimeout(onColdStart, COLD_START_THRESHOLD_MS)
    : undefined

  // Support an externally-provided signal (e.g. component unmount) alongside our own timeout.
  const externalAbort = () => controller.abort()
  options.signal?.addEventListener('abort', externalAbort)

  try {
    const res = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...(init.headers || {}),
      },
    })

    if (!res.ok) {
      return { ok: false, error: await parseErrorResponse(res) }
    }

    const data = (await res.json()) as T
    return { ok: true, data }
  } catch (err) {
    return { ok: false, error: toApiError(err) }
  } finally {
    clearTimeout(timeout)
    if (coldStartTimer) clearTimeout(coldStartTimer)
    options.signal?.removeEventListener('abort', externalAbort)
  }
}

async function parseErrorResponse(res: Response): Promise<ApiError> {
  let detailText = ''
  try {
    const body = await res.json()
    if (typeof body?.detail === 'string') {
      detailText = body.detail
    } else if (Array.isArray(body?.detail)) {
      // FastAPI / Pydantic validation error array
      detailText = body.detail
        .map((d: { msg?: string }) => d.msg)
        .filter(Boolean)
        .join(' ')
    } else if (typeof body?.error === 'string') {
      detailText = body.error
    }
  } catch {
    // Non-JSON error body — fall through with a generic message.
  }

  if (res.status === 422) {
    return {
      kind: 'validation',
      status: 422,
      message: detailText || 'The article text does not meet the minimum requirements.',
    }
  }
  if (res.status === 429) {
    return {
      kind: 'rate_limited',
      status: 429,
      message: 'Too many requests. Please wait a moment and try again.',
    }
  }
  if (res.status === 503) {
    return {
      kind: 'server',
      status: 503,
      message: detailText || 'The analysis service is still starting up. Please try again in a few seconds.',
    }
  }
  if (res.status >= 500) {
    return {
      kind: 'server',
      status: res.status,
      message: detailText || 'The analysis service ran into a problem. Please try again.',
    }
  }
  return {
    kind: 'unknown',
    status: res.status,
    message: detailText || `Request failed with status ${res.status}.`,
  }
}

function toApiError(err: unknown): ApiError {
  if (err instanceof DOMException && err.name === 'AbortError') {
    return {
      kind: 'timeout',
      message:
        'The analysis service is taking longer than expected to respond. It may be waking up from idle — please try again.',
    }
  }
  if (err instanceof TypeError) {
    // fetch() throws a TypeError for network-level failures (offline, CORS, DNS, connection refused)
    return {
      kind: 'network',
      message: 'Unable to reach the analysis service. Check your connection and try again.',
    }
  }
  return {
    kind: 'unknown',
    message: 'Something unexpected went wrong. Please try again.',
  }
}

export async function checkHealth(signal?: AbortSignal): Promise<ApiResult<HealthResponse>> {
  return request<HealthResponse>(
    '/health',
    { method: 'GET' },
    { signal, timeoutMs: HEALTH_TIMEOUT_MS }
  )
}

export async function getModelInfo(signal?: AbortSignal): Promise<ApiResult<ModelInfoResponse>> {
  return request<ModelInfoResponse>(
    '/model/info',
    { method: 'GET' },
    { signal, timeoutMs: HEALTH_TIMEOUT_MS }
  )
}

export async function predict(
  body: PredictRequest,
  onColdStart?: () => void
): Promise<ApiResult<PredictResponse>> {
  return request<PredictResponse>(
    '/predict',
    { method: 'POST', body: JSON.stringify(body) },
    { onColdStart }
  )
}

export async function batchPredict(
  body: BatchPredictRequest,
  onColdStart?: () => void
): Promise<ApiResult<BatchPredictResponse>> {
  return request<BatchPredictResponse>(
    '/batch_predict',
    { method: 'POST', body: JSON.stringify(body) },
    { onColdStart }
  )
}
