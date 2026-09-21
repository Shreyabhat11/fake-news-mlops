/**
 * Types mirror the Pydantic schemas in the deployed FastAPI backend
 * (api/main.py) exactly. Do not add fields the API does not return.
 */

export type DriftStatus = 'normal' | 'warning' | 'critical'
export type PredictionLabel = 'REAL' | 'FAKE'

export interface PredictRequest {
  text: string
  title?: string
}

export interface PredictResponse {
  request_id: string
  prediction: PredictionLabel
  label: number // 1 = FAKE, 0 = REAL
  confidence: number
  fake_probability: number
  real_probability: number
  drift_status: DriftStatus
  model_version: string
  processing_time_ms: number
  cached: boolean
}

export interface BatchPredictRequest {
  texts: string[]
  titles?: string[]
}

export interface BatchPredictResponse {
  request_id: string
  predictions: PredictResponse[]
  total_processed: number
  processing_time_ms: number
}

export interface HealthResponse {
  status: string // "healthy" | "degraded"
  model_loaded: boolean
  embedding_service_ready: boolean
  redis_connected: boolean
  timestamp: string
}

export interface ModelInfoResponse {
  model_version: string
  classifier: string
  val_f1: number | null
  test_f1: number | null
  loaded_at: string
  canary_mode: boolean
  shadow_mode: boolean
}

/** Discriminated result wrapper so callers must handle failure explicitly. */
export type ApiResult<T> =
  | { ok: true; data: T }
  | { ok: false; error: ApiError }

export type ApiErrorKind =
  | 'cold_start'
  | 'timeout'
  | 'network'
  | 'validation'
  | 'rate_limited'
  | 'server'
  | 'unknown'

export interface ApiError {
  kind: ApiErrorKind
  message: string
  status?: number
}
