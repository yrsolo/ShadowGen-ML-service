# Контракт ShadowGen ML Service

Статус: актуальный интеграционный контракт для frontend/backend/worker.

Этот документ является основным handoff-документом для команды продукта. Формальная машинная схема доступна у запущенного сервиса:

- `GET /openapi.json`
- `GET /docs`

## 1. Граница интеграции

Рекомендуемый production flow:

```text
Browser frontend -> ShadowGen-v2 backend/worker -> ShadowGen ML Service
```

Браузер не должен обращаться к ML-service напрямую:

- ML-service не реализует пользовательскую авторизацию;
- CORS для произвольного frontend origin не настраивается;
- product backend/worker отвечает за очередь, retries и доступ пользователя;
- ML-service отвечает только за ML pipeline и его внутреннюю async-очередь.

Base URL задаётся конфигурацией worker/backend, например:

```text
http://ml-core:9001
```

Для локального Docker-запуска:

```text
http://127.0.0.1:9001
```

Порт задаётся в `.env` через `SERVICE_HTTP_PORT`; default — `9001`. Это значение одновременно задаёт порт Uvicorn внутри контейнера и опубликованный host port.

## 2. Публичные endpoints

| Method | Path | Назначение |
|---|---|---|
| `GET` | `/health` | readiness и возможность принимать jobs |
| `GET` | `/v1/capabilities` | handshake возможностей сервиса |
| `POST` | `/v1/render` | синхронный render |
| `POST` | `/v1/render/jobs` | асинхронная постановка render job, ответ `202` |
| `GET` | `/v1/render/jobs/{job_id}` | статус и результат async job |
| `DELETE` | `/v1/render/jobs/{job_id}` | отмена job согласно `cancel_mode` |

`/playground` и `/v1/dev/*` не являются production API и по умолчанию отключены.

## 3. Обязательный handshake

После старта worker/backend должен вызвать:

```http
GET /health
GET /v1/capabilities
```

Пример ключевых полей `/health`:

```json
{
  "status": "ok",
  "service_version": "0.1.0",
  "active_backend_mode": "local",
  "async_enabled": true,
  "accepting_jobs": true,
  "preferred_submit_mode": "async"
}
```

Допустимые `status`:

- `ok` — сервис готов;
- `degraded` — работает с fallback/ограничениями;
- `draining` — новые задачи принимать нельзя;
- `overloaded` — достигнут capacity limit.

Frontend/backend не должен хардкодить поддержку async. Источник истины:

- `supported_submit_modes`;
- `preferred_submit_mode`;
- `job_execution.accepting_jobs`;
- `job_execution.max_running_jobs`;
- `job_execution.max_pending_jobs`.

## 4. Render request

Один и тот же JSON используется для sync и async endpoints.

```ts
type ShadowModel = "v1-gan" | "v2-diff";

type RenderRequest = {
  request_id?: string | null;
  pipeline_version: string;
  source: {
    mime_type: "image/jpeg" | "image/png" | "image/webp";
    image_base64: string;
  };
  preprocess?: {
    padding_px?: number;
  };
  shadow: {
    model?: ShadowModel | null;
    angle_deg: number;
    elevation_deg: number;
    softness: number;
    opacity: number;
    reflection: number;
  };
  background: {
    mode: "solid";
    color_hex: `#${string}`;
  };
  output: {
    format: "png" | "webp";
    width: number | null;
    height: number | null;
    return_debug: boolean;
  };
};
```

Правила:

- `source.image_base64` содержит чистый Base64 без префикса `data:image/...;base64,`;
- максимальный размер декодированного изображения по умолчанию `10 MiB`, точное значение приходит в `capabilities.max_image_bytes`;
- `pipeline_version` должен быть непустым; текущая рекомендуемая строка — `ml-shadowgen-v1`;
- `preprocess.padding_px`: `0..4096`, default `100`;
- `background.color_hex`: строго `#RRGGBB`;
- неизвестные поля верхнего уровня запрещены;
- `request_id` рекомендуется всегда передавать UUID/уникальный business id.

## 5. Модели тени и UI

### `v1-gan`

Назначение: управляемая классическая тень.

| Поле | Поведение сейчас |
|---|---|
| `angle_deg` | используется, диапазон `0..360` |
| `opacity` | используется для плотности тени, диапазон `0..1` |
| `elevation_deg` | принимается, но не влияет |
| `softness` | принимается, но не влияет |
| `reflection` | зарезервировано, не влияет |

Рекомендуемый UI: включить `Angle` и `Opacity`, остальные controls скрыть или disable.

### `v2-diff`

Назначение: автоматическая diffusion-тень без ручного управления направлением.

| Поле | Поведение сейчас |
|---|---|
| `elevation_deg` | может выбирать prompt/view bucket модели; это не точное геометрическое управление |
| `angle_deg` | не используется |
| `softness` | не используется |
| `opacity` | не применяется к текущему full-image output модели |
| `reflection` | не используется |

Рекомендуемый UI: модель выбирается пользователем, но ручные shadow controls выключены. При необходимости можно оставить `Elevation` как experimental control с явной подписью, что результат недетерминирован геометрически.

Важно: `v2-diff` возвращает полную картинку `background + shadow + generated object`. ML-core поверх неё повторно композит исходный вырезанный объект, чтобы сохранить качество предмета.

Frontend должен всегда явно отправлять `shadow.model`. Если поле не передано, используется server-side default, который может различаться между deployments.

## 6. Полный пример запроса

```json
{
  "request_id": "8ae65ca5-3a3b-40bf-a1bb-cbb3d3252782",
  "pipeline_version": "ml-shadowgen-v1",
  "source": {
    "mime_type": "image/jpeg",
    "image_base64": "<BASE64_WITHOUT_DATA_URL_PREFIX>"
  },
  "preprocess": {
    "padding_px": 100
  },
  "shadow": {
    "model": "v2-diff",
    "angle_deg": 45,
    "elevation_deg": 35,
    "softness": 0.5,
    "opacity": 0.65,
    "reflection": 0.0
  },
  "background": {
    "mode": "solid",
    "color_hex": "#FFFFFF"
  },
  "output": {
    "format": "png",
    "width": null,
    "height": null,
    "return_debug": false
  }
}
```

Поля shadow controls пока остаются обязательными даже для `v2-diff`; отправляйте безопасные defaults из примера.

## 7. Sync response

`POST /v1/render` возвращает `200`:

```ts
type Artifact = {
  name: string;
  kind: "final" | "debug";
  mime_type: string;
  image_base64: string;
};

type RenderResponse = {
  request_id: string | null;
  artifacts: Artifact[];
  metrics: {
    total_ms: number;
    decode_ms?: number | null;
    geometry_ms?: number | null;
    detection_ms?: number | null;
    segmentation_ms?: number | null;
    foreground_refinement_ms?: number | null;
    depth_ms?: number | null;
    normals_ms?: number | null;
    shadow_ms?: number | null;
    composition_ms?: number | null;
    encode_ms?: number | null;
    cache_ms?: number | null;
  };
  warnings: string[];
  model_info: {
    service_version: string;
    model_version: string;
  };
};
```

Клиент должен искать final artifact по `kind === "final"` или `name === "final"`, а не полагаться на позицию элемента в массиве.

Known non-fatal warnings:

- `detector_empty_full_frame_fallback`: detector did not find a confident object, so ML-core used the full source frame as the object bbox and continued with segmentation.
- `main_object_low_confidence`: detector confidence is below the service threshold; the render result is still valid, but product UI may show a soft quality warning.

Для отображения:

```ts
const src = `data:${artifact.mime_type};base64,${artifact.image_base64}`;
```

При `return_debug=false` достаточно обрабатывать только final artifact. Набор debug artifacts не является стабильным product contract.

## 8. Async flow

Рекомендуемый production flow, если capabilities содержит `async`:

1. `POST /v1/render/jobs` с полным `RenderRequest`.
2. Получить `202` и `job_id`.
3. Poll `GET /v1/render/jobs/{job_id}`.
4. Завершить polling при `completed`, `failed` или `cancelled`.
5. При `completed` взять `result` — это обычный `RenderResponse`.

Job statuses:

- `pending`;
- `running`;
- `completed`;
- `failed`;
- `cancelled`.

Текущий default `cancel_mode=pending_only`: гарантированно отменяются только `pending` jobs. Для `running` или terminal job `DELETE` возвращает актуальную запись, которая может остаться `running`, `completed` или `failed`.

Submit response:

```json
{
  "job_id": "f74a...",
  "request_id": "8ae65ca5-3a3b-40bf-a1bb-cbb3d3252782",
  "status": "pending",
  "created_at": "2026-06-19T12:00:00+00:00",
  "updated_at": "2026-06-19T12:00:00+00:00",
  "submit_mode": "async"
}
```

Idempotency:

- одинаковый `request_id` для job в состоянии `pending`, `running` или `completed` возвращает существующий `job_id`;
- поэтому retry submit должен повторять тот же `request_id`;
- новый пользовательский render должен получать новый `request_id`.

Текущий job backend in-memory: после рестарта ML-service jobs и результаты теряются. Product queue и durable business state должны оставаться в ShadowGen-v2.

## 9. Ошибки и retry policy

Формат ошибки:

```json
{
  "error": {
    "code": "queue_full",
    "message": "...",
    "details": null,
    "request_id": "..."
  }
}
```

| HTTP | Code | Рекомендация |
|---|---|---|
| `400` | `validation_error` | исправить payload, без автоматического retry |
| `415` | `unsupported_input` | исправить MIME/изображение |
| `422` | `validation_error` | исправить Pydantic schema/ranges |
| `400` | `validation_error` для неизвестного `job_id` | прекратить polling, job отсутствует или сервис был перезапущен |
| `429` | `queue_full` | retry с exponential backoff и jitter |
| `503` | `not_accepting_jobs` | проверить `/health`, retry позже |
| `503` | `async_disabled` | перейти на sync, если capabilities это разрешает |
| `500` | `processing_failed` | ограниченный retry с тем же `request_id`, затем ошибка пользователю |
| `504` | `timeout` | retry с тем же `request_id` для async submit/status flow |

Не повторяйте бесконечно `400`, `415`, `422`.

## 10. Рекомендуемый TypeScript client

```ts
type MlError = {
  error?: {
    code?: string;
    message?: string;
    request_id?: string | null;
  };
};

async function mlRequest<T>(baseUrl: string, path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, init);
  const payload = (await response.json()) as T | MlError;

  if (!response.ok) {
    const failure = payload as MlError;
    throw new Error(failure.error?.message ?? `ML service error ${response.status}`);
  }

  return payload as T;
}

async function submitRender(baseUrl: string, request: RenderRequest) {
  return mlRequest<{ job_id: string; status: string }>(baseUrl, "/v1/render/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}
```

## 11. Checklist миграции frontend/backend

- использовать `shadow.model: "v1-gan" | "v2-diff"`;
- не отправлять внутренние `backend_kind`, `local`, `triton` в public render request;
- отправлять Base64 без data URL prefix;
- генерировать стабильный `request_id` до первого submit и сохранять его на retries;
- выбирать sync/async после `/v1/capabilities`, а не по hardcoded флагу;
- искать результат в `artifacts` по `kind/name`, не по индексу;
- обрабатывать `warnings` как non-fatal;
- отключать controls согласно выбранной shadow model;
- не зависеть от debug artifacts и `/v1/dev/*`;
- не обращаться к ML-service напрямую из браузера в production.

## 12. Связанные документы

- [api.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/api.md) — расширенное описание всех API полей;
- [worker-core-contract.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/worker-core-contract.md) — ответственность worker, concurrency и batching;
- [frontend-shadow-model-contract.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/frontend-shadow-model-contract.md) — UX-тексты и детали выбора shadow model;
- [docker-local.md](/n:/PROJECTS/ML/ShadowGen-ML-core/ShadowGen-ML-service/docs/docker-local.md) — запуск контейнера и выбор GPU.
