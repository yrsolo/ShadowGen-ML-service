Да. Ниже — компактное, но уже продовое ТЗ на **Python ML service v2**, согласованное с контрактом из файла: сервис должен быть **stateless**, **synchronous per request**, с endpoint’ами `GET /health`, `GET /v1/capabilities`, `POST /v1/render`, machine-readable errors, одной финальной артефактной картинкой и опциональными debug-артефактами. Это прямо соответствует требованиям контракта. 

---

# ТЗ на разработку ML-сервиса ShadowGen v1

## 1. Цель

Разработать продовый ML-сервис, который по одному входному изображению:

* находит главный объект,
* оценивает геометрию съёмки,
* вырезает объект,
* строит вспомогательные карты,
* генерирует тень по явным параметрам света,
* композитит результат на заданный фон,
* возвращает финальное изображение и опциональные debug-артефакты.

Сервис реализуется **как синхронный stateless Python HTTP service**.
В перспективе архитектура должна без существенной переработки переноситься в **Docker** и далее в **Triton / model serving runtime**.

---

## 2. Границы ответственности сервиса

### Сервис делает

* принимает **один** источник изображения на запрос;
* валидирует запрос;
* декодирует изображение;
* запускает полный inference pipeline;
* возвращает финальный результат;
* возвращает debug-артефакты по запросу;
* возвращает metrics и warnings.

### Сервис не делает

* очереди задач;
* хранение job state;
* загрузку в облачные бакеты;
* user management;
* backward compatibility со старым ShadowGEN transport;
* batch inference в v1.

Это совпадает с контрактом: один синхронный запрос, одна финальная артефактная картинка, без оркестрации и без cloud-coupling.  

---

## 3. API-контракт

### Обязательные endpoint’ы

* `GET /health`
* `GET /v1/capabilities`
* `POST /v1/render`

Это обязательное требование контракта. 

### Формат входа

`POST /v1/render` принимает JSON с:

* `request_id`
* `pipeline_version`
* `source`
* `shadow`
* `background`
* `output`

Поддержка полей `angle_deg`, `elevation_deg`, `softness`, `opacity`, `reflection`, `background.mode=solid`, `output.return_debug` обязательна.  

### Формат выхода

Успешный ответ обязан содержать:

* `artifacts`
* `metrics`
* `warnings`
* `model_info`

Должен быть **ровно один** `final` artifact. Debug-артефакты — только если `return_debug=true`. `metrics.total_ms` обязателен всегда. 

### Ошибки

Обязательная machine-readable форма:

```json
{
  "error": {
    "code": "validation_error",
    "message": "..."
  }
}
```

Категории:

* `validation_error`
* `unsupported_input`
* `processing_failed`
* `timeout`
* `internal_error` 

---

## 4. Рабочий пайплайн inference

Ниже — рекомендованный v1 pipeline с учётом твоего описания и контракта.

### Шаг 1. Decode + validate

Вход:

* `source.mime_type`
* `source.image_base64`

Что делаем:

* проверяем MIME;
* декодируем base64;
* читаем изображение;
* проверяем размер, битность, цветовое пространство;
* нормализуем в RGB uint8 / float32.

Выход:

* `input_rgb`
* `decode_ms`

### Шаг 2. Detection / subject localization

Что делаем:

* определяем главный объект на фото;
* строим bbox основного объекта;
* оцениваем confidence;
* вычисляем crop с безопасными полями под будущую тень.

Требование:

* crop должен включать запас по краям;
* целевой внутренний рабочий размер v1: **512×512**.

Правило:

* если confidence низкий, но объект выделен, сервис продолжает работу и пишет warning;
* если объект не найден — `processing_failed` или `validation_error`, в зависимости от причины.

### Шаг 3. Geometry-from-full-image

Этот шаг выполняется **до удаления фона**, как ты и предлагаешь.

Что делаем:

* предсказываем:

  * угол обзора / camera fov
  * tilt / наклон камеры,
  * plane / support geometry proxy.

Почему до вырезания:

* фон и перспектива помогают понять геометрию съёмки.

Выход:

* `camera_geom`
* `geometry_ms`

### Шаг 4. Background removal / cutout

Что делаем:

* строим точную маску объекта;
* вырезаем объект из фона;
* создаём cutout RGBA.

Выход:

* `mask`
* `cutout_rgba`
* `segmentation_ms`

### Шаг 5. Crop + resize + padding

Что делаем:

* кропим основной объект;
* добавляем поля для будущей тени;
* приводим к `512×512`.

Требование:

* паддинги должны зависеть от предполагаемого направления тени;
* но в v1 допустимо делать симметричный padding с коэффициентом запаса.

Выход:

* `crop_rgb_512`
* `crop_mask_512`
* `crop_rgba_512`

### Шаг 6. Depth + normal estimation

Этот шаг выполняется **по cutout / foreground-centered input**, как ты и предлагаешь.

Что делаем:

* строим monocular depth;
* строим normal map;
* optional: contact map / support map.

Выход:

* `depth_512`
* `normals_512`
* `depth_ms`

### Шаг 7. Preprocess cache

Что делаем:

* сохраняем промежуточные результаты в кэш по ключу, зависящему от:

  * хэша входного изображения,
  * версии моделей preprocessing,
  * размера пайплайна.

Кэшируем:

* bbox / crop info
* camera geometry
* mask
* cutout
* depth
* normals

Не кэшируем:

* финальную тень, потому что она зависит от `shadow.*` и `background.*`

### Шаг 8. Shadow generation

Вход:

* `crop_rgb_512`
* `crop_mask_512`
* `depth_512`
* `normals_512`
* `camera_geom`
* явные пользовательские параметры:

  * `angle_deg`
  * `elevation_deg`
  * `softness`
  * `opacity`
  * `reflection`

Что делаем:

* генератор строит shadow/light interaction layer;
* отдельно может строить:

  * `shadow_alpha`
  * `shadow_rgb`
  * `reflection layer`

Выход:

* `shadow_rgba`
* `shadow_ms`

### Шаг 9. Composition

Что делаем:

* создаём requested background;
* композитим объект;
* композитим тень;
* применяем output resize, если запрошен.

Выход:

* `final_image`
* `composition_ms`

### Шаг 10. Encode response

Что делаем:

* кодируем `final` в `png/webp`;
* при `return_debug=true` кодируем debug-артефакты;
* собираем response body.

Выход:

* `encode_ms`
* `total_ms`

---

## 5. Состав debug-артефактов

Если `output.return_debug=true`, сервис должен по возможности возвращать:

* `cutout`
* `mask`
* `crop`
* `depth`
* `normals`
* `shadow`
* `final`

Минимально полезный набор:

* `final`
* `cutout`
* `shadow`
* `mask`

Это хорошо ложится на контракт, где debug-артефакты допускаются как отдельные `artifacts[]`. 

---

## 6. Рекомендуемая внутренняя архитектура Python-сервиса

### HTTP layer

Рекомендуется:

* **FastAPI**
* **Pydantic v2** для строгой валидации схем
* **Uvicorn** для локального и серверного запуска

### Internal modules

Структура:

* `api/`
* `schemas/`
* `pipeline/`
* `models/`
* `cache/`
* `utils/`
* `tests/`

### Pipeline decomposition

Рекомендуемые интерфейсы:

* `Detector`
* `GeometryEstimator`
* `Segmenter`
* `DepthEstimator`
* `NormalEstimator`
* `ShadowGenerator`
* `Composer`
* `ArtifactEncoder`

### Model loading

Модели должны:

* загружаться один раз на старте;
* иметь явную версию;
* быть thread-safe или process-safe;
* не перезагружаться на каждый запрос.

---

## 7. Кэширование

### Цель

Ускорить повторные вызовы для одного и того же изображения при разных параметрах света.

### Что кэшируется

По ключу:
`sha256(image_bytes) + preprocess_model_versions + target_size`

Сохраняем:

* `detected_bbox`
* `camera_geom`
* `mask`
* `cutout_rgba`
* `crop_rgb_512`
* `crop_mask_512`
* `depth_512`
* `normals_512`

### Формат кэша

На первом этапе допустимо:

* filesystem cache
* `joblib` / `pickle` для metadata
* PNG/NPY/NPZ для карт

### Ограничения

* кэш должен быть optional;
* отключение кэша не должно ломать сервис;
* сервис остаётся stateless на уровне API, даже если локальный кэш существует.

---

## 8. Производительность

### Целевые метрики v1

* один запрос за раз;
* рабочий размер: **512×512**;
* целевой `total_ms` — определяется моделью, но должен логироваться всегда;
* `metrics.total_ms` обязателен по контракту. 

### Стадии, для которых желательно возвращать метрики

* `decode_ms`
* `detection_ms`
* `geometry_ms`
* `segmentation_ms`
* `depth_ms`
* `shadow_ms`
* `composition_ms`
* `encode_ms`

---

## 9. Warnings

Сервис должен уметь завершаться успешно, но возвращать warnings.

Примеры:

* `main_object_low_confidence`
* `geometry_estimation_low_confidence`
* `foreground_mask_low_confidence`
* `reflection_clamped`
* `debug_artifact_unavailable`

Warnings обязательны как поле ответа даже если массив пустой. 

---

## 10. Требования к детерминизму

Сервис должен быть “deterministic enough for repeated use with the same input and settings”, как указано в контракте. 

Практически это значит:

* фиксированный inference mode;
* отключённый train mode;
* фиксированные seed там, где это применимо;
* отсутствие скрытых fallback-веток.

---

## 11. Логирование

Структурированные логи должны содержать:

* timestamp
* request_id
* service_version
* model_version
* total_ms
* stage timings
* error code on failure

Нельзя логировать:

* полный base64 payload
* пользовательское изображение целиком в лог

Это соответствует контракту. 

---

## 12. Acceptance criteria для v1

Сервис считается готовым, если:

1. Реализованы:

   * `GET /health`
   * `GET /v1/capabilities`
   * `POST /v1/render`

2. `POST /v1/render`:

   * принимает один input image;
   * принимает `angle_deg` и `elevation_deg`;
   * валидирует `softness`, `opacity`, `reflection`, `background`, `output`.

3. На успехе:

   * всегда возвращается ровно один `final` artifact;
   * всегда возвращается `metrics.total_ms`;
   * всегда возвращаются `warnings`;
   * при `return_debug=true` сервис возвращает debug-артефакты, если они доступны.

4. На ошибке:

   * возвращается machine-readable `error.code` и `error.message`.

5. Pipeline реально выполняет:

   * detection/crop
   * geometry estimation
   * segmentation
   * depth/normals
   * shadow generation
   * composition

6. Есть unit tests на валидацию запроса и integration tests минимум на:

   * JPEG
   * PNG
   * change of `angle_deg`
   * change of `elevation_deg`
   * `return_debug=true`
   * invalid MIME type
   * timeout behavior

Это полностью совпадает с чеклистом контракта. 

---

## 13. Что я бы улучшил в твоей схеме

Твоя логика в целом правильная. Я бы поменял только две вещи.

### 1. Detection и geometry — до маски

С этим я согласен.

### 2. Crop лучше делать в два этапа

Не сразу “вырезали и 512×512”, а:

* сначала rough detect bbox,
* затем geometry on full image,
* затем fine crop + mask + padding,
* потом уже 512×512.

Так меньше шанс потерять контекст, нужный для оценки камеры.

---

## 14. Самая практичная формулировка пайплайна в одну строку

**Input image → detect main object → estimate camera/scene geometry on original image → segment foreground → crop/pad/resize to 512 → estimate depth/normals on cutout → cache preprocess → generate shadow from explicit light params → composite on requested background → return final + debug artifacts**

