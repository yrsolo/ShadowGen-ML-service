# Рекомендуемый стек готовых моделей и pipeline сервиса ShadowGen v2

## 1. Общая схема

Для первой production-версии сервиса рекомендуется использовать следующий фиксированный стек моделей и компонентов:

- **Определение главного объекта:** `IDEA-Research/grounding-dino-base`
- **Оценка параметров камеры:** `GeoCalib`
- **Вырезание объекта / маттинг:** `ZhengPeng7/BiRefNet_lite-matting`
- **Оценка глубины:** `depth-anything/Depth-Anything-V2-Small-hf`
- **Карта нормалей:** вычисляется из карты глубины
- **Генератор тени:** временная заглушка на основе сдвига маски, перспективного искажения и Gaussian blur
- **Слой композиции:** собственный Python-модуль поверх PIL / OpenCV / NumPy

Такой стек позволяет быстро собрать рабочий сервис, сохранить модульность и подготовить инфраструктуру для последующей замены shadow-заглушки на обученную модель.

---

## 2. Модель определения главного объекта

### Выбранная модель
`IDEA-Research/grounding-dino-base` 

### Назначение
Модель используется для определения главного объекта на исходной фотографии до удаления фона.

### Причина выбора
Grounding DINO поддерживается через `transformers` и предназначен для open-set / zero-shot object detection, что позволяет использовать фиксированный prompt вида `"object."` и получать bbox основного объекта без дополнительной captioning-модели. 

### Как использовать
1. Сервис получает исходное изображение.
2. В модель передается изображение и фиксированный текстовый prompt:
   - `"object."`
3. Из всех найденных bbox выбирается основной:
   - по максимальной уверенности;
   - при близких confidence — по наибольшей площади.
4. Bbox расширяется на безопасный padding под будущую тень.
5. Полученный bbox используется как основа для дальнейшего crop.

### Выход модели
- `bbox`
- `detection_confidence`

---

## 3. Модель оценки параметров камеры

### Выбранная модель
`GeoCalib` 

### Назначение
GeoCalib используется для оценки параметров камеры по исходному изображению до вырезания объекта из фона.

### Какие параметры используются
Из результатов GeoCalib сервис использует:
- `fov` / `focal length proxy`
- `pitch`
- `roll`

GeoCalib является моделью single-image camera calibration и оценивает focal length / field of view и ориентацию камеры относительно гравитации, включая pitch и roll. 

### Как использовать
1. Исходное изображение до кропа и удаления фона передается в GeoCalib.
2. Из результата извлекаются:
   - `camera_fov`
   - `camera_pitch`
   - `camera_roll`
3. Эти параметры сохраняются в preprocessing cache и дальше передаются в генератор тени.

### Выход модели
- `camera_fov`
- `camera_pitch`
- `camera_roll`

---

## 4. Модель вырезания объекта и маттинга

### Выбранная модель
`ZhengPeng7/BiRefNet_lite-matting` 

### Назначение
Модель используется для получения точной маски и alpha matte главного объекта.

### Причина выбора
BiRefNet Lite Matting опубликован на Hugging Face, имеет лицензию MIT и предназначен для matting / dichotomous segmentation. Это делает модель удобной для production-встраивания в Python-сервис. 

### Как использовать
1. После детекции объекта исходное изображение кропится по bbox с запасом.
2. Crop передается в BiRefNet Lite Matting.
3. На выходе сервис получает:
   - alpha matte;
   - бинарную маску.
4. Далее выполняется постобработка:
   - largest connected component;
   - удаление мелкого мусора;
   - optional feathering на границе.

### Выход модели
- `mask`
- `alpha`
- `cutout_rgba`

---

## 5. Модель оценки глубины

### Выбранная модель
`depth-anything/Depth-Anything-V2-Small-hf` 

### Назначение
Модель используется для получения monocular depth карты объекта после вырезания из фона.

### Причина выбора
Depth Anything V2 доступен в формате Hugging Face, поддерживается через `transformers` и позиционируется как более быстрый и более легкий depth estimator по сравнению с diffusion-based depth подходами. В model card также отмечается улучшенная детализация depth maps. 

### Как использовать
1. После получения `cutout_rgba` сервис формирует RGB crop объекта.
2. RGB crop передается в Depth Anything V2 Small.
3. На выходе получается относительная depth map.
4. При необходимости depth нормализуется в диапазон `[0, 1]`.
5. Depth сохраняется в preprocessing cache.

### Выход модели
- `depth_map`

---

## 6. Получение карты нормалей

### Выбранный подход
Карта нормалей рассчитывается из карты глубины внутри сервиса.

### Назначение
Нормали используются как дополнительный geometric prior для shadow generation.

### Как использовать
1. Depth map преобразуется в поле градиентов по осям X и Y.
2. По градиентам вычисляются surface normals в camera space.
3. Нормали нормализуются и сохраняются в формате `H x W x 3`.
4. Результат кэшируется вместе с depth map.

### Выход
- `normal_map`

---

## 7. Временный генератор тени

### Выбранный подход
На первом этапе вместо обученной генеративной модели используется детерминированная заглушка.

### Назначение
Обеспечить полностью рабочий production pipeline до подключения обученной shadow model.

### Как работает заглушка
На вход подаются:
- `mask`
- `depth_map`
- `normal_map`
- `camera_fov`
- `camera_pitch`
- `camera_roll`
- пользовательские параметры:
  - `angle_deg`
  - `elevation_deg`
  - `softness`
  - `opacity`
  - `reflection`

Далее выполняются шаги:
1. Бинарная маска объекта проецируется в сторону `angle_deg`.
2. Сдвиг и деформация маски зависят от:
   - `elevation_deg`
   - `camera_fov`
   - `camera_pitch`
3. Формируется черновая shadow mask.
4. Мягкость тени задается Gaussian blur, радиус которого зависит от `softness`.
5. Плотность тени регулируется параметром `opacity`.
6. При необходимости добавляется слабый color tint / reflection component.
7. Полученный shadow layer передается на этап композиции.

### Выход
- `shadow_rgba`

---

## 8. Слой композиции

### Назначение
Собрать итоговое изображение из объекта, тени и фона.

### Как использовать
1. Создается фон, заданный параметрами запроса.
2. На фон накладывается слой тени.
3. Затем накладывается вырезанный объект.
4. Формируется итоговый `final` artifact.
5. При включенном debug возвращаются промежуточные артефакты.

### Выход
- `final_image`
- optional debug artifacts:
  - `cutout`
  - `mask`
  - `depth`
  - `normals`
  - `shadow`

---

## 9. Порядок работы pipeline

### Шаг 1. Decode
- декодирование входного изображения;
- проверка MIME и структуры payload.

### Шаг 2. Camera estimation
- запуск `GeoCalib` на полном исходном изображении;
- получение:
  - `camera_fov`
  - `camera_pitch`
  - `camera_roll`

### Шаг 3. Main object detection
- запуск `grounding-dino-base` на полном изображении;
- получение bbox главного объекта.

### Шаг 4. Crop
- расширение bbox;
- подготовка crop с полями под будущую тень;
- приведение к рабочему формату.

### Шаг 5. Matting
- запуск `BiRefNet_lite-matting`;
- получение `mask`, `alpha`, `cutout_rgba`.

### Шаг 6. Depth estimation
- запуск `Depth Anything V2 Small`;
- получение `depth_map`.

### Шаг 7. Normal map
- вычисление `normal_map` из `depth_map`.

### Шаг 8. Preprocessing cache
- сохранение в кэш:
  - bbox
  - camera params
  - mask
  - alpha
  - cutout
  - depth
  - normals

### Шаг 9. Shadow generation
- запуск shadow stub на основе mask/depth/normals/camera params и параметров запроса.

### Шаг 10. Composition
- сборка финального изображения.

### Шаг 11. Response encoding
- кодирование финального изображения;
- сборка debug artifacts при необходимости;
- возврат JSON-ответа.

---

## 10. Кэширование

### Что кэшируется
По ключу вида:

`sha256(image_bytes) + preprocess_version + working_size`

сохраняются:
- `bbox`
- `camera_fov`
- `camera_pitch`
- `camera_roll`
- `mask`
- `alpha`
- `cutout_rgba`
- `depth_map`
- `normal_map`

### Что не кэшируется
- финальная тень;
- финальный композит.

Это позволяет повторно использовать preprocessing при изменении угла и параметров света.

---

## 11. Реализация в коде

### Рекомендуемый runtime
- `FastAPI`
- `Pydantic v2`
- `transformers`
- `torch`
- `opencv-python`
- `Pillow`
- `numpy`

### Использование моделей

#### Grounding DINO
Загружается через `transformers`:
- `AutoProcessor`
- `AutoModelForZeroShotObjectDetection` 

#### Depth Anything V2
Загружается через `transformers`:
- `AutoImageProcessor`
- `AutoModelForDepthEstimation` 

#### BiRefNet Lite Matting
Загружается из Hugging Face как отдельная matting-модель с кастомным inference wrapper. 

#### GeoCalib
Подключается как отдельный модуль single-image camera calibration. 

---

## 12. Итоговый рекомендуемый стек

### Основной production stack v1
- `IDEA-Research/grounding-dino-base` — детекция главного объекта 
- `GeoCalib` — оценка `camera_fov`, `pitch`, `roll` 
- `ZhengPeng7/BiRefNet_lite-matting` — вырезание объекта и alpha matte 
- `depth-anything/Depth-Anything-V2-Small-hf` — depth map объекта 
- `normal_map from depth` — расчет нормалей в сервисе
- `shadow stub` — временная заглушка генератора тени
- `composer` — финальная композиция на фоне

---

## 13. Роль этого стека в развитии сервиса

Этот набор моделей и компонентов является базовым production pipeline для версии v1 и должен использоваться как стабильная опора для:

- интеграции HTTP API;
- контрактного взаимодействия с клиентом;
- тестирования кэширования;
- замера latency;
- сборки Docker-образа;
- последующего переноса в Triton / model serving;
- последующей замены shadow stub на обученную shadow generation model.

На первом этапе именно этот стек считается основным и обязательным к реализации.