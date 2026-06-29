# Project Page Image Prompts

Все изображения сгенерированы встроенным `image_gen` в едином стиле. В изображениях намеренно нет текста: подписи и пояснения управляются в HTML.

## Product runtime

Файл: `assets/product-runtime.png`

```text
Use case: infographic-diagram
Asset type: wide technical illustration for an academic ML project page, 16:9 landscape
Primary request: illustrate the complete ShadowGen product runtime from a smartphone photo to the finished result
Scene/backdrop: warm off-white editorial background with a faint technical grid and generous margins
Subject: far left, a real user photographs an everyday object with a smartphone in a visually varied environment containing books, a plant, cables, textured surfaces and other background objects; upper center, a clearly separated cloud boundary contains only the API entry point, task queue, shared state and object storage; lower center, a clearly separated local boundary contains a desktop computer running the worker and a separate local GPU inference server; the worker pulls a task from the cloud queue and sends it to the GPU server; far right, the result returns to the user's smartphone with the isolated object and a realistic soft contact shadow on a clean target background
Style/medium: premium technical editorial infographic, precise isometric 3D modules mixed with clean diagrammatic flows, sophisticated research project figure, not corporate clipart
Composition/framing: clear end-to-end flow, user and source photo on the left, cloud infrastructure above, local worker and separate GPU server below, final smartphone result on the right
Lighting/mood: soft studio lighting, calm, serious, technically credible
Color palette: warm white, charcoal, muted steel blue, restrained terracotta accents
Materials/textures: matte paper, frosted glass interfaces, brushed dark metal compute modules
Constraints: no words, no labels, no letters, no numbers, no logos, no watermark; do not depict the worker inside the cloud; do not merge the local computer and GPU server; keep GPU server internals visually simple; all modules must be visually distinct; no decorative blobs; readable at website hero size
```

## ML service

Файл: `assets/ml-service.png`

```text
Use case: infographic-diagram
Asset type: wide technical illustration for an academic ML project page, 16:9 landscape
Primary request: illustrate the internal ShadowGen ML inference service as a multi-stage image-processing pipeline
Scene/backdrop: warm off-white editorial background with a faint precise grid and generous margins
Subject: one realistic smartphone product photograph with a visually varied natural background containing a textured desk, books, plant, cables and other objects passes through visually distinct stages: object detection on the busy photo, clean segmentation cutout with transparent background, refined alpha edges, depth map, surface normal map, generated contact shadow, and final composition on a clean neutral target background; include a compact local GPU inference server and stage diagnostics as supporting elements
Style/medium: premium scientific editorial infographic, precise isometric 3D panels and image-processing visualizations, clean and technically credible, suitable for a machine learning project release
Composition/framing: left-to-right pipeline with seven visually recognizable image panels connected by thin directional paths, final result emphasized on the right
Lighting/mood: neutral studio lighting, calm, rigorous
Color palette: warm white, charcoal, muted steel blue, restrained terracotta, depth-map blue-gray, normal-map RGB accents only where technically appropriate
Materials/textures: matte paper panels, subtle glass, dark metal compute module
Constraints: no words, no labels, no letters, no numbers, no logos, no watermark; source image must not be a studio image or use a plain monochrome background; preserve the same object consistently through every stage; keep GPU server internals visually simple; no decorative blobs; each processing stage must visually differ; readable at website hero size
```

## Dataset pipeline

Файл: `assets/dataset-pipeline.png`

```text
Use case: infographic-diagram
Asset type: wide technical illustration for an academic ML project page, 16:9 landscape
Primary request: illustrate the ShadowGen synthetic dataset production workflow for studio object shadows
Scene/backdrop: warm off-white editorial background, subtle technical grid, generous margins
Subject: a curated collection of varied everyday 3D objects enters a filtering and quality-control gate, selected objects move into a Blender-like virtual studio with white cyclorama, camera, softbox and directional light, then a batch renderer produces aligned output passes: RGB product render with soft shadow, normalized depth, surface normals, object mask, and metadata manifest; include a quarantine tray for rejected models
Style/medium: premium scientific editorial infographic, precise isometric 3D modules, sophisticated research project figure, technically credible and visually striking
Composition/framing: horizontal left-to-right data production flow with candidate pool on left, studio rendering in center, aligned dataset passes on right
Lighting/mood: soft studio light, orderly, rigorous
Color palette: warm white, charcoal, muted steel blue, restrained terracotta, technically correct normal-map colors
Materials/textures: matte paper, translucent dataset trays, clean studio surfaces
Constraints: no words, no labels, no letters, no numbers, no logos, no watermark; no decorative blobs; make filtering, studio rendering, and output passes visually clear; readable at website hero size
```

## Training workflow

Файл: `assets/training-workflow.png`

```text
Use case: infographic-diagram
Asset type: wide technical illustration for an academic ML project page, 16:9 landscape
Primary request: illustrate the ShadowGen model training and evaluation workflow
Scene/backdrop: warm off-white editorial background with a faint technical grid and generous margins
Subject: versioned dataset shards containing RGB, depth, normals and masks feed into an experiment configuration module, then into a GPU training workstation showing a diffusion-model style neural network core with lightweight adaptation branches, followed by experiment tracking charts, validation comparisons, evaluation metrics, and an exported model package handed to an inference server
Style/medium: premium scientific editorial infographic, precise isometric 3D modules, clean research project figure, technically credible without being overly abstract
Composition/framing: clear horizontal left-to-right training lifecycle, dataset on left, training compute central and dominant, evaluation and model export on right
Lighting/mood: calm, rigorous, purposeful
Color palette: warm white, charcoal, muted steel blue, restrained terracotta, subtle violet only for diffusion model internals
Materials/textures: matte paper modules, dark brushed metal GPUs, translucent data layers
Constraints: no words, no labels, no letters, no numbers, no logos, no watermark; no decorative blobs; show iteration and evaluation visually using arrows and comparison panels; readable at website hero size
```

## Dataset passes

Файл: `assets/dataset-passes.png`

```text
Use case: infographic-diagram
Asset type: wide companion illustration for a Russian ML project article, 16:9 landscape
Primary request: illustrate the ShadowGen dataset itself, not the infrastructure
Scene/backdrop: warm off-white editorial background with subtle technical grid and generous margins
Subject: a clean Blender-style studio render setup producing a dataset of everyday objects on a white background with soft directed shadows; show rows of rendered sample cards: RGB object on white background with shadow, object mask, normalized depth pass, surface normal pass, and metadata manifest represented visually without text; include hints of ObjaverseXL source 3D models as small varied objects entering the rendering setup
Style/medium: premium scientific editorial infographic, refined isometric 3D, visually consistent with previous ShadowGen diagrams
Composition/framing: left side source 3D models, center Blender studio/camera/lights, right side paired sample passes and masks; make it look like a serious dataset production process
Lighting/mood: soft studio lighting, precise, calm
Color palette: warm white, charcoal, muted steel blue, restrained terracotta, normal-map colors only in normals pass
Constraints: no words, no labels, no letters, no numbers, no logos, no watermark; no decorative blobs; readable on a website; 16:9 landscape
```

## Model training directions

Файл: `assets/model-training-directions.png`

```text
Use case: infographic-diagram
Asset type: wide companion illustration for a Russian ML project article, 16:9 landscape
Primary request: illustrate two ShadowGen model training directions: PatchGAN and Stable Diffusion based shadow generation
Scene/backdrop: warm off-white editorial background with subtle grid, same visual language as existing ShadowGen diagrams
Subject: left half shows PatchGAN training for top-down shadow generation: object mask and top-view object representation enter a compact GAN training loop, output shadow map can rotate around the object; right half shows Stable Diffusion 1.5 adaptation with LoRA, ControlNet-style conditioning and LCM acceleration: real-view object image, control passes, diffusion core, lightweight adaptation branches, faster inference path; outputs show soft shadows from arbitrary perspective with angle and blur controls represented visually as knobs/sliders without text
Style/medium: premium scientific editorial infographic, precise isometric 3D modules, serious ML research figure, not marketing clipart
Composition/framing: split composition with two balanced model tracks, shared dataset inputs at bottom, outputs on the far sides; use arrows and compact modules
Lighting/mood: calm, technical, rigorous
Color palette: warm white, charcoal, muted steel blue, restrained terracotta, subtle violet for diffusion internals
Constraints: no words, no labels, no letters, no numbers, no logos, no watermark; no decorative blobs; make PatchGAN and diffusion tracks visually different; readable on a website; 16:9 landscape
```

## ML service conveyor

Файл: `assets/ml-service-conveyor.png`

```text
Use case: infographic-diagram
Asset type: wide companion illustration for a Russian ML project article, 16:9 landscape
Primary request: illustrate the ShadowGen ML service processing conveyor from a real photo to final composition
Scene/backdrop: warm off-white editorial background, subtle grid, consistent with existing ShadowGen diagrams
Subject: a busy real smartphone photo of an object on a cluttered desk enters a conveyor of ML stages: detection bounding box, matting/foreground extraction with fine alpha edges, depth map, surface normal map, shadow generation, and compositing; show local GPU server as the compute engine below the conveyor; show final output as object on clean background with soft shadow
Style/medium: premium technical editorial infographic, precise isometric panels and image processing cards
Composition/framing: horizontal pipeline with large image cards across the top and GPU server under the middle; final composition emphasized on the right
Lighting/mood: focused, clean, technically credible
Color palette: warm white, charcoal, muted steel blue, restrained terracotta, technical normal-map RGB only where appropriate
Constraints: no words, no labels, no letters, no numbers, no logos, no watermark; source image must be a real cluttered background, not a studio background; no decorative blobs; readable on a website; 16:9 landscape
```

## Apps infrastructure

Файл: `assets/apps-infrastructure.png`

```text
Use case: infographic-diagram
Asset type: wide companion illustration for a Russian ML project article, 16:9 landscape
Primary request: illustrate ShadowGen applications and execution architecture: web app, Telegram app, Yandex Cloud backend, queue, local worker, local GPU ML server
Scene/backdrop: warm off-white editorial background with subtle grid and generous margins
Subject: left side shows user entry points: a browser web app on a laptop and a Telegram-style chat app on a smartphone, both submitting product photos; center top shows Yandex-cloud-like backend components without logos: serverless API, object storage, shared state, and task queue; center bottom shows a local workstation worker pulling tasks from the cloud queue; right side shows a separate local GPU ML server receiving jobs from the worker and returning artifacts; final result appears back in both web and chat interfaces
Style/medium: premium technical editorial infographic, precise isometric modules, serious project architecture figure
Composition/framing: two input apps on the left, cloud backend across the top center, local worker and GPU server separated below, result loop back to apps; clear grouping and arrows
Lighting/mood: calm, operational, technically credible
Color palette: warm white, charcoal, muted steel blue, restrained terracotta
Constraints: no words, no labels, no letters, no numbers, no logos, no watermark; do not use Telegram or Yandex logos; do not place worker inside cloud; do not merge local worker and GPU server; no decorative blobs; readable on a website; 16:9 landscape
```
