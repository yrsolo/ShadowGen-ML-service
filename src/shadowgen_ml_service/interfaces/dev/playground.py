from __future__ import annotations


def render_playground_html() -> str:
    return """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ShadowGen Pipeline Playground</title>
  <style>
    :root {
      --bg-a: #f4efe8;
      --bg-b: #dfe8f6;
      --card: rgba(255,255,255,0.76);
      --card-strong: rgba(255,255,255,0.92);
      --line: rgba(27, 35, 59, 0.12);
      --text: #1a2233;
      --muted: #5d677d;
      --accent: #f07b5a;
      --ok: #2f855a;
      --warn: #b86b22;
      --err: #c53030;
      --preview-width: 320px;
      --pipeline-min-height: 620px;
      --shadow: 0 18px 42px rgba(34, 43, 69, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", sans-serif;
      color: var(--text);
      overflow: auto;
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.94), transparent 28%),
        linear-gradient(135deg, var(--bg-a), var(--bg-b));
    }
    .shell {
      width: calc(100vw - 28px);
      min-height: calc(100vh - 32px);
      height: auto;
      margin: 16px auto;
      display: grid;
      grid-template-rows: auto minmax(var(--pipeline-min-height), 1fr);
      gap: 18px;
    }
    .hero, .stage-card {
      backdrop-filter: blur(18px);
      background: var(--card);
      border: 1px solid rgba(255,255,255,0.52);
      box-shadow: var(--shadow);
    }
    .hero { border-radius: 30px; padding: 18px 24px; }
    h1 { margin: 0 0 6px; font-size: clamp(24px, 3vw, 34px); letter-spacing: -0.04em; }
    p { margin: 0; color: var(--muted); }
    .hero-grid {
      margin-top: 12px;
      display: grid;
      grid-template-columns: minmax(220px, 3fr) minmax(300px, 4fr) minmax(220px, 3fr) minmax(220px, 2fr);
      gap: 12px;
    }
    .panel {
      background: var(--card-strong);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 12px 16px;
      min-width: 0;
    }
    .panel h2 {
      margin: 0 0 8px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
    }
    .field { display: grid; gap: 4px; margin-bottom: 8px; }
    .field label { font-size: 12px; color: var(--muted); }
    .field input, .field select {
      width: 100%;
      padding: 8px 12px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: white;
      color: var(--text);
    }
    .field input[type="range"] {
      padding: 0;
      accent-color: var(--accent);
      border: 0;
      background: transparent;
    }
    .value-row { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-end; }
    button {
      border: 0;
      border-radius: 999px;
      padding: 10px 18px;
      cursor: pointer;
      font-weight: 600;
    }
    .primary { background: linear-gradient(135deg, var(--accent), #f79c7f); color: white; }
    .ghost { background: rgba(255,255,255,0.9); color: var(--text); border: 1px solid var(--line); }
    .note {
      margin-top: 0;
      padding: 12px 14px;
      border-radius: 18px;
      background: rgba(27, 35, 59, 0.06);
      color: var(--muted);
      white-space: pre-wrap;
      max-height: 96px;
      overflow: auto;
    }
    .pipeline {
      display: grid;
      grid-auto-flow: column;
      grid-auto-columns: minmax(760px, calc(100vw - 64px));
      gap: 16px;
      overflow-x: auto;
      overflow-y: hidden;
      padding: 4px 4px 14px;
      scroll-snap-type: x proximity;
      overscroll-behavior-x: contain;
      align-items: stretch;
      min-height: var(--pipeline-min-height);
    }
    .pipeline::-webkit-scrollbar { height: 12px; }
    .pipeline::-webkit-scrollbar-track { background: rgba(255,255,255,0.3); border-radius: 999px; }
    .pipeline::-webkit-scrollbar-thumb { background: rgba(27, 35, 59, 0.18); border-radius: 999px; }
    .stage-card {
      border-radius: 26px;
      padding: 18px;
      display: grid;
      grid-template-rows: auto auto;
      gap: 18px;
      min-height: 100%;
      max-height: 100%;
      overflow-x: hidden;
      overflow-y: auto;
      overscroll-behavior: contain;
      scroll-snap-align: start;
    }
    .stage-card::-webkit-scrollbar { width: 10px; }
    .stage-card::-webkit-scrollbar-track { background: rgba(255,255,255,0.25); border-radius: 999px; }
    .stage-card::-webkit-scrollbar-thumb { background: rgba(27, 35, 59, 0.14); border-radius: 999px; }
    .stage-main { display: grid; gap: 12px; }
    .stage-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
    .stage-title { display: flex; gap: 12px; align-items: center; }
    .badge {
      width: 42px; height: 42px; border-radius: 16px; display: grid; place-items: center;
      background: linear-gradient(135deg, rgba(240,123,90,0.16), rgba(255,184,102,0.2));
      color: #8f4b31; font-weight: 700;
    }
    .stage-title h3 { margin: 0; font-size: 24px; letter-spacing: -0.02em; }
    .stage-title p { margin-top: 4px; font-size: 14px; }
    .status {
      padding: 8px 12px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.08em;
      font-size: 12px; font-weight: 700; background: rgba(93,103,125,0.12); color: var(--muted);
    }
    .status.completed { background: rgba(47,133,90,0.14); color: var(--ok); }
    .status.failed { background: rgba(197,48,48,0.12); color: var(--err); }
    .status.running { background: rgba(240,123,90,0.14); color: var(--warn); }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 14px; }
    .mode-toggle {
      display: inline-flex; gap: 4px; border: 1px solid var(--line); background: rgba(255,255,255,0.9);
      border-radius: 999px; padding: 4px;
    }
    .mode-toggle button { background: transparent; color: var(--muted); box-shadow: none; padding: 8px 12px; }
    .mode-toggle button.active { background: #1e2638; color: white; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .chip {
      padding: 8px 12px; border-radius: 999px; background: rgba(27,35,59,0.06); color: var(--muted); font-size: 12px;
    }
    .detail-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 8px; margin-top: 12px;
    }
    .detail {
      padding: 10px 12px; border-radius: 18px; background: rgba(255,255,255,0.84); border: 1px solid var(--line);
    }
    .detail small {
      display: block; color: var(--muted); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px;
    }
    .detail strong { font-size: 15px; }
    .notice, .error {
      margin-top: 12px; padding: 12px 14px; border-radius: 18px; white-space: pre-wrap; font-size: 13px;
    }
    .notice { background: rgba(27,35,59,0.06); color: var(--muted); }
    .notice.ok { background: rgba(47,133,90,0.12); color: var(--ok); }
    .notice.warn { background: rgba(240,123,90,0.12); color: #9f4f26; }
    .error { background: rgba(197,48,48,0.09); color: var(--err); }
    .previews {
      display: grid;
      gap: 12px;
      grid-auto-flow: column;
      grid-auto-columns: minmax(220px, min(var(--preview-width), 36vw));
      align-content: start;
      overflow-x: auto;
      overflow-y: visible;
      padding-bottom: 6px;
    }
    .previews::-webkit-scrollbar { height: 10px; }
    .previews::-webkit-scrollbar-track { background: rgba(255,255,255,0.35); border-radius: 999px; }
    .previews::-webkit-scrollbar-thumb { background: rgba(27, 35, 59, 0.14); border-radius: 999px; }
    .preview {
      background: rgba(255,255,255,0.88);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 12px;
      min-height: 100%;
    }
    .preview strong {
      display: block; margin-bottom: 8px; color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em;
    }
    .preview img {
      width: 100%; display: block; border-radius: 14px;
      max-height: none;
      object-fit: contain;
      background:
        linear-gradient(45deg, rgba(0,0,0,0.04) 25%, transparent 25%, transparent 75%, rgba(0,0,0,0.04) 75%),
        linear-gradient(45deg, rgba(0,0,0,0.04) 25%, transparent 25%, transparent 75%, rgba(0,0,0,0.04) 75%);
      background-size: 18px 18px;
      background-position: 0 0, 9px 9px;
    }
    @media (max-width: 1100px) {
      :root { --pipeline-min-height: 560px; }
      .hero-grid { grid-template-columns: repeat(12, 1fr); }
      .panel, .panel.wide { grid-column: span 12; }
      .shell { width: calc(100vw - 20px); min-height: calc(100vh - 20px); margin: 10px auto; }
      .pipeline { grid-auto-columns: minmax(560px, calc(100vw - 28px)); }
    }
    @media (max-width: 720px) {
      :root { --pipeline-min-height: 0px; }
      .shell { min-height: 0; }
      .pipeline {
        grid-auto-flow: row;
        grid-auto-columns: unset;
        min-height: 0;
        overflow-x: visible;
        overflow-y: visible;
      }
      .stage-card { min-height: auto; max-height: none; overflow: visible; }
      .previews {
        grid-auto-flow: row;
        grid-auto-columns: unset;
        overflow: visible;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>ShadowGen Pipeline Playground</h1>
      <p>Локальный отладчик pipeline: загрузка изображения, per-stage rerun, выбор execution backend и model variant без изменения HTTP paths.</p>
      <div class="hero-grid">
        <div class="panel">
          <h2>Source</h2>
          <div class="field">
            <label for="imageFile">Изображение</label>
            <input id="imageFile" type="file" accept="image/png,image/jpeg,image/webp" />
          </div>
          <div class="field">
            <label for="requestId">Request ID</label>
            <input id="requestId" type="text" value="playground-request" />
          </div>
          <div class="field">
            <label for="paddingPx">Padding</label>
            <input id="paddingPx" type="range" min="0" max="400" value="100" />
            <div class="value-row"><span>поля вокруг объекта</span><span id="paddingPxValue">100 px</span></div>
          </div>
        </div>
        <div class="panel">
          <h2>Shadow</h2>
          <div class="field"><label>Angle</label><input id="angleDeg" type="range" min="0" max="360" value="45" /><div class="value-row"><span>azimuth</span><span id="angleDegValue">45°</span></div></div>
          <div class="field"><label>Elevation</label><input id="elevationDeg" type="range" min="0" max="90" value="35" /><div class="value-row"><span>light height</span><span id="elevationDegValue">35°</span></div></div>
          <div class="field"><label>Softness</label><input id="softness" type="range" min="0" max="1" step="0.01" value="0.5" /><div class="value-row"><span>model input</span><span id="softnessValue">0.50</span></div></div>
          <div class="field"><label>Opacity</label><input id="opacity" type="range" min="0" max="1" step="0.01" value="0.65" /><div class="value-row"><span>density</span><span id="opacityValue">0.65</span></div></div>
          <div class="field"><label>Reflection</label><input id="reflection" type="range" min="0" max="1" step="0.01" value="0.10" /><div class="value-row"><span>reserved input</span><span id="reflectionValue">0.10</span></div></div>
          <div class="note" id="shadowControlNote"></div>
        </div>
        <div class="panel">
          <h2>Output</h2>
          <div class="field"><label for="backgroundColor">Background</label><input id="backgroundColor" type="color" value="#ffffff" /></div>
          <div class="field"><label for="outputFormat">Format</label><select id="outputFormat"><option value="png">png</option><option value="webp">webp</option></select></div>
          <div class="field"><label for="returnDebug">Debug artifacts</label><select id="returnDebug"><option value="true">true</option><option value="false">false</option></select></div>
          <div class="field">
            <label for="previewSize">Preview size</label>
            <input id="previewSize" type="range" min="180" max="520" value="320" />
            <div class="value-row"><span>global width</span><span id="previewSizeValue">320 px</span></div>
          </div>
          <div class="actions">
            <button class="primary" id="runAllBtn">Запустить всё</button>
            <button class="ghost" id="clearBtn">Очистить</button>
          </div>
        </div>
        <div class="panel wide">
          <h2>Runtime</h2>
          <div class="note" id="globalNote">Сервис готов. Загрузите изображение и запускайте pipeline.</div>
        </div>
      </div>
    </section>

    <section class="pipeline" id="pipeline"></section>
  </div>

  <script>
    const stageDefinitions = [
      { key: "decode", title: "Decode", description: "Декодирование входного изображения.", backendKinds: ["internal"], variants: ["internal"] },
      { key: "geometry_estimator", title: "Geometry", description: "Оценка геометрии сцены по исходному изображению.", backendKinds: ["mock", "local"], variants: ["geocalib"] },
      { key: "detector", title: "Detection", description: "Поиск основного объекта на полном изображении.", backendKinds: ["mock", "local", "triton"], variants: ["grounding-dino"] },
      { key: "segmenter", title: "Segmentation", description: "Сегментация после crop/pad/resize.", backendKinds: ["mock", "local", "triton"], variants: ["birefnet"] },
      { key: "foreground_refiner", title: "Foreground", description: "Коррекция цвета полупрозрачных пикселей.", backendKinds: ["mock", "local"], variants: ["fast-foreground-estimation"] },
      { key: "depth_estimator", title: "Depth", description: "Карта глубины на working crop.", backendKinds: ["mock", "local", "triton"], variants: ["depth-anything-v2-small"] },
      { key: "normal_estimator", title: "Normals", description: "Normals через neural backend или fallback от depth.", backendKinds: ["mock", "local", "triton"], variants: ["stable-normal"] },
      { key: "shadow_generator", title: "Shadow", description: "Тень через mock, V1-GAN или Triton-ready V2-DIFF.", backendKinds: ["mock", "local", "triton"], variants: ["mock", "v1-gan", "v2-diff"] },
      { key: "composer", title: "Composition", description: "Композиция объекта и тени на фоне.", backendKinds: ["mock", "local"], variants: ["python-composer"] }
    ];

    const pipelineEl = document.getElementById("pipeline");
    const globalNote = document.getElementById("globalNote");
    const imageFile = document.getElementById("imageFile");
    const previewSize = document.getElementById("previewSize");

    const state = {
      imageBase64: null,
      mimeType: null,
      capabilities: {},
      stages: {},
      stageBackendKinds: {
        geometry_estimator: "local",
        detector: "local",
        segmenter: "local",
        foreground_refiner: "local",
        depth_estimator: "local",
        normal_estimator: "local",
        shadow_generator: "local",
        composer: "local"
      },
      stageVariants: {
        geometry_estimator: "geocalib",
        detector: "grounding-dino",
        segmenter: "birefnet",
        foreground_refiner: "fast-foreground-estimation",
        depth_estimator: "depth-anything-v2-small",
        normal_estimator: "stable-normal",
        shadow_generator: "v1-gan",
        composer: "python-composer"
      }
    };

    function bindRange(id, formatter) {
      const input = document.getElementById(id);
      const output = document.getElementById(id + "Value");
      const render = () => { output.textContent = formatter(input.value); };
      input.addEventListener("input", render);
      render();
    }

    bindRange("paddingPx", (value) => `${value} px`);
    bindRange("angleDeg", (value) => `${value}°`);
    bindRange("elevationDeg", (value) => `${value}°`);
    bindRange("softness", (value) => Number(value).toFixed(2));
    bindRange("opacity", (value) => Number(value).toFixed(2));
    bindRange("reflection", (value) => Number(value).toFixed(2));
    bindRange("previewSize", (value) => `${value} px`);

    previewSize.addEventListener("input", () => {
      document.documentElement.style.setProperty("--preview-width", `${previewSize.value}px`);
    });
    document.documentElement.style.setProperty("--preview-width", `${previewSize.value}px`);
    pipelineEl.addEventListener("wheel", (event) => {
      if (window.innerWidth <= 720) return;
      if (event.shiftKey) {
        const target = event.target instanceof Element ? event.target : null;
        const card = target ? target.closest(".stage-card") : null;
        if (!card) return;
        event.preventDefault();
        card.scrollBy({ top: event.deltaY, behavior: "auto" });
        return;
      }
      if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
      event.preventDefault();
      pipelineEl.scrollBy({ left: event.deltaY, behavior: "auto" });
    }, { passive: false });

    imageFile.addEventListener("change", async (event) => {
      const file = event.target.files?.[0];
      if (!file) return;
      state.mimeType = file.type || "image/png";
      state.imageBase64 = await fileToBase64(file);
      globalNote.textContent = `Загружено: ${file.name}. Можно запускать pipeline.`;
    });

    document.getElementById("runAllBtn").addEventListener("click", () => runPipeline());
    document.getElementById("clearBtn").addEventListener("click", clearResults);

    function renderCards() {
      pipelineEl.innerHTML = "";
      stageDefinitions.forEach((stage, index) => {
        const stageState = state.stages[stage.key] || {};
        const capability = state.capabilities[stage.key] || null;
        const card = document.createElement("article");
        card.className = "stage-card";
        card.innerHTML = `
          <div class="stage-main">
            <div class="stage-head">
              <div class="stage-title">
                <div class="badge">${String(index + 1).padStart(2, "0")}</div>
                <div>
                  <h3>${stage.title}</h3>
                  <p>${stage.description}</p>
                </div>
              </div>
              <div class="status ${stageState.status || ""}">${stageState.status || "idle"}</div>
            </div>
            <div class="controls">
              ${renderBackendKindSelector(stage)}
              ${renderVariantSelector(stage)}
              ${stage.key !== "decode" ? `<button class="ghost" data-rerun="${stage.key}">Перезапустить этап</button>` : ""}
            </div>
            <div class="chips">
              <div class="chip">requested: ${requestedLabel(stage.key)}</div>
              <div class="chip">actual: ${stageState.actual_mode || "n/a"}</div>
              <div class="chip">variant: ${stageState.model_variant || state.stageVariants[stage.key] || "n/a"}</div>
              <div class="chip">time: ${stageState.elapsed_ms != null ? `${stageState.elapsed_ms} ms` : "n/a"}</div>
              ${stageState.device ? `<div class="chip">device: ${stageState.device}</div>` : ""}
              ${stageState.endpoint ? `<div class="chip">endpoint: ${stageState.endpoint}</div>` : ""}
              ${stageState.cache_status ? `<div class="chip">cache: ${stageState.cache_status}</div>` : ""}
            </div>
            ${renderStageNotice(stage, stageState, capability)}
            <div class="detail-grid">${renderDetails(stageState)}</div>
            ${stageState.error ? `<div class="error">${stageState.error}</div>` : ""}
          </div>
          <div class="previews">${renderPreviews(stageState.previews || [])}</div>
        `;
        pipelineEl.appendChild(card);
      });

      pipelineEl.querySelectorAll("[data-backend-kind]").forEach((button) => {
        button.addEventListener("click", () => {
          state.stageBackendKinds[button.dataset.stage] = button.dataset.backendKind;
          if (button.dataset.stage === "shadow_generator") {
            syncShadowModelSelection("backend");
          }
          renderCards();
          updateShadowControlNote();
        });
      });
      pipelineEl.querySelectorAll("[data-variant]").forEach((button) => {
        button.addEventListener("click", () => {
          state.stageVariants[button.dataset.stage] = button.dataset.variant;
          if (button.dataset.stage === "shadow_generator") {
            syncShadowModelSelection("variant");
          }
          renderCards();
          updateShadowControlNote();
        });
      });
      pipelineEl.querySelectorAll("[data-rerun]").forEach((button) => {
        button.addEventListener("click", () => runPipeline(button.dataset.rerun));
      });
    }

    function renderBackendKindSelector(stage) {
      if (stage.key === "decode") return "";
      return `
        <div class="mode-toggle">
          ${stage.backendKinds.map((backendKind) => `
            <button type="button" data-stage="${stage.key}" data-backend-kind="${backendKind}" class="${state.stageBackendKinds[stage.key] === backendKind ? "active" : ""}">${backendKind}</button>
          `).join("")}
        </div>
      `;
    }

    function renderVariantSelector(stage) {
      if (stage.variants.length <= 1 || stage.key === "decode") return "";
      return `
        <div class="mode-toggle">
          ${stage.variants.map((variant) => `
            <button type="button" data-stage="${stage.key}" data-variant="${variant}" class="${state.stageVariants[stage.key] === variant ? "active" : ""}">${variant}</button>
          `).join("")}
        </div>
      `;
    }

    function renderStageNotice(stage, stageState, capability) {
      if (stageState.error) return "";
      if (stageState.fallback_reason) {
        return `<div class="notice warn">fallback: ${stageState.fallback_reason}</div>`;
      }
      if (stageState.actual_backend_kind === "triton") {
        return `<div class="notice ok">Этап исполнился через Triton backend.</div>`;
      }
      if (stageState.actual_backend_kind === "local") {
        return `<div class="notice ok">Этап исполнился через local backend.</div>`;
      }
      if (stageState.actual_backend_kind === "mock") {
        return `<div class="notice warn">Этап исполнился через mock backend.</div>`;
      }
      if (capability && capability.fallback_reason) {
        return `<div class="notice warn">${capability.fallback_reason}</div>`;
      }
      return "";
    }

    function renderDetails(stageState) {
      const details = stageState.details || {};
      const items = [
        ["backend_kind", stageState.actual_backend_kind || null],
        ["model_name", stageState.model_name || null],
        ["model_version", stageState.model_version || null],
        ["device", stageState.device || null],
        ["endpoint", stageState.endpoint || null],
        ...Object.entries(details)
      ].filter(([, value]) => value !== null && value !== undefined && value !== "");

      return items.map(([key, value]) => `
        <div class="detail">
          <small>${String(key).replaceAll("_", " ")}</small>
          <strong>${typeof value === "number" ? Number(value).toFixed(String(key).includes("confidence") ? 3 : 2) : value}</strong>
        </div>
      `).join("");
    }

    function renderPreviews(previews) {
      if (!previews.length) {
        return `<div class="preview"><strong>preview</strong><div style="color:var(--muted);font-size:14px;">Нет данных для показа.</div></div>`;
      }
      return previews.map((preview) => `
        <div class="preview">
          <strong>${preview.name}</strong>
          <img alt="${preview.name}" src="data:${preview.mime_type};base64,${preview.image_base64}" />
        </div>
      `).join("");
    }

    function requestedLabel(stageKey) {
      const kind = state.stageBackendKinds[stageKey];
      const variant = state.stageVariants[stageKey];
      if (!kind) return "internal";
      return variant ? `${kind}/${variant}` : kind;
    }

    async function runPipeline(stopAfter = null) {
      if (!state.imageBase64 || !state.mimeType) {
        globalNote.textContent = "Сначала загрузите изображение.";
        return;
      }
      globalNote.textContent = stopAfter ? `Запускаю этап ${stopAfter}...` : "Запускаю весь pipeline...";
      if (stopAfter) {
        state.stages[stopAfter] = { status: "running" };
      } else {
        stageDefinitions.forEach((stage) => { state.stages[stage.key] = { status: "running" }; });
      }
      renderCards();

      const body = buildRequestBody();
      const endpoint = stopAfter ? `/v1/dev/pipeline/run-stage/${stopAfter}` : "/v1/dev/pipeline/run-all";
      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload?.error?.message || "Pipeline request failed");
        }
        payload.stages.forEach((stage) => { state.stages[stage.stage_key] = stage; });
        stageDefinitions.forEach((stage) => {
          if (!state.stages[stage.key]) state.stages[stage.key] = { status: "skipped" };
        });
        const failed = payload.stages.find((stage) => stage.status === "failed");
        globalNote.textContent = failed
          ? `Этап ${failed.title} завершился с ошибкой: ${failed.error}`
          : `Готово. Выполнено этапов: ${payload.stages.length}. Предупреждения: ${payload.warnings.join(", ") || "нет"}.`;
      } catch (error) {
        globalNote.textContent = `Ошибка запуска: ${error.message}`;
      }
      renderCards();
      updateShadowControlNote();
    }

    function updateShadowControlNote() {
      const variant = state.stageVariants.shadow_generator;
      const note = document.getElementById("shadowControlNote");
      if (!note) return;
      if (variant === "v2-diff") {
        note.textContent = "V2-DIFF сейчас рисует тень без настроек: angle/elevation/softness/reflection игнорируются. Для управляемого rot используйте V1-GAN.";
      } else if (variant === "v1-gan") {
        note.textContent = "V1-GAN использует angle/rot для тени на виде сверху. Elevation, softness и reflection оставлены для будущей модели.";
      } else {
        note.textContent = "Mock остаётся отладочной моделью: здесь грубая softness-логика допустима только для быстрого preview.";
      }
    }

    function syncShadowModelSelection(source) {
      if (source === "backend") {
        const backendKind = state.stageBackendKinds.shadow_generator;
        if (backendKind === "mock") state.stageVariants.shadow_generator = "mock";
        if (backendKind === "local" && state.stageVariants.shadow_generator !== "v2-diff") state.stageVariants.shadow_generator = "v1-gan";
        if (backendKind === "triton") state.stageVariants.shadow_generator = "v2-diff";
        return;
      }
      const variant = state.stageVariants.shadow_generator;
      if (variant === "mock") state.stageBackendKinds.shadow_generator = "mock";
      if (variant === "v1-gan") state.stageBackendKinds.shadow_generator = "local";
      if (variant === "v2-diff") state.stageBackendKinds.shadow_generator = "local";
    }

    function buildRequestBody() {
      const stageModes = {};
      Object.entries(state.stageBackendKinds).forEach(([stageKey, backendKind]) => {
        if (stageKey === "shadow_generator") {
          stageModes[stageKey] = state.stageVariants[stageKey] === "mock" ? "mock" : state.stageVariants[stageKey];
        } else {
          stageModes[stageKey] = backendKind === "mock" ? "mock" : "real";
        }
      });

      return {
        render_request: {
          request_id: document.getElementById("requestId").value || null,
          pipeline_version: "ml-shadowgen-v1",
          source: { mime_type: state.mimeType, image_base64: state.imageBase64 },
          preprocess: { padding_px: Number(document.getElementById("paddingPx").value) },
          shadow: {
            angle_deg: Number(document.getElementById("angleDeg").value),
            elevation_deg: Number(document.getElementById("elevationDeg").value),
            softness: Number(document.getElementById("softness").value),
            opacity: Number(document.getElementById("opacity").value),
            reflection: Number(document.getElementById("reflection").value)
          },
          background: { mode: "solid", color_hex: document.getElementById("backgroundColor").value.toUpperCase() },
          output: {
            format: document.getElementById("outputFormat").value,
            width: null,
            height: null,
            return_debug: document.getElementById("returnDebug").value === "true"
          }
        },
        stage_modes: stageModes,
        stage_backend_kinds: state.stageBackendKinds,
        stage_variants: state.stageVariants
      };
    }

    function clearResults() {
      state.stages = {};
      globalNote.textContent = "Результаты очищены.";
      renderCards();
      updateShadowControlNote();
    }

    function fileToBase64(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || "").split(",")[1]);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    }

    async function loadCapabilities() {
      try {
        const response = await fetch("/v1/capabilities");
        if (!response.ok) throw new Error("capabilities request failed");
        const payload = await response.json();
        state.capabilities = Object.fromEntries((payload.components || []).map((component) => [component.name, component]));
        globalNote.textContent = `runtime: ${payload.active_backend_mode}; async: ${payload.async_enabled}; default backend: ${payload.execution_default_backend}`;
      } catch (error) {
        globalNote.textContent = `Не удалось загрузить runtime capabilities: ${error.message}`;
      }
      renderCards();
      updateShadowControlNote();
    }

    loadCapabilities();
  </script>
</body>
</html>
"""
