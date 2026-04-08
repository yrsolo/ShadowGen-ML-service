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
      --bg-a: #f5efe6;
      --bg-b: #dbe6f6;
      --card: rgba(255,255,255,0.72);
      --card-strong: rgba(255,255,255,0.9);
      --line: rgba(34,43,69,0.12);
      --text: #1d2433;
      --muted: #5a6478;
      --accent: #ff7a59;
      --accent-soft: #ffd7cd;
      --ok: #2f855a;
      --err: #c53030;
      --shadow: 0 18px 48px rgba(43, 52, 69, 0.12);
      --preview-width: 320px;
      --radius: 26px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", "Inter", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.9), transparent 30%),
        linear-gradient(135deg, var(--bg-a), var(--bg-b));
    }
    .shell {
      width: min(1440px, calc(100vw - 32px));
      margin: 20px auto 48px;
    }
    .hero, .stage-card {
      backdrop-filter: blur(20px);
      background: var(--card);
      border: 1px solid rgba(255,255,255,0.52);
      box-shadow: var(--shadow);
    }
    .hero {
      padding: 28px;
      border-radius: 32px;
    }
    .hero h1 {
      margin: 0 0 10px;
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1;
      letter-spacing: -0.04em;
    }
    .hero p {
      margin: 0 0 18px;
      color: var(--muted);
      max-width: 820px;
    }
    .hero-grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 14px;
    }
    .panel {
      background: var(--card-strong);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 16px;
    }
    .panel.span-4 { grid-column: span 4; }
    .panel.span-3 { grid-column: span 3; }
    .panel.span-2 { grid-column: span 2; }
    .panel.span-6 { grid-column: span 6; }
    .panel h2 {
      margin: 0 0 12px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
    }
    .field { display: grid; gap: 6px; margin-bottom: 10px; }
    .field label { font-size: 13px; color: var(--muted); }
    .field input[type="range"] { width: 100%; accent-color: var(--accent); }
    .field input[type="file"],
    .field input[type="text"],
    .field input[type="number"],
    .field input[type="color"],
    .field select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      background: white;
      color: var(--text);
    }
    .value-row {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: var(--muted);
    }
    .actions {
      display: flex;
      align-items: flex-end;
      gap: 12px;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font-weight: 600;
      cursor: pointer;
      transition: transform .18s ease, box-shadow .18s ease;
    }
    button:hover { transform: translateY(-1px); }
    .primary {
      background: linear-gradient(135deg, var(--accent), #ff9d80);
      color: white;
      box-shadow: 0 12px 24px rgba(255, 122, 89, 0.25);
    }
    .ghost {
      background: rgba(255,255,255,0.85);
      color: var(--text);
      border: 1px solid var(--line);
    }
    .pipeline {
      margin-top: 20px;
      display: grid;
      gap: 14px;
    }
    .stage-card {
      border-radius: var(--radius);
      padding: 18px;
      display: grid;
      grid-template-columns: minmax(280px, 0.9fr) minmax(360px, 1.1fr);
      gap: 20px;
      align-items: start;
    }
    .stage-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }
    .stage-title {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .badge {
      min-width: 42px;
      height: 42px;
      display: inline-grid;
      place-items: center;
      border-radius: 16px;
      background: linear-gradient(135deg, rgba(255,122,89,0.16), rgba(255,180,102,0.22));
      color: #8e492f;
      font-weight: 700;
    }
    .stage-title h3 { margin: 0; font-size: 22px; }
    .stage-title p { margin: 4px 0 0; color: var(--muted); font-size: 14px; }
    .status-pill {
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: rgba(90,100,120,0.12);
      color: var(--muted);
    }
    .status-pill.completed { background: rgba(47,133,90,0.14); color: var(--ok); }
    .status-pill.failed { background: rgba(197,48,48,0.12); color: var(--err); }
    .status-pill.running { background: rgba(255,122,89,0.16); color: #b34d2e; }
    .stage-controls {
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    .mode-toggle {
      display: inline-flex;
      background: rgba(255,255,255,0.9);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px;
      gap: 4px;
    }
    .mode-toggle button {
      padding: 8px 12px;
      background: transparent;
      color: var(--muted);
      box-shadow: none;
    }
    .mode-toggle button.active {
      background: #1d2433;
      color: white;
    }
    .meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 8px;
      margin-top: 12px;
    }
    .detail-card {
      border-radius: 18px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.82);
      border: 1px solid var(--line);
    }
    .detail-card small {
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }
    .detail-card strong {
      font-size: 15px;
      color: var(--text);
    }
    .chip {
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(29,36,51,0.06);
      color: var(--muted);
      font-size: 12px;
    }
    .notice {
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 18px;
      background: rgba(29,36,51,0.06);
      color: var(--muted);
      white-space: pre-wrap;
      font-size: 13px;
    }
    .notice.warn {
      background: rgba(255,122,89,0.14);
      color: #9f4428;
    }
    .notice.ok {
      background: rgba(47,133,90,0.12);
      color: var(--ok);
    }
    .error {
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 18px;
      background: rgba(197,48,48,0.08);
      color: var(--err);
      display: none;
      white-space: pre-wrap;
    }
    .previews {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(min(var(--preview-width), 100%), 1fr));
    }
    .preview {
      background: rgba(255,255,255,0.86);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 12px;
    }
    .preview strong {
      display: block;
      margin-bottom: 8px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .preview img {
      width: 100%;
      display: block;
      border-radius: 14px;
      background:
        linear-gradient(45deg, rgba(0,0,0,0.04) 25%, transparent 25%, transparent 75%, rgba(0,0,0,0.04) 75%),
        linear-gradient(45deg, rgba(0,0,0,0.04) 25%, transparent 25%, transparent 75%, rgba(0,0,0,0.04) 75%);
      background-size: 18px 18px;
      background-position: 0 0, 9px 9px;
    }
    .global-note {
      margin-top: 14px;
      padding: 12px 16px;
      border-radius: 18px;
      background: rgba(29,36,51,0.06);
      color: var(--muted);
      font-size: 14px;
    }
    @media (max-width: 1100px) {
      .panel.span-4, .panel.span-3, .panel.span-2, .panel.span-6 { grid-column: span 12; }
      .stage-card { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>ShadowGen Pipeline Playground</h1>
      <p>Загружайте изображение, крутите параметры контракта и прогоняйте весь pipeline или отдельные этапы. Каждый модуль можно переключать между <strong>mock</strong> и <strong>real</strong> режимом.</p>
      <div class="hero-grid">
        <div class="panel span-4">
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
        <div class="panel span-4">
          <h2>Shadow</h2>
          <div class="field"><label>Angle</label><input id="angleDeg" type="range" min="0" max="360" value="45" /><div class="value-row"><span>direction</span><span id="angleDegValue">45°</span></div></div>
          <div class="field"><label>Elevation</label><input id="elevationDeg" type="range" min="0" max="90" value="35" /><div class="value-row"><span>height</span><span id="elevationDegValue">35°</span></div></div>
          <div class="field"><label>Softness</label><input id="softness" type="range" min="0" max="1" step="0.01" value="0.5" /><div class="value-row"><span>blur</span><span id="softnessValue">0.50</span></div></div>
          <div class="field"><label>Opacity</label><input id="opacity" type="range" min="0" max="1" step="0.01" value="0.65" /><div class="value-row"><span>density</span><span id="opacityValue">0.65</span></div></div>
          <div class="field"><label>Reflection</label><input id="reflection" type="range" min="0" max="1" step="0.01" value="0.10" /><div class="value-row"><span>gloss</span><span id="reflectionValue">0.10</span></div></div>
        </div>
        <div class="panel span-4">
          <h2>Output</h2>
          <div class="field">
            <label for="backgroundColor">Background color</label>
            <input id="backgroundColor" type="color" value="#ffffff" />
          </div>
          <div class="field">
            <label for="outputFormat">Format</label>
            <select id="outputFormat"><option value="png">png</option><option value="webp">webp</option></select>
          </div>
          <div class="field">
            <label for="returnDebug">Debug artifacts</label>
            <select id="returnDebug"><option value="true">true</option><option value="false">false</option></select>
          </div>
          <div class="field">
            <label for="previewSize">Размер превью</label>
            <input id="previewSize" type="range" min="180" max="520" value="320" />
            <div class="value-row"><span>global preview width</span><span id="previewSizeValue">320 px</span></div>
          </div>
          <div class="actions">
            <button class="primary" id="runAllBtn">Запустить всё</button>
            <button class="ghost" id="clearBtn">Очистить результаты</button>
          </div>
        </div>
      </div>
      <div class="global-note" id="globalNote">Сервис готов. Загрузите изображение и запустите pipeline.</div>
    </section>

    <section class="pipeline" id="pipeline"></section>
  </div>

  <script>
    const stageDefinitions = [
      { key: "decode", title: "Decode", description: "Декодирование и базовая валидация входного изображения.", modeLocked: true },
      { key: "geometry_estimator", title: "Geometry", description: "Оценка камеры и общей геометрии сцены." },
      { key: "detector", title: "Detection", description: "Поиск главного объекта и подготовка рамки кропа." },
      { key: "segmenter", title: "Segmentation", description: "Маска объекта, cutout и кроп для дальнейших модулей." },
      { key: "foreground_refiner", title: "Foreground", description: "Коррекция цвета полупрозрачных пикселей после matting." },
      { key: "depth_estimator", title: "Depth", description: "Построение карты глубины на вырезанном объекте." },
      { key: "normal_estimator", title: "Normals", description: "Предсказание карты нормалей по refined cutout с fallback через depth." },
      { key: "shadow_generator", title: "Shadow", description: "Генерация слоя тени по light controls." },
      { key: "composer", title: "Composition", description: "Композит объекта и тени на выбранный фон." }
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
      stageModes: {
        geometry_estimator: "mock",
        detector: "mock",
        segmenter: "mock",
        foreground_refiner: "real",
        depth_estimator: "mock",
        normal_estimator: "real",
        shadow_generator: "real",
        composer: "real"
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

    imageFile.addEventListener("change", async (event) => {
      const file = event.target.files?.[0];
      if (!file) { return; }
      state.mimeType = file.type || "image/png";
      state.imageBase64 = await fileToBase64(file);
      globalNote.textContent = `Загружено: ${file.name}. Можно запускать pipeline.`;
    });

    document.getElementById("runAllBtn").addEventListener("click", () => runPipeline());
    document.getElementById("clearBtn").addEventListener("click", clearResults);

    function stageModeFor(key) {
      if (key === "decode") { return "real"; }
      return state.stageModes[key] || "mock";
    }

    function capabilityFor(key) {
      return state.capabilities[key] || null;
    }

    function renderCapabilityNotice(stage, stageState) {
      const capability = capabilityFor(stage.key);
      if (!capability) {
        return "";
      }
      if (stageState.actual_mode === "mock-fallback") {
        return `<div class="notice warn">Запрошен real, но сейчас выполнен mock-fallback.\nПричина: ${capability.detail || "реальный backend недоступен в текущем runtime."}</div>`;
      }
      if (stageModeFor(stage.key) === "real" && capability.using_mock) {
        return `<div class="notice warn">Для этого этапа real backend сейчас не активен.\nПричина: ${capability.detail || "runtime работает через mock backend."}</div>`;
      }
      if (stageState.actual_mode === "real") {
        return `<div class="notice ok">Этап действительно выполнился через real backend.</div>`;
      }
      return "";
    }

    function renderCards() {
      pipelineEl.innerHTML = "";
      stageDefinitions.forEach((stage, index) => {
        const card = document.createElement("article");
        card.className = "stage-card";
        const stageState = state.stages[stage.key] || {};
        const capability = capabilityFor(stage.key);
        card.innerHTML = `
          <div>
            <div class="stage-head">
              <div class="stage-title">
                <div class="badge">${String(index + 1).padStart(2, "0")}</div>
                <div>
                  <h3>${stage.title}</h3>
                  <p>${stage.description}</p>
                </div>
              </div>
              <div class="status-pill ${stageState.status || ""}" id="status-${stage.key}">${stageState.status || "idle"}</div>
            </div>
            <div class="stage-controls">
              <div class="mode-toggle">
                <button type="button" data-stage="${stage.key}" data-mode="mock" class="${stageModeFor(stage.key) === "mock" ? "active" : ""}" ${stage.modeLocked ? "disabled" : ""}>mock</button>
                <button type="button" data-stage="${stage.key}" data-mode="real" class="${stageModeFor(stage.key) === "real" ? "active" : ""}">real</button>
              </div>
              <button class="ghost" data-rerun="${stage.key}">Перезапустить этап</button>
            </div>
            <div class="meta">
              <div class="chip">requested: ${stageModeFor(stage.key)}</div>
              <div class="chip">actual: ${stageState.actual_mode || "n/a"}</div>
              <div class="chip">time: ${stageState.elapsed_ms != null ? `${stageState.elapsed_ms} ms` : "n/a"}</div>
              ${stageState.details?.device ? `<div class="chip">device: ${stageState.details.device}</div>` : ""}
              ${capability ? `<div class="chip">runtime: ${capability.implementation}</div>` : ""}
            </div>
            ${renderCapabilityNotice(stage, stageState)}
            <div class="detail-grid">${renderDetailsMarkup(stageState.details || null)}</div>
            <div class="error" id="error-${stage.key}" style="${stageState.error ? "display:block;" : ""}">${stageState.error || ""}</div>
          </div>
          <div class="previews" id="previews-${stage.key}">${renderPreviewMarkup(stageState.previews || [])}</div>
        `;
        pipelineEl.appendChild(card);
      });

      pipelineEl.querySelectorAll("[data-mode]").forEach((button) => {
        button.addEventListener("click", () => {
          const { stage, mode } = button.dataset;
          if (stage === "decode") { return; }
          state.stageModes[stage] = mode;
          const capability = capabilityFor(stage);
          if (mode === "real" && capability && capability.using_mock) {
            globalNote.textContent = `${stage}: real backend сейчас не активен, поэтому запуск пойдёт через mock-fallback. ${capability.detail || ""}`.trim();
          }
          renderCards();
        });
      });

      pipelineEl.querySelectorAll("[data-rerun]").forEach((button) => {
        button.addEventListener("click", () => runPipeline(button.dataset.rerun));
      });
    }

    function renderDetailsMarkup(details) {
      if (!details) {
        return "";
      }
      return Object.entries(details).map(([key, value]) => `
        <div class="detail-card">
          <small>${key.replaceAll("_", " ")}</small>
          <strong>${typeof value === "number" ? Number(value).toFixed(key.includes("confidence") ? 3 : 2) : value}</strong>
        </div>
      `).join("");
    }

    function renderPreviewMarkup(previews) {
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

      const body = {
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
          background: {
            mode: "solid",
            color_hex: document.getElementById("backgroundColor").value.toUpperCase()
          },
          output: {
            format: document.getElementById("outputFormat").value,
            width: null,
            height: null,
            return_debug: document.getElementById("returnDebug").value === "true"
          }
        },
        stage_modes: state.stageModes
      };

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
          if (!state.stages[stage.key]) {
            state.stages[stage.key] = { status: "skipped" };
          }
        });
        const failed = payload.stages.find((stage) => stage.status === "failed");
        globalNote.textContent = failed
          ? `Этап ${failed.title} завершился с ошибкой: ${failed.error}`
          : `Готово. Выполнено этапов: ${payload.stages.length}. Предупреждения: ${payload.warnings.join(", ") || "нет"}.`;
      } catch (error) {
        globalNote.textContent = `Ошибка запуска: ${error.message}`;
      }
      renderCards();
    }

    function clearResults() {
      state.stages = {};
      globalNote.textContent = "Результаты очищены.";
      renderCards();
    }

    function fileToBase64(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const result = String(reader.result || "");
          resolve(result.split(",")[1]);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    }

    async function loadCapabilities() {
      try {
        const response = await fetch("/v1/capabilities");
        if (!response.ok) {
          throw new Error("capabilities request failed");
        }
        const payload = await response.json();
        state.capabilities = Object.fromEntries((payload.components || []).map((component) => [component.name, component]));
        const segmenterCapability = state.capabilities.segmenter;
        if (segmenterCapability?.using_mock) {
          globalNote.textContent = `Segmentation real backend сейчас не активен. ${segmenterCapability.detail || ""}`.trim();
        }
      } catch (error) {
        globalNote.textContent = `Не удалось загрузить runtime capabilities: ${error.message}`;
      }
      renderCards();
    }

    loadCapabilities();
  </script>
</body>
</html>
"""
