AGENT_LATENCY_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Latency Console</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #647084;
      --line: #d9e0ea;
      --accent: #0f766e;
      --accent-2: #2563eb;
      --bad: #b42318;
      --warn: #a16207;
      --good: #067647;
      --shadow: 0 12px 30px rgba(23, 32, 51, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 36px;
    }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 30px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    .subhead {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 14px;
    }
    .topology {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .topo-chip {
      font-size: 12px;
      padding: 4px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      color: var(--muted);
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #98a2b3;
    }
    .status.live .dot { background: var(--good); }
    .status.error .dot { background: var(--bad); }
    .grid {
      display: grid;
      grid-template-columns: 330px 1fr;
      gap: 16px;
      align-items: start;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .controls {
      padding: 16px;
    }
    .section-title {
      margin: 0 0 12px;
      font-size: 15px;
      letter-spacing: 0;
    }
    .control-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 10px;
    }
    button, label.file {
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      color: var(--ink);
      font: inherit;
      font-size: 14px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 8px 10px;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
    }
    button.secondary {
      background: var(--accent-2);
      border-color: var(--accent-2);
      color: #ffffff;
    }
    button:disabled {
      opacity: 0.52;
      cursor: not-allowed;
    }
    input[type="file"] { display: none; }
    .field {
      display: grid;
      gap: 6px;
      margin: 12px 0;
    }
    .field label {
      font-size: 12px;
      color: var(--muted);
    }
    .field input, .field select {
      width: 100%;
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 9px;
      font: inherit;
      font-size: 14px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
      gap: 10px;
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }
    .metric {
      min-height: 86px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcff;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }
    .metric strong {
      display: block;
      font-size: clamp(22px, 3vw, 30px);
      line-height: 1;
      letter-spacing: 0;
    }
    .timeline-wrap {
      padding: 14px;
    }
    .quality {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }
    .quality-block {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcff;
      min-height: 128px;
    }
    .quality-block h3 {
      margin: 0 0 8px;
      font-size: 13px;
      letter-spacing: 0;
    }
    .text-box {
      min-height: 54px;
      max-height: 150px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .quality-score {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .score {
      min-height: 26px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 4px 8px;
      color: var(--ink);
      background: #ffffff;
      font-size: 12px;
    }
    .timeline {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
    }
    .timeline th,
    .timeline td {
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    .timeline th {
      color: var(--muted);
      font-weight: 600;
      background: #fbfcff;
    }
    .timeline th:nth-child(1), .timeline td:nth-child(1) { width: 120px; }
    .timeline th:nth-child(2), .timeline td:nth-child(2) { width: 96px; }
    .timeline th:nth-child(3), .timeline td:nth-child(3) { width: 96px; }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 999px;
      background: #eef4ff;
      color: #1849a9;
      font-size: 12px;
      white-space: nowrap;
    }
    .pill.error {
      background: #fef3f2;
      color: var(--bad);
    }
    .log {
      min-height: 120px;
      max-height: 250px;
      overflow: auto;
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0f172a;
      color: #dbeafe;
      padding: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      white-space: pre-wrap;
    }
    .audio {
      width: 100%;
      margin-top: 10px;
    }
    .audio-list {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }
    .audio-clip {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #ffffff;
    }
    .audio-clip-header {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 860px) {
      main { width: min(100vw - 20px, 720px); padding-top: 14px; }
      header { align-items: flex-start; flex-direction: column; }
      .grid { grid-template-columns: 1fr; }
      .quality { grid-template-columns: 1fr; }
      .control-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Agent Latency Console</h1>
        <p class="subhead">Realtime ASR -> LLM -> TTS through /v1/voice/sessions. Mic and speaker stay on this device; ASR/TTS can sit on a resident edge GPU.</p>
        <div id="topology" class="topology"></div>
      </div>
      <div id="status" class="status"><span class="dot"></span><span id="statusText">idle</span></div>
    </header>

    <div class="grid">
      <section class="controls">
        <h2 class="section-title">Session</h2>
        <div class="control-row">
          <button id="connectBtn" class="primary">Connect</button>
          <button id="resetBtn">Reset</button>
        </div>
        <div class="control-row">
          <button id="recordBtn" class="secondary" disabled>Record</button>
          <button id="stopBtn" disabled>Stop</button>
        </div>
        <div class="control-row">
          <label class="file" for="fileInput">Upload WAV</label>
          <button id="closeBtn" disabled>Close Session</button>
        </div>
        <input id="fileInput" type="file" accept="audio/*">

        <div class="field">
          <label for="chunkMs">Realtime upload chunk ms</label>
          <input id="chunkMs" type="number" min="40" max="2000" step="20" value="400">
        </div>
        <div class="field">
          <label for="datasetSelect">ASR dataset sample</label>
          <select id="datasetSelect"></select>
        </div>
        <div class="control-row">
          <button id="loadDatasetBtn">Load Samples</button>
          <button id="streamDatasetBtn" disabled>Stream Sample</button>
        </div>

        <div id="log" class="log"></div>
      </section>

      <section>
        <div class="metrics">
          <div class="metric"><span>First ASR</span><strong id="firstAsrMs">--</strong></div>
          <div class="metric"><span>LLM Start</span><strong id="llmStartMs">--</strong></div>
          <div class="metric"><span>LLM First Token</span><strong id="llmFirstChunkMs">--</strong></div>
          <div class="metric"><span>First TTS Text</span><strong id="firstTtsTextMs">--</strong></div>
          <div class="metric"><span>TTS First Chunk</span><strong id="ttsFirstChunkMs">--</strong></div>
          <div class="metric"><span>First Audio</span><strong id="firstAudioMs">--</strong></div>
          <div class="metric"><span>Total</span><strong id="totalMs">--</strong></div>
          <div class="metric"><span>Input Audio</span><strong id="inputAudioMs">--</strong></div>
          <div class="metric"><span>ASR Final After Input</span><strong id="asrFinalAfterInputMs">--</strong></div>
          <div class="metric"><span>First Audio After Input</span><strong id="firstAudioAfterInputMs">--</strong></div>
          <div class="metric"><span>LLM Duration</span><strong id="llmDurationMs">--</strong></div>
          <div class="metric"><span>TTS Wait</span><strong id="ttsWaitMs">--</strong></div>
        </div>
        <div class="quality">
          <div class="quality-block">
            <h3>ASR Golden</h3>
            <div id="goldenAsrText" class="text-box">No dataset sample selected.</div>
            <div class="quality-score">
              <span class="score">WER <strong id="asrWer">--</strong></span>
              <span class="score">CER <strong id="asrCer">--</strong></span>
            </div>
          </div>
          <div class="quality-block">
            <h3>Latest ASR Hypothesis</h3>
            <div id="asrHypText" class="text-box">Waiting for ASR events.</div>
          </div>
          <div class="quality-block">
            <h3>LLM Generated Output</h3>
            <div id="llmOutputText" class="text-box">Waiting for LLM chunks.</div>
          </div>
          <div class="quality-block">
            <h3>TTS Input From LLM</h3>
            <div id="ttsInputText" class="text-box">Waiting for LLM text sent to TTS.</div>
          </div>
          <div class="quality-block">
            <h3>TTS Audio</h3>
            <div id="ttsAudioInfo" class="text-box">No audio returned yet.</div>
            <div id="audioClips" class="audio-list"></div>
          </div>
        </div>
        <div class="timeline-wrap">
          <h2 class="section-title">Events</h2>
          <table class="timeline">
            <thead>
              <tr>
                <th>Event</th>
                <th>Offset</th>
                <th>Delta</th>
                <th>Text</th>
              </tr>
            </thead>
            <tbody id="eventsBody"></tbody>
          </table>
        </div>
      </section>
    </div>
  </main>

  <script>
    const endpointPath = "/v1/voice/sessions";
    const els = {
      status: document.getElementById("status"),
      statusText: document.getElementById("statusText"),
      connectBtn: document.getElementById("connectBtn"),
      resetBtn: document.getElementById("resetBtn"),
      recordBtn: document.getElementById("recordBtn"),
      stopBtn: document.getElementById("stopBtn"),
      closeBtn: document.getElementById("closeBtn"),
      fileInput: document.getElementById("fileInput"),
      chunkMs: document.getElementById("chunkMs"),
      datasetSelect: document.getElementById("datasetSelect"),
      loadDatasetBtn: document.getElementById("loadDatasetBtn"),
      streamDatasetBtn: document.getElementById("streamDatasetBtn"),
      firstAsrMs: document.getElementById("firstAsrMs"),
      llmStartMs: document.getElementById("llmStartMs"),
      llmFirstChunkMs: document.getElementById("llmFirstChunkMs"),
      firstTtsTextMs: document.getElementById("firstTtsTextMs"),
      ttsFirstChunkMs: document.getElementById("ttsFirstChunkMs"),
      firstAudioMs: document.getElementById("firstAudioMs"),
      totalMs: document.getElementById("totalMs"),
      inputAudioMs: document.getElementById("inputAudioMs"),
      asrFinalAfterInputMs: document.getElementById("asrFinalAfterInputMs"),
      firstAudioAfterInputMs: document.getElementById("firstAudioAfterInputMs"),
      llmDurationMs: document.getElementById("llmDurationMs"),
      ttsWaitMs: document.getElementById("ttsWaitMs"),
      goldenAsrText: document.getElementById("goldenAsrText"),
      asrHypText: document.getElementById("asrHypText"),
      llmOutputText: document.getElementById("llmOutputText"),
      asrWer: document.getElementById("asrWer"),
      asrCer: document.getElementById("asrCer"),
      ttsInputText: document.getElementById("ttsInputText"),
      ttsAudioInfo: document.getElementById("ttsAudioInfo"),
      audioClips: document.getElementById("audioClips"),
      eventsBody: document.getElementById("eventsBody"),
      log: document.getElementById("log"),
      topology: document.getElementById("topology")
    };

    const state = {
      ws: null,
      startedAt: null,
      lastEventAt: null,
      firstAsrMs: null,
      llmStartMs: null,
      llmFirstChunkMs: null,
      firstTtsTextMs: null,
      ttsFirstChunkMs: null,
      firstAudioMs: null,
      totalMs: null,
      asrFinalMs: null,
      llmDoneMs: null,
      inputEndedAt: null,
      audioSamplesSent: 0,
      goldenAsrText: "",
      asrHypText: "",
      llmOutputText: "",
      ttsInputText: "",
      ttsAudioChunks: 0,
      ttsAudioBytes: 0,
      datasetSamples: [],
      audioContext: null,
      mediaStream: null,
      source: null,
      processor: null,
      recording: false,
      playbackQueue: [],
      playbackBusy: false
    };

    function nowMs() {
      return performance.now();
    }

    function fmt(ms) {
      return ms == null ? "--" : `${ms.toFixed(1)} ms`;
    }

    function setStatus(text, mode) {
      els.statusText.textContent = text;
      els.status.className = `status ${mode || ""}`.trim();
    }

    function log(line) {
      const time = new Date().toLocaleTimeString();
      els.log.textContent += `[${time}] ${line}\n`;
      els.log.scrollTop = els.log.scrollHeight;
    }

    function updateMetrics() {
      els.firstAsrMs.textContent = fmt(state.firstAsrMs);
      els.llmStartMs.textContent = fmt(state.llmStartMs);
      els.llmFirstChunkMs.textContent = fmt(state.llmFirstChunkMs);
      els.firstTtsTextMs.textContent = fmt(state.firstTtsTextMs);
      els.ttsFirstChunkMs.textContent = fmt(state.ttsFirstChunkMs);
      els.firstAudioMs.textContent = fmt(state.firstAudioMs);
      els.totalMs.textContent = fmt(state.totalMs);
      const inputAudioMs = state.audioSamplesSent > 0 ? state.audioSamplesSent / 16 : null;
      const asrFinalAfterInputMs = state.inputEndedAt && state.asrFinalMs != null
        ? state.asrFinalMs - (state.inputEndedAt - state.startedAt)
        : null;
      const firstAudioAfterInputMs = state.inputEndedAt && state.firstAudioMs != null
        ? state.firstAudioMs - (state.inputEndedAt - state.startedAt)
        : null;
      const llmDurationMs = state.llmStartMs != null && state.llmDoneMs != null
        ? state.llmDoneMs - state.llmStartMs
        : null;
      const ttsWaitMs = state.llmDoneMs != null && state.firstAudioMs != null
        ? state.firstAudioMs - state.llmDoneMs
        : null;
      els.inputAudioMs.textContent = fmt(inputAudioMs);
      els.asrFinalAfterInputMs.textContent = fmt(asrFinalAfterInputMs);
      els.firstAudioAfterInputMs.textContent = fmt(firstAudioAfterInputMs);
      els.llmDurationMs.textContent = fmt(llmDurationMs);
      els.ttsWaitMs.textContent = fmt(ttsWaitMs);
    }

    function resetMetrics() {
      state.startedAt = null;
      state.lastEventAt = null;
      state.firstAsrMs = null;
      state.llmStartMs = null;
      state.llmFirstChunkMs = null;
      state.firstTtsTextMs = null;
      state.ttsFirstChunkMs = null;
      state.firstAudioMs = null;
      state.totalMs = null;
      state.asrFinalMs = null;
      state.llmDoneMs = null;
      state.inputEndedAt = null;
      state.audioSamplesSent = 0;
      state.asrHypText = "";
      state.llmOutputText = "";
      state.ttsInputText = "";
      state.ttsAudioChunks = 0;
      state.ttsAudioBytes = 0;
      state.playbackQueue = [];
      state.playbackBusy = false;
      els.eventsBody.textContent = "";
      els.log.textContent = "";
      els.asrHypText.textContent = "Waiting for ASR events.";
      els.llmOutputText.textContent = "Waiting for LLM chunks.";
      els.asrWer.textContent = "--";
      els.asrCer.textContent = "--";
      els.ttsInputText.textContent = "Waiting for LLM text sent to TTS.";
      els.ttsAudioInfo.textContent = "No audio returned yet.";
      els.audioClips.textContent = "";
      updateMetrics();
    }

    function markStart() {
      if (state.startedAt == null) {
        state.startedAt = nowMs();
        state.lastEventAt = state.startedAt;
      }
    }

    function addEventRow(type, text, at) {
      const offset = state.startedAt == null ? null : at - state.startedAt;
      const delta = state.lastEventAt == null ? null : at - state.lastEventAt;
      state.lastEventAt = at;
      const tr = document.createElement("tr");
      const kind = document.createElement("td");
      const offsetTd = document.createElement("td");
      const deltaTd = document.createElement("td");
      const textTd = document.createElement("td");
      const pill = document.createElement("span");
      pill.className = type === "error" ? "pill error" : "pill";
      pill.textContent = type;
      kind.appendChild(pill);
      offsetTd.textContent = fmt(offset);
      deltaTd.textContent = fmt(delta);
      textTd.textContent = text || "";
      tr.append(kind, offsetTd, deltaTd, textTd);
      els.eventsBody.appendChild(tr);
    }

    function websocketUrl() {
      const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
      return `${scheme}//${window.location.host}${endpointPath}`;
    }

    async function connect() {
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        return state.ws;
      }
      resetMetrics();
      setStatus("connecting", "");
      const ws = new WebSocket(websocketUrl());
      ws.binaryType = "arraybuffer";
      state.ws = ws;
      ws.onopen = () => {
        markStart();
        ws.send(JSON.stringify({type: "start"}));
        setStatus("connected", "live");
        els.recordBtn.disabled = false;
        els.closeBtn.disabled = false;
        log(`connected ${endpointPath}`);
      };
      ws.onmessage = (message) => {
        if (message.data instanceof ArrayBuffer) {
          handleAudio(message.data);
          return;
        }
        handleJsonEvent(JSON.parse(message.data));
      };
      ws.onerror = () => {
        setStatus("websocket error", "error");
        log("websocket error");
      };
      ws.onclose = () => {
        setStatus("closed", "");
        els.recordBtn.disabled = true;
        els.stopBtn.disabled = true;
        els.closeBtn.disabled = true;
        state.recording = false;
      };
      await new Promise((resolve, reject) => {
        ws.addEventListener("open", resolve, {once: true});
        ws.addEventListener("error", reject, {once: true});
      });
      return ws;
    }

    function handleJsonEvent(event) {
      const at = nowMs();
      if (event.type !== "session_started") {
        markStart();
      }
      if (event.type && event.type.startsWith("asr_") && state.firstAsrMs == null) {
        state.firstAsrMs = at - state.startedAt;
      }
      if (event.type === "llm_start" && state.llmStartMs == null) {
        state.llmStartMs = at - state.startedAt;
      }
      if (event.type === "llm_first_chunk" && state.llmFirstChunkMs == null) {
        state.llmFirstChunkMs = at - state.startedAt;
      }
      if (event.type === "llm_sentence_ready" && state.firstTtsTextMs == null) {
        state.firstTtsTextMs = at - state.startedAt;
      }
      if (event.type === "tts_first_chunk" && state.ttsFirstChunkMs == null) {
        state.ttsFirstChunkMs = at - state.startedAt;
      }
      if (event.type && event.type.startsWith("asr_")) {
        updateAsrHypothesis(event);
      }
      if (event.type === "asr_final") {
        state.asrFinalMs = at - state.startedAt;
      }
      if (event.type === "tts_chunk") {
        state.ttsInputText = [state.ttsInputText, event.text || ""].filter(Boolean).join("\n");
        els.ttsInputText.textContent = state.ttsInputText || "(empty)";
      }
      if (event.type === "llm_chunk" || event.type === "llm_done") {
        updateLlmOutput(event);
      }
      if (event.type === "llm_done") {
        state.llmDoneMs = at - state.startedAt;
      }
      if (event.type === "done") {
        state.totalMs = at - state.startedAt;
      }
      updateMetrics();
      addEventRow(event.type, event.text || event.committed_text || "", at);
      if (event.type === "error") {
        setStatus("agent error", "error");
      }
    }

    function fullAsrText(event) {
      const text = (event.text || "").trim();
      const committed = (event.committed_text || "").trim();
      if (event.type === "asr_final") {
        return text || committed;
      }
      if (event.type === "asr_committed") {
        return event.committed_text || event.text || "";
      }
      if (committed && text && text !== committed && !text.startsWith(committed)) {
        return `${committed} ${text}`.replace(/\s+/g, " ").trim();
      }
      return text || committed;
    }

    function updateAsrHypothesis(event) {
      const transcript = fullAsrText(event);
      if (!transcript) {
        return;
      }
      state.asrHypText = transcript;
      els.asrHypText.textContent = state.asrHypText;
      updateAsrQuality();
    }

    function updateLlmOutput(event) {
      if (event.type === "llm_done") {
        state.llmOutputText = event.text || state.llmOutputText;
      } else {
        state.llmOutputText += event.text || "";
      }
      els.llmOutputText.textContent = state.llmOutputText || "(empty)";
    }

    function handleAudio(buffer) {
      const at = nowMs();
      if (state.firstAudioMs == null) {
        state.firstAudioMs = at - state.startedAt;
      }
      state.ttsAudioChunks += 1;
      state.ttsAudioBytes += buffer.byteLength;
      const wavSeconds = wavDurationSeconds(buffer);
      els.ttsAudioInfo.textContent = `${state.ttsAudioChunks} chunk(s), ${state.ttsAudioBytes} bytes total`;
      updateMetrics();
      addEventRow("tts_audio", `${buffer.byteLength} bytes${wavSeconds == null ? "" : `, ${wavSeconds.toFixed(2)}s`}`, at);
      appendAudioClip(buffer, wavSeconds);
    }

    function wavDurationSeconds(buffer) {
      const view = new DataView(buffer);
      if (buffer.byteLength < 44 || readAscii(view, 0, 4) !== "RIFF" || readAscii(view, 8, 4) !== "WAVE") {
        return null;
      }
      let offset = 12;
      let byteRate = null;
      let dataBytes = null;
      while (offset + 8 <= buffer.byteLength) {
        const id = readAscii(view, offset, 4);
        const size = view.getUint32(offset + 4, true);
        if (id === "fmt " && offset + 16 <= buffer.byteLength) {
          byteRate = view.getUint32(offset + 12, true);
        }
        if (id === "data") {
          dataBytes = size;
          break;
        }
        offset += 8 + size + (size % 2);
      }
      if (!byteRate || !dataBytes) {
        return null;
      }
      return dataBytes / byteRate;
    }

    function readAscii(view, offset, length) {
      let out = "";
      for (let i = 0; i < length; i += 1) {
        out += String.fromCharCode(view.getUint8(offset + i));
      }
      return out;
    }

    function appendAudioClip(buffer, wavSeconds) {
      const blob = new Blob([buffer], {type: "audio/wav"});
      const url = URL.createObjectURL(blob);
      const index = state.ttsAudioChunks;
      const clip = document.createElement("div");
      clip.className = "audio-clip";
      const header = document.createElement("div");
      header.className = "audio-clip-header";
      const label = document.createElement("span");
      label.textContent = `Chunk ${index}`;
      const meta = document.createElement("span");
      meta.textContent = `${buffer.byteLength} bytes${wavSeconds == null ? "" : `, ${wavSeconds.toFixed(2)}s`}`;
      const audio = document.createElement("audio");
      audio.className = "audio";
      audio.controls = true;
      audio.preload = "metadata";
      audio.src = url;
      audio.addEventListener("loadedmetadata", () => {
        if (Number.isFinite(audio.duration) && wavSeconds == null) {
          meta.textContent = `${buffer.byteLength} bytes, ${audio.duration.toFixed(2)}s`;
        }
      });
      audio.addEventListener("ended", () => {
        sendPlaybackAck(index, wavSeconds ?? audio.duration ?? null);
        playNextLocalClip();
      });
      header.append(label, meta);
      clip.append(header, audio);
      els.audioClips.appendChild(clip);
      enqueueLocalPlayback(audio);
    }

    function enqueueLocalPlayback(audio) {
      state.playbackQueue.push(audio);
      if (!state.playbackBusy) {
        playNextLocalClip();
      }
    }

    function playNextLocalClip() {
      const audio = state.playbackQueue.shift();
      if (!audio) {
        state.playbackBusy = false;
        return;
      }
      state.playbackBusy = true;
      const played = audio.play();
      if (played && typeof played.catch === "function") {
        played.catch((error) => {
          state.playbackBusy = false;
          log(`local speaker blocked: ${error && error.message ? error.message : error}`);
        });
      }
    }

    async function loadTopology() {
      if (!els.topology) {
        return;
      }
      try {
        const report = await fetch("/health").then((response) => response.json());
        const placement = report.topology || {};
        const chips = [
          ["Mic", placement.microphone || "browser-local"],
          ["ASR", placement.asr || "server"],
          ["LLM", placement.llm || "server"],
          ["TTS", placement.tts || "server"],
          ["Speaker", placement.speaker || "browser-local"]
        ];
        els.topology.replaceChildren(...chips.map(([label, value]) => {
          const chip = document.createElement("span");
          chip.className = "topo-chip";
          chip.textContent = `${label}: ${value}`;
          return chip;
        }));
      } catch (error) {
        log(`topology unavailable: ${error.message}`);
      }
    }

    function sendPlaybackAck(chunkIndex, audioSeconds) {
      if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
        return;
      }
      state.ws.send(JSON.stringify({
        type: "tts_played",
        chunk_index: chunkIndex,
        audio_ms: Number.isFinite(audioSeconds) ? Math.round(audioSeconds * 1000) : null
      }));
    }

    function downmix(inputBuffer) {
      if (inputBuffer.numberOfChannels === 1) {
        return inputBuffer.getChannelData(0);
      }
      const length = inputBuffer.length;
      const mixed = new Float32Array(length);
      for (let ch = 0; ch < inputBuffer.numberOfChannels; ch += 1) {
        const data = inputBuffer.getChannelData(ch);
        for (let i = 0; i < length; i += 1) {
          mixed[i] += data[i] / inputBuffer.numberOfChannels;
        }
      }
      return mixed;
    }

    function resampleTo16k(input, sourceRate) {
      if (sourceRate === 16000) {
        return new Float32Array(input);
      }
      const ratio = sourceRate / 16000;
      const length = Math.max(1, Math.floor(input.length / ratio));
      const output = new Float32Array(length);
      for (let i = 0; i < length; i += 1) {
        output[i] = input[Math.min(input.length - 1, Math.floor(i * ratio))];
      }
      return output;
    }

    function sendPcm16k(float32) {
      if (!state.ws || state.ws.readyState !== WebSocket.OPEN || float32.length === 0) {
        return;
      }
      markStart();
      state.audioSamplesSent += float32.length;
      state.ws.send(float32.buffer.slice(float32.byteOffset, float32.byteOffset + float32.byteLength));
      updateMetrics();
    }

    async function startRecording() {
      const ws = await connect();
      if (ws.readyState !== WebSocket.OPEN) {
        return;
      }
      state.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}
      });
      state.audioContext = new AudioContext();
      state.source = state.audioContext.createMediaStreamSource(state.mediaStream);
      state.processor = state.audioContext.createScriptProcessor(4096, 1, 1);
      state.processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        sendPcm16k(resampleTo16k(input, state.audioContext.sampleRate));
      };
      state.source.connect(state.processor);
      state.processor.connect(state.audioContext.destination);
      state.recording = true;
      els.recordBtn.disabled = true;
      els.stopBtn.disabled = false;
      setStatus("recording", "live");
      log("microphone streaming started");
    }

    async function stopRecording(closeSession) {
      if (state.processor) {
        state.processor.disconnect();
      }
      if (state.source) {
        state.source.disconnect();
      }
      if (state.mediaStream) {
        state.mediaStream.getTracks().forEach((track) => track.stop());
      }
      if (state.audioContext) {
        await state.audioContext.close();
      }
      state.processor = null;
      state.source = null;
      state.mediaStream = null;
      state.audioContext = null;
      state.recording = false;
      els.recordBtn.disabled = false;
      els.stopBtn.disabled = true;
      if (closeSession) {
        state.inputEndedAt = nowMs();
        updateMetrics();
        closeVoiceSession();
      }
      log("microphone streaming stopped");
    }

    function closeVoiceSession() {
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        if (state.inputEndedAt == null) {
          state.inputEndedAt = nowMs();
          updateMetrics();
        }
        state.ws.send(JSON.stringify({type: "close"}));
        log("close sent");
      }
    }

    async function resetSession() {
      if (state.recording) {
        await stopRecording(false);
      }
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({type: "reset"}));
      } else {
        await connect();
      }
      resetMetrics();
      markStart();
      log("session reset");
    }

    function setGoldenAsrText(text) {
      state.goldenAsrText = text || "";
      els.goldenAsrText.textContent = state.goldenAsrText || "No golden ASR text for this input.";
      updateAsrQuality();
    }

    function normalizeForScore(text) {
      return String(text || "").toLowerCase().replace(/[^\p{L}\p{N}'\s]/gu, " ").replace(/\s+/g, " ").trim();
    }

    function editDistance(left, right) {
      const previous = new Array(right.length + 1);
      const current = new Array(right.length + 1);
      for (let j = 0; j <= right.length; j += 1) {
        previous[j] = j;
      }
      for (let i = 1; i <= left.length; i += 1) {
        current[0] = i;
        for (let j = 1; j <= right.length; j += 1) {
          const cost = left[i - 1] === right[j - 1] ? 0 : 1;
          current[j] = Math.min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost);
        }
        for (let j = 0; j <= right.length; j += 1) {
          previous[j] = current[j];
        }
      }
      return previous[right.length];
    }

    function updateAsrQuality() {
      if (!state.goldenAsrText || !state.asrHypText) {
        els.asrWer.textContent = "--";
        els.asrCer.textContent = "--";
        return;
      }
      const ref = normalizeForScore(state.goldenAsrText);
      const hyp = normalizeForScore(state.asrHypText);
      const refWords = ref ? ref.split(" ") : [];
      const hypWords = hyp ? hyp.split(" ") : [];
      const wer = refWords.length ? editDistance(refWords, hypWords) / refWords.length : NaN;
      const refChars = Array.from(ref.replace(/\s+/g, ""));
      const hypChars = Array.from(hyp.replace(/\s+/g, ""));
      const cer = refChars.length ? editDistance(refChars, hypChars) / refChars.length : NaN;
      els.asrWer.textContent = Number.isFinite(wer) ? wer.toFixed(4) : "--";
      els.asrCer.textContent = Number.isFinite(cer) ? cer.toFixed(4) : "--";
    }

    async function loadDatasetSamples() {
      els.datasetSelect.textContent = "";
      const response = await fetch("/agent-latency/datasets?split=test-clean&limit=8");
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
      state.datasetSamples = payload.samples || [];
      for (const sample of state.datasetSamples) {
        const option = document.createElement("option");
        option.value = String(sample.index);
        option.textContent = `${sample.sample_id} (${sample.duration_seconds}s)`;
        els.datasetSelect.appendChild(option);
      }
      els.streamDatasetBtn.disabled = state.datasetSamples.length === 0;
      if (state.datasetSamples[0]) {
        setGoldenAsrText(state.datasetSamples[0].golden_asr_text);
      }
      log(`loaded ${state.datasetSamples.length} dataset sample(s)`);
    }

    async function streamAudioBufferRealtime(arrayBuffer, label) {
      const ws = await connect();
      const context = new AudioContext();
      const audioBuffer = await context.decodeAudioData(arrayBuffer.slice(0));
      const pcm16k = resampleTo16k(downmix(audioBuffer), audioBuffer.sampleRate);
      await context.close();
      const chunkMs = Math.max(40, Number.parseInt(els.chunkMs.value || "400", 10));
      const chunkSamples = Math.max(1, Math.floor(16000 * chunkMs / 1000));
      log(`realtime streaming ${label}, ${pcm16k.length} samples`);
      for (let offset = 0; offset < pcm16k.length; offset += chunkSamples) {
        if (!state.ws || state.ws.readyState !== WebSocket.OPEN || ws.readyState !== WebSocket.OPEN) {
          break;
        }
        sendPcm16k(pcm16k.slice(offset, offset + chunkSamples));
        await new Promise((resolve) => setTimeout(resolve, chunkMs));
      }
      state.inputEndedAt = nowMs();
      updateMetrics();
      closeVoiceSession();
    }

    async function streamUploadedAudioRealtime(file) {
      setGoldenAsrText("");
      await streamAudioBufferRealtime(await file.arrayBuffer(), file.name);
    }

    async function streamSelectedDatasetSampleRealtime() {
      const selected = state.datasetSamples.find((sample) => String(sample.index) === els.datasetSelect.value);
      if (!selected) {
        return;
      }
      setGoldenAsrText(selected.golden_asr_text);
      const response = await fetch(`/agent-latency/datasets/${selected.index}/audio?split=test-clean&limit=8`);
      if (!response.ok) {
        throw new Error(await response.text());
      }
      await streamAudioBufferRealtime(await response.arrayBuffer(), selected.sample_id);
    }

    els.connectBtn.addEventListener("click", () => connect().catch((error) => {
      setStatus("connect failed", "error");
      log(error.message);
    }));
    els.recordBtn.addEventListener("click", () => startRecording().catch((error) => {
      setStatus("record failed", "error");
      log(error.message);
    }));
    els.stopBtn.addEventListener("click", () => stopRecording(true));
    els.closeBtn.addEventListener("click", closeVoiceSession);
    els.resetBtn.addEventListener("click", () => resetSession().catch((error) => log(error.message)));
    els.loadDatasetBtn.addEventListener("click", () => loadDatasetSamples().catch((error) => log(error.message)));
    els.streamDatasetBtn.addEventListener("click", () => streamSelectedDatasetSampleRealtime().catch((error) => {
      setStatus("dataset stream failed", "error");
      log(error.message);
    }));
    els.datasetSelect.addEventListener("change", () => {
      const selected = state.datasetSamples.find((sample) => String(sample.index) === els.datasetSelect.value);
      setGoldenAsrText(selected ? selected.golden_asr_text : "");
    });
    els.fileInput.addEventListener("change", (event) => {
      const file = event.target.files && event.target.files[0];
      if (file) {
        streamUploadedAudioRealtime(file).catch((error) => {
          setStatus("upload failed", "error");
          log(error.message);
        });
      }
      event.target.value = "";
    });
    loadTopology().catch((error) => log(`topology unavailable: ${error.message}`));
    loadDatasetSamples().catch((error) => log(`dataset samples unavailable: ${error.message}`));
  </script>
</body>
</html>
"""
