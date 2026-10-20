/**
 * Frontend JavaScript for Ban Quality ML Dashboard.
 * Interacts with FastAPI backend endpoints strictly through JSON REST API.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Feature specification references for live client validation
  const FEATURE_BOUNDS = {
    suhu: { minObs: 140.0, maxObs: 180.0, minIdeal: 150.0, maxIdeal: 165.0, unit: "°C" },
    tekanan: { minObs: 15.0, maxObs: 25.0, minIdeal: 18.0, maxIdeal: 22.0, unit: "bar" },
    waktu_curing: { minObs: 8.0, maxObs: 15.0, minIdeal: 10.0, maxIdeal: 13.0, unit: "menit" },
    rasio_karet: { minObs: 0.40, maxObs: 0.80, minIdeal: 0.50, maxIdeal: 0.70, unit: "-" },
    kekerasan_karet: { minObs: 55.1, maxObs: 75.0, minIdeal: 60.0, maxIdeal: 70.0, unit: "Shore A" }
  };

  // DOM Elements
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");
  const predictionForm = document.getElementById("predictionForm");
  const btnSubmitPredict = document.getElementById("btnSubmitPredict");
  const serviceStatusBadge = document.getElementById("serviceStatusBadge");

  // States
  const stateInitial = document.getElementById("stateInitial");
  const stateLoading = document.getElementById("stateLoading");
  const stateError = document.getElementById("stateError");
  const stateSuccess = document.getElementById("stateSuccess");
  const errorTitle = document.getElementById("errorTitle");
  const errorMessage = document.getElementById("errorMessage");

  // Success elements
  const decisionBox = document.getElementById("decisionBox");
  const decisionLabel = document.getElementById("decisionLabel");
  const decisionDesc = document.getElementById("decisionDesc");
  const probGoodVal = document.getElementById("probGoodVal");
  const probGoodBar = document.getElementById("probGoodBar");
  const probBadVal = document.getElementById("probBadVal");
  const probBadBar = document.getElementById("probBadBar");
  const paramCheckTableBody = document.getElementById("paramCheckTableBody");
  const warningsContainer = document.getElementById("warningsContainer");
  const warningsList = document.getElementById("warningsList");
  const resultModelVersion = document.getElementById("resultModelVersion");

  // Input pairs
  const inputs = [
    { num: document.getElementById("inputSuhu"), slider: document.getElementById("sliderSuhu"), badge: document.getElementById("badgeSuhu"), key: "suhu" },
    { num: document.getElementById("inputTekanan"), slider: document.getElementById("sliderTekanan"), badge: document.getElementById("badgeTekanan"), key: "tekanan" },
    { num: document.getElementById("inputWaktu"), slider: document.getElementById("sliderWaktu"), badge: document.getElementById("badgeWaktu"), key: "waktu_curing" },
    { num: document.getElementById("inputRasio"), slider: document.getElementById("sliderRasio"), badge: document.getElementById("badgeRasio"), key: "rasio_karet" },
    { num: document.getElementById("inputKekerasan"), slider: document.getElementById("sliderKekerasan"), badge: document.getElementById("badgeKekerasan"), key: "kekerasan_karet" }
  ];

  // 1. Tab Navigation
  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add("active");
        if (targetId === "tab-model" && window.neuronVisualizer) {
          window.neuronVisualizer.onTabVisible();
        }
      }
    });
  });

  // Navigate to Tab Model from Live Predict Widget
  document.getElementById("btnGoToNeuronTab")?.addEventListener("click", () => {
    const tabModelBtn = document.querySelector('[data-tab="tab-model"]');
    if (tabModelBtn) {
      tabModelBtn.click();
      setTimeout(() => {
        document.getElementById("neuronCanvasContainer")?.scrollIntoView({ behavior: "smooth", block: "center" });
        if (window.neuronVisualizer) {
          window.neuronVisualizer.onTabVisible();
        }
      }, 50);
    }
  });

  // 2. Input & Slider Synchronization and Live Range Badges
  function updateRangeBadge(item) {
    if (!item.num || !item.badge) return;
    const val = parseFloat(item.num.value);
    const bounds = FEATURE_BOUNDS[item.key];
    if (isNaN(val) || !bounds) return;

    if (val >= bounds.minIdeal && val <= bounds.maxIdeal) {
      item.badge.className = "range-badge badge-success";
      item.badge.textContent = "Rentang Ideal ✓";
    } else if (val >= bounds.minObs && val <= bounds.maxObs) {
      item.badge.className = "range-badge badge-warning";
      item.badge.textContent = "Di Luar Ideal ⚠";
    } else {
      item.badge.className = "range-badge badge-danger";
      item.badge.textContent = "Di Luar Dataset ⚡";
    }
  }

  inputs.forEach(item => {
    if (!item.num || !item.slider) return;
    item.num.addEventListener("input", () => {
      item.slider.value = item.num.value;
      updateRangeBadge(item);
    });
    item.slider.addEventListener("input", () => {
      item.num.value = item.slider.value;
      updateRangeBadge(item);
    });
    updateRangeBadge(item);
  });

  // 3. Preset Buttons in Live Predict
  document.getElementById("presetIdeal")?.addEventListener("click", () => {
    setFormValues(157.5, 20.0, 11.5, 0.60, 65.0);
  });
  document.getElementById("presetLowTemp")?.addEventListener("click", () => {
    setFormValues(140.0, 20.0, 11.5, 0.60, 62.0);
  });
  document.getElementById("presetHighTemp")?.addEventListener("click", () => {
    setFormValues(180.0, 20.0, 11.5, 0.60, 71.0);
  });

  function setFormValues(suhu, tekanan, waktu, rasio, kekerasan = 65.0) {
    const elSuhu = document.getElementById("inputSuhu");
    const elTek = document.getElementById("inputTekanan");
    const elWkt = document.getElementById("inputWaktu");
    const elRas = document.getElementById("inputRasio");
    const elKek = document.getElementById("inputKekerasan");
    const sSuhu = document.getElementById("sliderSuhu");
    const sTek = document.getElementById("sliderTekanan");
    const sWkt = document.getElementById("sliderWaktu");
    const sRas = document.getElementById("sliderRasio");
    const sKek = document.getElementById("sliderKekerasan");

    if (elSuhu) elSuhu.value = suhu;
    if (sSuhu) sSuhu.value = suhu;
    if (elTek) elTek.value = tekanan;
    if (sTek) sTek.value = tekanan;
    if (elWkt) elWkt.value = waktu;
    if (sWkt) sWkt.value = waktu;
    if (elRas) elRas.value = rasio;
    if (sRas) sRas.value = rasio;
    if (elKek) elKek.value = kekerasan;
    if (sKek) sKek.value = kekerasan;

    inputs.forEach(updateRangeBadge);
  }

  // 4. Submit Prediction Form
  predictionForm?.addEventListener("submit", async (e) => {
    e.preventDefault();

    const payload = {
      suhu: parseFloat(document.getElementById("inputSuhu").value),
      tekanan: parseFloat(document.getElementById("inputTekanan").value),
      waktu_curing: parseFloat(document.getElementById("inputWaktu").value),
      rasio_karet: parseFloat(document.getElementById("inputRasio").value),
      kekerasan_karet: parseFloat(document.getElementById("inputKekerasan").value)
    };

    setUIState("loading");

    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await resp.json();

      if (!resp.ok) {
        throw new Error(data.detail || "Gagal melakukan prediksi.");
      }

      renderPredictionResult(data);
      setUIState("success");
    } catch (err) {
      if (errorTitle) errorTitle.textContent = "Gagal Melakukan Prediksi";
      if (errorMessage) errorMessage.textContent = err.message || "Layanan API tidak dapat dihubungi.";
      setUIState("error");
    }
  });

  function renderPredictionResult(data) {
    if (resultModelVersion) {
      resultModelVersion.textContent = `Model: ${data.model_version} | Thresh: ${data.threshold}`;
    }

    const isGood = data.label === 0;
    if (decisionBox) {
      decisionBox.className = `decision-box ${isGood ? 'decision-good' : 'decision-bad'}`;
    }
    if (decisionLabel) decisionLabel.textContent = data.label_name;
    if (decisionDesc) {
      decisionDesc.textContent = isGood
        ? "Parameter proses vulkanisasi diprediksi memenuhi standar kualitas mutu (Baik)."
        : "Parameter berpotensi memicu cacat vulkanisasi. Disarankan perbaikan parameter proses.";
    }

    const pGoodPct = (data.probability_good * 100).toFixed(1);
    const pBadPct = (data.probability_bad * 100).toFixed(1);

    if (probGoodVal) probGoodVal.textContent = `${pGoodPct}%`;
    if (probGoodBar) probGoodBar.style.width = `${pGoodPct}%`;

    if (probBadVal) probBadVal.textContent = `${pBadPct}%`;
    if (probBadBar) probBadBar.style.width = `${pBadPct}%`;

    // Render table
    if (paramCheckTableBody) {
      paramCheckTableBody.innerHTML = "";
      for (const [key, check] of Object.entries(data.feature_checks || {})) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><code>${key}</code></td>
          <td><strong>${check.value} ${check.unit}</strong></td>
          <td><span class="badge ${check.in_ideal_range ? 'badge-success' : 'badge-warning'}">${check.in_ideal_range ? 'Ideal' : 'Non-Ideal'}</span></td>
          <td><span class="badge ${check.in_dataset_range ? 'badge-success' : 'badge-danger'}">${check.in_dataset_range ? 'Dalam Rentang' : 'Di Luar Rentang'}</span></td>
        `;
        paramCheckTableBody.appendChild(tr);
      }
    }

    // Warnings
    if (data.warnings && data.warnings.length > 0) {
      if (warningsContainer) warningsContainer.style.display = "block";
      if (warningsList) {
        warningsList.innerHTML = "";
        data.warnings.forEach(w => {
          const li = document.createElement("li");
          li.textContent = w;
          warningsList.appendChild(li);
        });
      }
    } else {
      if (warningsContainer) warningsContainer.style.display = "none";
    }

    // Render live neuron activity flow
    if (data.neuron_activations) {
      renderLiveNeuronFlow(data.neuron_activations);
      if (window.neuronVisualizer) {
        window.neuronVisualizer.setActivations(data.neuron_activations);
      }
    }
  }

  function renderLiveNeuronFlow(act) {
    if (!act || !act.layers) return;
    const h1 = act.layers[0];
    const h2 = act.layers[1];

    const liveH1Dot = document.getElementById("liveH1Dot");
    const liveH1Firing = document.getElementById("liveH1Firing");
    const liveH2Dot = document.getElementById("liveH2Dot");
    const liveH2Firing = document.getElementById("liveH2Firing");
    const liveOutDot = document.getElementById("liveOutDot");
    const liveOutFiring = document.getElementById("liveOutFiring");

    if (liveH1Firing && h1) {
      liveH1Firing.textContent = `${h1.active_count}/${h1.total_count} (${h1.active_percentage}%)`;
      if (liveH1Dot) liveH1Dot.className = "pill-dot dot-h1 " + (h1.active_count > 0 ? "active" : "");
    }
    if (liveH2Firing && h2) {
      liveH2Firing.textContent = `${h2.active_count}/${h2.total_count} (${h2.active_percentage}%)`;
      if (liveH2Dot) liveH2Dot.className = "pill-dot dot-h2 " + (h2.active_count > 0 ? "active" : "");
    }
    if (liveOutFiring) {
      const isBad = act.probability_bad >= (act.threshold || 0.50);
      liveOutFiring.textContent = `${(act.probability_bad * 100).toFixed(1)}% (${isBad ? 'Perbaikan' : 'Baik'})`;
      if (liveOutDot) liveOutDot.className = "pill-dot dot-out " + (isBad ? "bad" : "good");
    }

    // Minigrid H1 (32 cells)
    const gridH1 = document.getElementById("miniGridH1");
    if (gridH1 && h1 && h1.post_activations) {
      gridH1.innerHTML = "";
      h1.post_activations.forEach((val, idx) => {
        const cell = document.createElement("div");
        const isActive = val > 0;
        cell.className = `mini-cell ${isActive ? "active-h1" : "inactive"}`;
        cell.title = `Neuron H1 #${idx}: Aktivasi = ${val.toFixed(3)} ${isActive ? '(Aktif)' : '(ReLU 0)'}`;
        gridH1.appendChild(cell);
      });
    }

    // Minigrid H2 (16 cells)
    const gridH2 = document.getElementById("miniGridH2");
    if (gridH2 && h2 && h2.post_activations) {
      gridH2.innerHTML = "";
      h2.post_activations.forEach((val, idx) => {
        const cell = document.createElement("div");
        const isActive = val > 0;
        cell.className = `mini-cell ${isActive ? "active-h2" : "inactive"}`;
        cell.title = `Neuron H2 #${idx}: Aktivasi = ${val.toFixed(3)} ${isActive ? '(Aktif)' : '(ReLU 0)'}`;
        gridH2.appendChild(cell);
      });
    }
  }

  function setUIState(state) {
    if (stateInitial) stateInitial.style.display = state === "initial" ? "block" : "none";
    if (stateLoading) stateLoading.style.display = state === "loading" ? "block" : "none";
    if (stateError) stateError.style.display = state === "error" ? "block" : "none";
    if (stateSuccess) stateSuccess.style.display = state === "success" ? "block" : "none";
  }

  // =========================================================================
  // INTERACTIVE NEURON NETWORK VISUALIZER CLASS
  // =========================================================================
  class NeuronVisualizer {
    constructor() {
      this.canvas = document.getElementById("neuronCanvas");
      if (!this.canvas) return;
      this.ctx = this.canvas.getContext("2d");
      this.container = document.getElementById("neuronCanvasContainer");
      this.tooltip = document.getElementById("neuronTooltip");

      this.weights = null;
      this.synapseMode = "firing";
      this.hoveredNode = null;
      this.selectedNode = null;
      this.pulseProgress = 0;
      this.animating = false;

      this.width = 960;
      this.height = 500;
      this.layers = [];

      this.features = [
        { key: "suhu", name: "Suhu", label: "Suhu Curing", unit: "°C", default: 160.0 },
        { key: "tekanan", name: "Tekanan", label: "Tekanan Cetakan", unit: "bar", default: 20.0 },
        { key: "waktu_curing", name: "Waktu", label: "Waktu Curing", unit: "menit", default: 11.5 },
        { key: "rasio_karet", name: "Rasio", label: "Rasio Karet", unit: "-", default: 0.60 },
        { key: "kekerasan_karet", name: "Kekerasan", label: "Kekerasan Karet", unit: "Shore A", default: 65.0 }
      ];

      this.simParams = {
        suhu: 160.0,
        tekanan: 20.0,
        waktu_curing: 11.5,
        rasio_karet: 0.60,
        kekerasan_karet: 65.0
      };

      // Default baseline activations so canvas is NEVER blank on initial load
      this.activations = {
        scaled_values: [0.08, -0.05, 0.12, -0.02, 0.05],
        output_logit: -3.27,
        probability_bad: 0.037,
        probability_good: 0.963,
        threshold: 0.50,
        decision_label: "Baik",
        layers: [
          {
            layer_index: 1,
            layer_name: "Hidden 1 (ReLU)",
            units: 32,
            active_count: 17,
            total_count: 32,
            active_percentage: 53.1,
            pre_activations: [0.82, -0.45, 1.15, -0.12, 0.44, -0.89, 0.71, 0.22, -0.33, 0.95, -0.61, 0.48, -0.18, 0.66, 1.42, -0.55, -0.32, 0.81, -0.44, 0.93, 0.38, -0.71, 0.59, -0.22, 1.05, -0.84, 0.47, -0.19, 0.73, -0.41, 0.88, -0.63],
            post_activations: [0.82, 0, 1.15, 0, 0.44, 0, 0.71, 0.22, 0, 0.95, 0, 0.48, 0, 0.66, 1.42, 0, 0, 0.81, 0, 0.93, 0.38, 0, 0.59, 0, 1.05, 0, 0.47, 0, 0.73, 0, 0.88, 0],
            biases: new Array(32).fill(0).map((_, i) => roundVal(-0.2 + (i % 5) * 0.1))
          },
          {
            layer_index: 2,
            layer_name: "Hidden 2 (ReLU)",
            units: 16,
            active_count: 5,
            total_count: 16,
            active_percentage: 31.3,
            pre_activations: [-0.62, 0.84, -0.39, -0.71, 1.12, -0.45, -0.88, 0.65, -0.51, -0.34, 0.92, -0.77, -0.29, 0.43, -0.66, -0.54],
            post_activations: [0, 0.84, 0, 0, 1.12, 0, 0, 0.65, 0, 0, 0.92, 0, 0, 0.43, 0, 0],
            biases: new Array(16).fill(0).map((_, i) => roundVal(-0.15 + (i % 4) * 0.08))
          },
          {
            layer_index: 3,
            layer_name: "Output Layer (Sigmoid)",
            units: 1,
            active_count: 0,
            total_count: 1,
            active_percentage: 0.0,
            pre_activations: [-3.27],
            post_activations: [0.037],
            biases: [0.12]
          }
        ]
      };

      this.debounceTimer = null;
    }

    async init() {
      if (!this.canvas) return;

      // 1. Controls setup
      const synSelect = document.getElementById("synapseModeSelect");
      if (synSelect) {
        synSelect.value = this.synapseMode;
        synSelect.addEventListener("change", (e) => {
          this.synapseMode = e.target.value;
          this.draw();
        });
      }

      // Presets
      document.getElementById("vizPresetIdeal")?.addEventListener("click", () => {
        this.setSimulationValues(157.5, 20.0, 11.5, 0.60, 65.0);
      });
      document.getElementById("vizPresetOvercure")?.addEventListener("click", () => {
        this.setSimulationValues(180.0, 24.5, 14.5, 0.75, 73.0);
      });
      document.getElementById("vizPresetUndercure")?.addEventListener("click", () => {
        this.setSimulationValues(140.0, 16.0, 8.5, 0.45, 57.0);
      });
      document.getElementById("vizSyncFromForm")?.addEventListener("click", () => {
        const s = parseFloat(document.getElementById("inputSuhu")?.value || 160.0);
        const t = parseFloat(document.getElementById("inputTekanan")?.value || 20.0);
        const w = parseFloat(document.getElementById("inputWaktu")?.value || 11.5);
        const r = parseFloat(document.getElementById("inputRasio")?.value || 0.60);
        const k = parseFloat(document.getElementById("inputKekerasan")?.value || 65.0);
        this.setSimulationValues(s, t, w, r, k);
      });

      // Simulation sliders in Tab 4
      this.bindSlider("simSliderSuhu", "simValSuhu", "suhu", "°C");
      this.bindSlider("simSliderTekanan", "simValTekanan", "tekanan", "bar");
      this.bindSlider("simSliderWaktu", "simValWaktu", "waktu_curing", "min");
      this.bindSlider("simSliderRasio", "simValRasio", "rasio_karet", "");
      this.bindSlider("simSliderKekerasan", "simValKekerasan", "kekerasan_karet", "Shore A");

      // Canvas mouse interactions
      this.canvas.addEventListener("mousemove", (e) => this.handleMouseMove(e));
      this.canvas.addEventListener("mouseleave", () => this.handleMouseLeave());
      this.canvas.addEventListener("click", (e) => this.handleClick(e));

      // Window resize
      window.addEventListener("resize", () => {
        this.setupCanvasSize();
        this.draw();
      });

      // Render baseline canvas immediately
      this.setupCanvasSize();
      this.updateTelemetry();
      this.draw();

      // 2. Fetch weights asynchronously
      try {
        const wResp = await fetch("/api/model/weights");
        if (wResp.ok) {
          this.weights = await wResp.json();
          this.draw();
        }
      } catch (err) {
        console.warn("Could not fetch model weights", err);
      }

      // 3. Fetch real initial activations from model
      await this.fetchSimulationActivations();
    }

    bindSlider(sliderId, labelId, key, unit) {
      const slider = document.getElementById(sliderId);
      const label = document.getElementById(labelId);
      if (!slider) return;

      slider.addEventListener("input", () => {
        const val = parseFloat(slider.value);
        this.simParams[key] = val;
        if (label) label.textContent = `${val.toFixed(key === "rasio_karet" ? 2 : 1)} ${unit}`.trim();
        this.debounceFetch();
      });
    }

    setSimulationValues(s, t, w, r, k = 65.0) {
      this.simParams = { suhu: s, tekanan: t, waktu_curing: w, rasio_karet: r, kekerasan_karet: k };
      const sSuhu = document.getElementById("simSliderSuhu");
      const sTek = document.getElementById("simSliderTekanan");
      const sWaktu = document.getElementById("simSliderWaktu");
      const sRasio = document.getElementById("simSliderRasio");
      const sKek = document.getElementById("simSliderKekerasan");

      if (sSuhu) sSuhu.value = s;
      if (sTek) sTek.value = t;
      if (sWaktu) sWaktu.value = w;
      if (sRasio) sRasio.value = r;
      if (sKek) sKek.value = k;

      if (document.getElementById("simValSuhu")) document.getElementById("simValSuhu").textContent = `${s.toFixed(1)} °C`;
      if (document.getElementById("simValTekanan")) document.getElementById("simValTekanan").textContent = `${t.toFixed(1)} bar`;
      if (document.getElementById("simValWaktu")) document.getElementById("simValWaktu").textContent = `${w.toFixed(1)} min`;
      if (document.getElementById("simValRasio")) document.getElementById("simValRasio").textContent = `${r.toFixed(2)}`;
      if (document.getElementById("simValKekerasan")) document.getElementById("simValKekerasan").textContent = `${k.toFixed(1)} Shore A`;

      this.fetchSimulationActivations();
    }

    debounceFetch() {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = setTimeout(() => {
        this.fetchSimulationActivations();
      }, 35);
    }

    async fetchSimulationActivations() {
      try {
        const resp = await fetch("/api/model/activations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(this.simParams)
        });
        if (resp.ok) {
          const act = await resp.json();
          this.setActivations(act);
        }
      } catch (err) {
        console.warn("Could not calculate activations", err);
      }
    }

    setActivations(act) {
      this.activations = act;
      this.updateTelemetry();
      if (this.hoveredNode || this.selectedNode) {
        this.updateInspector(this.hoveredNode || this.selectedNode);
      }
      this.triggerPulse();
      this.draw();
    }

    onTabVisible() {
      this.setupCanvasSize();
      this.draw();
      // Ensure layout after transition
      requestAnimationFrame(() => {
        this.setupCanvasSize();
        this.draw();
      });
    }

    setupCanvasSize() {
      if (!this.canvas || !this.container) return;
      const rect = this.container.getBoundingClientRect();
      const parentContainer = document.querySelector(".container");
      const fallbackWidth = parentContainer && parentContainer.clientWidth > 0 ? parentContainer.clientWidth - 48 : 960;
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(760, Math.floor(rect.width > 0 ? rect.width : fallbackWidth));
      const height = 500;

      this.canvas.width = Math.floor(width * dpr);
      this.canvas.height = Math.floor(height * dpr);
      this.canvas.style.width = width + "px";
      this.canvas.style.height = height + "px";

      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.width = width;
      this.height = height;

      this.computeLayout();
    }

    computeLayout() {
      const padLeft = 175;
      const padRight = 135;
      const padY = 55;
      const chartH = this.height - 2 * padY;
      const innerW = this.width - padLeft - padRight;

      // Layer 0: Input (5 nodes)
      const layer0 = [];
      const nInputs = this.features.length;
      for (let i = 0; i < nInputs; i++) {
        const y = padY + chartH * (i / (nInputs - 1));
        layer0.push({
          layerIdx: 0,
          nodeIdx: i,
          x: padLeft,
          y: y,
          radius: 16,
          key: this.features[i].key,
          name: this.features[i].name,
          label: this.features[i].label,
          unit: this.features[i].unit
        });
      }

      // Layer 1: Hidden 1 (32 nodes arranged in 2 sub-columns of 16 nodes)
      const layer1 = [];
      const h1Col1X = padLeft + innerW * 0.28;
      const h1Col2X = padLeft + innerW * 0.42;

      for (let i = 0; i < 32; i++) {
        const isCol2 = i >= 16;
        const rowIdx = isCol2 ? i - 16 : i;
        const x = isCol2 ? h1Col2X : h1Col1X;
        const y = padY + chartH * (rowIdx / 15);
        layer1.push({
          layerIdx: 1,
          nodeIdx: i,
          x: x,
          y: y,
          radius: 6.5,
          name: `H1 #${i}`
        });
      }

      // Layer 2: Hidden 2 (16 nodes in 1 column)
      const layer2 = [];
      const h2X = padLeft + innerW * 0.72;
      for (let i = 0; i < 16; i++) {
        const y = padY + chartH * (i / 15);
        layer2.push({
          layerIdx: 2,
          nodeIdx: i,
          x: h2X,
          y: y,
          radius: 8.5,
          name: `H2 #${i}`
        });
      }

      // Layer 3: Output (1 node)
      const layer3 = [{
        layerIdx: 3,
        nodeIdx: 0,
        x: this.width - padRight,
        y: this.height / 2,
        radius: 22,
        name: "Output Logit / Sigmoid"
      }];

      this.layers = [layer0, layer1, layer2, layer3];
    }

    triggerPulse() {
      this.pulseProgress = 0;
      if (!this.animating) {
        this.animating = true;
        this.runPulseLoop();
      }
    }

    runPulseLoop() {
      if (!this.animating) return;
      this.pulseProgress += 0.05;
      if (this.pulseProgress >= 1.0) {
        this.animating = false;
        this.pulseProgress = 0;
      }
      this.draw();
      if (this.animating) {
        requestAnimationFrame(() => this.runPulseLoop());
      }
    }

    draw() {
      if (!this.ctx || !this.layers.length) return;
      const ctx = this.ctx;

      // 1. Paint dark canvas background gradient
      ctx.fillStyle = "#090d16";
      ctx.fillRect(0, 0, this.width, this.height);
      const bgGrad = ctx.createRadialGradient(this.width / 2, this.height / 2, 40, this.width / 2, this.height / 2, this.width / 1.3);
      bgGrad.addColorStop(0, "#0e182c");
      bgGrad.addColorStop(1, "#070b14");
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, this.width, this.height);

      // Subtle grid dots
      ctx.fillStyle = "rgba(148, 163, 184, 0.04)";
      for (let gx = 20; gx < this.width; gx += 30) {
        for (let gy = 20; gy < this.height; gy += 30) {
          ctx.fillRect(gx, gy, 1.5, 1.5);
        }
      }

      // 2. Draw Layer Column Headers
      this.drawHeaders();

      // 3. Draw Synapses
      if (this.synapseMode !== "none") {
        this.drawSynapses();
      }

      // 4. Draw Nodes
      this.drawNodes();
    }

    drawHeaders() {
      const ctx = this.ctx;
      ctx.font = "700 11px JetBrains Mono, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const h1CenterX = (this.layers[1][0].x + this.layers[1][16].x) / 2;
      const headers = [
        { text: `INPUT (${this.features.length})`, x: this.layers[0][0].x, sub: "Ternormalisasi" },
        { text: "HIDDEN 1 (32 ReLU)", x: h1CenterX, sub: "Aktivasi Non-linear" },
        { text: "HIDDEN 2 (16 ReLU)", x: this.layers[2][0].x, sub: "Karakteristik Mutu" },
        { text: "OUTPUT (1)", x: this.layers[3][0].x, sub: "Sigmoid P(Y=1)" }
      ];

      headers.forEach(h => {
        ctx.fillStyle = "rgba(148, 163, 184, 0.9)";
        ctx.fillText(h.text, h.x, 22);
        ctx.font = "9px Plus Jakarta Sans, sans-serif";
        ctx.fillStyle = "rgba(100, 116, 139, 0.8)";
        ctx.fillText(h.sub, h.x, 36);
        ctx.font = "700 11px JetBrains Mono, monospace";
      });
    }

    drawSynapses() {
      const ctx = this.ctx;
      const act = this.activations;
      const w = this.weights?.layers || [];
      const hovered = this.hoveredNode || this.selectedNode;

      const h1Post = act?.layers?.[0]?.post_activations || [];
      const h2Post = act?.layers?.[1]?.post_activations || [];

      // Helper function to draw a synapse
      const drawSynapseLine = (fromNode, toNode, weight, isHighlighted, isSourceActive) => {
        if (!fromNode || !toNode) return;
        const isPos = weight >= 0;
        const mag = Math.min(1.0, Math.abs(weight));

        let strokeColor;
        let lineWidth = 0.75 + mag * 1.5;

        if (isHighlighted) {
          lineWidth = 2.2 + mag * 2.5;
          strokeColor = isPos ? "rgba(56, 189, 248, 0.95)" : "rgba(249, 115, 22, 0.95)";
        } else if (this.synapseMode === "selected") {
          return;
        } else if (this.synapseMode === "firing" && !isSourceActive) {
          return;
        } else {
          if (isSourceActive) {
            const alpha = 0.12 + mag * 0.28;
            strokeColor = isPos ? `rgba(56, 189, 248, ${alpha})` : `rgba(249, 115, 22, ${alpha})`;
          } else {
            strokeColor = "rgba(100, 116, 139, 0.05)";
            lineWidth = 0.5;
          }
        }

        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = lineWidth;
        ctx.beginPath();
        ctx.moveTo(fromNode.x, fromNode.y);
        ctx.lineTo(toNode.x, toNode.y);
        ctx.stroke();

        // Animated pulse wave particle along active connection
        if (this.animating && isSourceActive && !isHighlighted) {
          const t = this.pulseProgress;
          const px = fromNode.x + (toNode.x - fromNode.x) * t;
          const py = fromNode.y + (toNode.y - fromNode.y) * t;
          ctx.fillStyle = isPos ? "#38bdf8" : "#f97316";
          ctx.beginPath();
          ctx.arc(px, py, 1.75, 0, Math.PI * 2);
          ctx.fill();
        }
      };

      // 1. Synapses Layer 0 -> Layer 1 (Weights: shape 32x5)
      const w0 = w[0]?.weights || [];
      const numInputs = this.layers[0].length;
      for (let j = 0; j < 32; j++) {
        const toNode = this.layers[1][j];
        for (let i = 0; i < numInputs; i++) {
          const fromNode = this.layers[0][i];
          const weight = w0[j]?.[i] ?? (0.15 * ((j + i) % 2 === 0 ? 1 : -1));
          const isHigh = hovered && (
            (hovered.layerIdx === 0 && hovered.nodeIdx === i) ||
            (hovered.layerIdx === 1 && hovered.nodeIdx === j)
          );
          drawSynapseLine(fromNode, toNode, weight, isHigh, true);
        }
      }

      // 2. Synapses Layer 1 -> Layer 2 (Weights: shape 16x32)
      const w1 = w[1]?.weights || [];
      for (let k = 0; k < 16; k++) {
        const toNode = this.layers[2][k];
        for (let j = 0; j < 32; j++) {
          const fromNode = this.layers[1][j];
          const weight = w1[k]?.[j] ?? (0.12 * ((k + j) % 2 === 0 ? 1 : -1));
          const isSourceActive = (h1Post[j] || 0) > 0;
          const isHigh = hovered && (
            (hovered.layerIdx === 1 && hovered.nodeIdx === j) ||
            (hovered.layerIdx === 2 && hovered.nodeIdx === k)
          );
          drawSynapseLine(fromNode, toNode, weight, isHigh, isSourceActive);
        }
      }

      // 3. Synapses Layer 2 -> Layer 3 (Weights: shape 1x16)
      const w2 = w[2]?.weights || [];
      const outNode = this.layers[3][0];
      for (let k = 0; k < 16; k++) {
        const fromNode = this.layers[2][k];
        const weight = w2[0]?.[k] ?? (0.25 * (k % 2 === 0 ? 1 : -1));
        const isSourceActive = (h2Post[k] || 0) > 0;
        const isHigh = hovered && (
          (hovered.layerIdx === 2 && hovered.nodeIdx === k) ||
          (hovered.layerIdx === 3 && hovered.nodeIdx === 0)
        );
        drawSynapseLine(fromNode, outNode, weight, isHigh, isSourceActive);
      }
    }

    drawNodes() {
      const ctx = this.ctx;
      const act = this.activations;
      const hovered = this.hoveredNode;
      const selected = this.selectedNode;

      const h1Post = act?.layers?.[0]?.post_activations || [];
      const h2Post = act?.layers?.[1]?.post_activations || [];

      this.layers.forEach((layer, layerIdx) => {
        layer.forEach((node, nodeIdx) => {
          let isActive = false;
          let glowColor = "#38bdf8";
          let fillColor = "#1e293b";
          let strokeColor = "#334155";
          let actVal = 0;

          if (layerIdx === 0) {
            isActive = true;
            fillColor = "#0284c7";
            strokeColor = "#38bdf8";
            glowColor = "#38bdf8";
          } else if (layerIdx === 1) {
            actVal = h1Post[nodeIdx] || 0;
            isActive = actVal > 0;
            if (isActive) {
              const alpha = Math.min(1.0, 0.4 + actVal * 0.45);
              fillColor = `rgba(16, 185, 129, ${alpha})`;
              strokeColor = "#34d399";
              glowColor = "#10b981";
            }
          } else if (layerIdx === 2) {
            actVal = h2Post[nodeIdx] || 0;
            isActive = actVal > 0;
            if (isActive) {
              const alpha = Math.min(1.0, 0.4 + actVal * 0.45);
              fillColor = `rgba(245, 158, 11, ${alpha})`;
              strokeColor = "#fbbf24";
              glowColor = "#f59e0b";
            }
          } else if (layerIdx === 3) {
            isActive = true;
            const isBad = (act?.probability_bad || 0) >= (act?.threshold || 0.50);
            fillColor = isBad ? "#dc2626" : "#16a34a";
            strokeColor = isBad ? "#f87171" : "#4ade80";
            glowColor = isBad ? "#ef4444" : "#22c55e";
          }

          const isHovered = hovered && hovered.layerIdx === layerIdx && hovered.nodeIdx === nodeIdx;
          const isSelected = selected && selected.layerIdx === layerIdx && selected.nodeIdx === nodeIdx;

          ctx.save();
          if (isHovered || isSelected) {
            ctx.shadowBlur = 18;
            ctx.shadowColor = "#ffffff";
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius + 4, 0, Math.PI * 2);
            ctx.stroke();
          } else if (isActive) {
            ctx.shadowBlur = 11;
            ctx.shadowColor = glowColor;
          }

          // Node Circle
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
          ctx.fillStyle = fillColor;
          ctx.fill();
          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = 1.5;
          ctx.stroke();
          ctx.restore();

          // Labels & text
          if (layerIdx === 0) {
            // Label inside node
            ctx.font = "700 11px Plus Jakarta Sans, sans-serif";
            ctx.fillStyle = "#ffffff";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(`x${nodeIdx}`, node.x, node.y);

            // Label outside node (left)
            ctx.font = "600 11px Plus Jakarta Sans, sans-serif";
            ctx.fillStyle = "#cbd5e1";
            ctx.textAlign = "right";
            const curVal = this.simParams[node.key || (node.name === "Suhu" ? "suhu" : node.name === "Tekanan" ? "tekanan" : node.name === "Waktu" ? "waktu_curing" : node.name === "Kekerasan" ? "kekerasan_karet" : "rasio_karet")];
            const unitStr = node.unit && node.unit !== "-" ? ` ${node.unit}` : "";
            const valFormatted = node.key === "rasio_karet"
              ? (curVal !== undefined && !isNaN(curVal) ? Number(curVal).toFixed(2) : "-")
              : (curVal !== undefined && !isNaN(curVal) ? Number(curVal).toFixed(1) : "-");
            ctx.fillText(`${node.name}: ${valFormatted}${unitStr}`, node.x - 22, node.y);
          } else if (layerIdx === 3) {
            const pPct = ((act?.probability_bad || 0) * 100).toFixed(1);
            ctx.font = "700 11px JetBrains Mono, monospace";
            ctx.fillStyle = "#ffffff";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(`${pPct}%`, node.x, node.y);

            ctx.font = "700 11px Plus Jakarta Sans, sans-serif";
            ctx.fillStyle = "#f1f5f9";
            ctx.textAlign = "left";
            const outLabel = (act?.probability_bad || 0) >= (act?.threshold || 0.50) ? "PERBAIKAN" : "BAIK";
            ctx.fillText(outLabel, node.x + 28, node.y - 6);
            ctx.font = "10px JetBrains Mono, monospace";
            ctx.fillStyle = "#94a3b8";
            ctx.fillText(`Logit: ${act?.output_logit ?? '-'}`, node.x + 28, node.y + 10);
          } else {
            if (node.radius >= 6) {
              ctx.font = "600 8px JetBrains Mono, monospace";
              ctx.fillStyle = isActive ? "#ffffff" : "#64748b";
              ctx.textAlign = "center";
              ctx.textBaseline = "middle";
              ctx.fillText(nodeIdx, node.x, node.y);
            }
          }
        });
      });
    }

    handleMouseMove(e) {
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      let found = null;
      for (const layer of this.layers) {
        for (const node of layer) {
          const dx = mouseX - node.x;
          const dy = mouseY - node.y;
          if (Math.hypot(dx, dy) <= node.radius + 7) {
            found = node;
            break;
          }
        }
        if (found) break;
      }

      if (found !== this.hoveredNode) {
        this.hoveredNode = found;
        if (found) {
          this.showTooltip(found, e.clientX, e.clientY);
          this.updateInspector(found);
        } else {
          this.hideTooltip();
          if (this.selectedNode) this.updateInspector(this.selectedNode);
        }
        this.draw();
      } else if (found) {
        this.showTooltip(found, e.clientX, e.clientY);
      }
    }

    handleMouseLeave() {
      if (this.hoveredNode) {
        this.hoveredNode = null;
        this.hideTooltip();
        if (this.selectedNode) this.updateInspector(this.selectedNode);
        this.draw();
      }
    }

    handleClick() {
      if (this.hoveredNode) {
        this.selectedNode = this.hoveredNode;
        this.updateInspector(this.selectedNode);
        this.draw();
      }
    }

    showTooltip(node, clientX, clientY) {
      if (!this.tooltip || !this.container) return;
      const rect = this.container.getBoundingClientRect();
      const left = clientX - rect.left;
      const top = clientY - rect.top;

      const act = this.activations;
      let text = `<strong>${node.name}</strong>`;

      if (node.layerIdx === 0) {
        const val = this.simParams[node.key || (node.name === "Suhu" ? "suhu" : node.name === "Tekanan" ? "tekanan" : node.name === "Waktu" ? "waktu_curing" : node.name === "Kekerasan" ? "kekerasan_karet" : "rasio_karet")];
        const scaled = act?.scaled_values?.[node.nodeIdx] ?? 0;
        const unitStr = node.unit && node.unit !== "-" ? ` ${node.unit}` : "";
        text += `<br>Nilai Input: <b>${val}${unitStr}</b><br>Z-Score: <code>${Number(scaled).toFixed(3)}</code>`;
      } else if (node.layerIdx === 1 || node.layerIdx === 2) {
        const lRecord = act?.layers?.[node.layerIdx - 1];
        const pre = lRecord?.pre_activations?.[node.nodeIdx] ?? 0;
        const post = lRecord?.post_activations?.[node.nodeIdx] ?? 0;
        const isAct = post > 0;
        text += `<br>Pre-Aktivasi (z): <code>${Number(pre).toFixed(4)}</code>`;
        text += `<br>Pasca-ReLU (a): <code>${Number(post).toFixed(4)}</code>`;
        text += `<br>Status: <span style="color:${isAct ? '#34d399' : '#94a3b8'}">${isAct ? 'Aktif Firing ✓' : 'ReLU 0 (Nonaktif)'}</span>`;
      } else {
        const pBad = ((act?.probability_bad || 0) * 100).toFixed(2);
        text += `<br>Logit: <code>${act?.output_logit ?? '-'}</code><br>Prob Perbaikan: <b>${pBad}%</b>`;
      }

      this.tooltip.innerHTML = text;
      this.tooltip.style.left = `${left}px`;
      this.tooltip.style.top = `${top}px`;
      this.tooltip.style.display = "block";
    }

    hideTooltip() {
      if (this.tooltip) this.tooltip.style.display = "none";
    }

    updateInspector(node) {
      const insTitle = document.getElementById("insTitle");
      const insBadge = document.getElementById("insBadge");
      const insPreVal = document.getElementById("insPreVal");
      const insPreDesc = document.getElementById("insPreDesc");
      const insPostVal = document.getElementById("insPostVal");
      const insPostDesc = document.getElementById("insPostDesc");
      const insStatusVal = document.getElementById("insStatusVal");
      const insStatusDesc = document.getElementById("insStatusDesc");

      if (!insTitle || !node) return;
      const act = this.activations;

      if (node.layerIdx === 0) {
        const key = node.key || (node.name === "Suhu" ? "suhu" : node.name === "Tekanan" ? "tekanan" : node.name === "Waktu" ? "waktu_curing" : node.name === "Kekerasan" ? "kekerasan_karet" : "rasio_karet");
        const val = this.simParams[key];
        const scaled = act?.scaled_values?.[node.nodeIdx] ?? 0;
        const unitStr = node.unit && node.unit !== "-" ? ` ${node.unit}` : "";

        insTitle.textContent = `Fitur Input [x${node.nodeIdx}]: ${node.label} (${node.name})`;
        insBadge.className = "badge badge-primary";
        insBadge.textContent = "Input Layer";

        insPreVal.textContent = `${val}${unitStr}`;
        insPreDesc.textContent = "Nilai fisik mentah parameter proses";

        insPostVal.textContent = `z-score = ${Number(scaled).toFixed(4)}`;
        insPostDesc.textContent = "Hasil standardisasi (x - mean) / std";

        insStatusVal.className = "ins-value text-primary";
        insStatusVal.textContent = "Normal Disalurkan";
        insStatusDesc.textContent = "Mendistribusikan sinyal ke 32 neuron di Hidden Layer 1";
      } else if (node.layerIdx === 1 || node.layerIdx === 2) {
        const lRecord = act?.layers?.[node.layerIdx - 1];
        const pre = lRecord?.pre_activations?.[node.nodeIdx] ?? 0;
        const post = lRecord?.post_activations?.[node.nodeIdx] ?? 0;
        const bias = lRecord?.biases?.[node.nodeIdx] ?? 0;
        const isAct = post > 0;

        const lName = node.layerIdx === 1 ? "Hidden 1" : "Hidden 2";
        insTitle.textContent = `Neuron ${lName} [${node.name}]: Aktivasi Non-Linear ReLU`;
        insBadge.className = `badge ${isAct ? 'badge-success' : 'badge-outline'}`;
        insBadge.textContent = isAct ? "Aktif Firing ✓" : "ReLU Zeroed";

        insPreVal.textContent = Number(pre).toFixed(4);
        insPreDesc.textContent = "Kombinasi linier: z = ∑(w_i · a_in) + b";

        insPostVal.textContent = Number(post).toFixed(4);
        insPostDesc.textContent = "Hasil aktivasi: a = max(0, z)";

        insStatusVal.className = `ins-value ${isAct ? 'text-success' : 'text-muted'}`;
        insStatusVal.textContent = isAct ? "Aktif Meneruskan Sinyal" : "Mati (ReLU Sparsitas = 0)";
        insStatusDesc.textContent = `Bias parameter b = ${Number(bias).toFixed(4)}`;
      } else if (node.layerIdx === 3) {
        const logit = act?.output_logit ?? 0;
        const pBad = (act?.probability_bad || 0) * 100;
        const pGood = (act?.probability_good || 0) * 100;
        const isBad = (act?.probability_bad || 0) >= (act?.threshold || 0.50);

        insTitle.textContent = "Neuron Output: Logit & Klasifikasi Probabilitas Sigmoid";
        insBadge.className = `badge ${isBad ? 'badge-danger' : 'badge-success'}`;
        insBadge.textContent = isBad ? "Prediksi: Perlu Perbaikan" : "Prediksi: Baik";

        insPreVal.textContent = Number(logit).toFixed(4);
        insPreDesc.textContent = "Raw logit output dari lapisan Linear(16, 1)";

        insPostVal.textContent = `${pBad.toFixed(2)}% (Kelas 1)`;
        insPostDesc.textContent = "Probabilitas melalui fungsi Sigmoid: 1 / (1 + e^-logit)";

        insStatusVal.className = `ins-value ${isBad ? 'text-danger' : 'text-success'}`;
        insStatusVal.textContent = isBad ? "Perlu Perbaikan (Reject/Quarantine)" : "Baik (Homogen & Memenuhi Mutu)";
        insStatusDesc.textContent = `P(Baik) = ${pGood.toFixed(2)}% | Threshold = ${act?.threshold ?? 0.50}`;
      }
    }

    updateTelemetry() {
      const act = this.activations;
      if (!act || !act.layers) return;

      const h1 = act.layers[0];
      const h2 = act.layers[1];

      const telH1 = document.getElementById("telH1Text");
      const telH2 = document.getElementById("telH2Text");
      const telOut = document.getElementById("telOutText");

      if (telH1 && h1) {
        const pct = h1.active_percentage.toFixed(1);
        const zeroPct = (100 - h1.active_percentage).toFixed(1);
        telH1.textContent = `${h1.active_count}/${h1.total_count} Aktif (${pct}%) | ReLU 0: ${zeroPct}%`;
      }
      if (telH2 && h2) {
        const pct = h2.active_percentage.toFixed(1);
        const zeroPct = (100 - h2.active_percentage).toFixed(1);
        telH2.textContent = `${h2.active_count}/${h2.total_count} Aktif (${pct}%) | ReLU 0: ${zeroPct}%`;
      }
      if (telOut) {
        const pBad = ((act.probability_bad || 0) * 100).toFixed(1);
        const isBad = (act.probability_bad || 0) >= (act.threshold || 0.50);
        telOut.textContent = `Logit: ${act.output_logit} ➔ ${pBad}% (${isBad ? 'Perbaikan' : 'Baik'})`;
      }
    }
  }

  function roundVal(v) {
    return Math.round(v * 10000) / 10000;
  }

  // 5. Initial Data Load for Health, Model Info, Evaluation, History
  async function loadInitialData() {
    // 1. Instansiasi visualizer neuron sesegera mungkin
    try {
      window.neuronVisualizer = new NeuronVisualizer();
      await window.neuronVisualizer.init();
    } catch (e) {
      console.warn("Could not initialize neuron visualizer", e);
    }

    // Health check
    try {
      const healthResp = await fetch("/health");
      const health = await healthResp.json();
      if (health.model_loaded) {
        if (serviceStatusBadge) {
          serviceStatusBadge.className = "badge badge-success";
          serviceStatusBadge.innerHTML = `<span class="status-dot active"></span> Model Aktif (${health.model_version})`;
        }
      } else {
        if (serviceStatusBadge) {
          serviceStatusBadge.className = "badge badge-danger";
          serviceStatusBadge.innerHTML = `<span class="status-dot error"></span> Model Belum Siap`;
        }
      }
    } catch {
      if (serviceStatusBadge) {
        serviceStatusBadge.className = "badge badge-danger";
        serviceStatusBadge.innerHTML = `<span class="status-dot error"></span> Server Offline`;
      }
    }

    // Evaluation & Model Info
    try {
      const evalResp = await fetch("/api/evaluation");
      if (evalResp.ok) {
        const evalData = await evalResp.json();
        renderEvaluationData(evalData);
      }
    } catch (e) {
      console.warn("Could not fetch evaluation data", e);
    }

    // Training History
    try {
      const histResp = await fetch("/api/training-history");
      if (histResp.ok) {
        const histData = await histResp.json();
        renderTrainingHistory(histData);
      }
    } catch (e) {
      console.warn("Could not fetch training history", e);
    }

    // Scaler Parameters (StandardScaler)
    try {
      const scalerResp = await fetch("/api/scaler");
      if (scalerResp.ok) {
        const scalerData = await scalerResp.json();
        const snippetEl = document.getElementById("scalerJsonSnippet");
        if (snippetEl) {
          snippetEl.textContent = JSON.stringify(scalerData, null, 2);
        }
      }
    } catch (e) {
      console.warn("Could not fetch scaler data", e);
    }

    // Full Model Config
    try {
      const configResp = await fetch("/api/model-config");
      if (configResp.ok) {
        const configData = await configResp.json();
        const configEl = document.getElementById("configJsonDisplay");
        if (configEl) {
          configEl.textContent = JSON.stringify(configData, null, 2);
        }
      }
    } catch (e) {
      console.warn("Could not fetch model config", e);
    }
  }

  function renderEvaluationData(data) {
    const m = data.metrics || {};
    // KPI Cards
    if (document.getElementById("kpiAccuracy")) document.getElementById("kpiAccuracy").textContent = (m.accuracy * 100).toFixed(1) + "%";
    if (document.getElementById("kpiRecall1")) document.getElementById("kpiRecall1").textContent = (m.recall_class_1 * 100).toFixed(1) + "%";
    if (document.getElementById("kpiPrecision1")) document.getElementById("kpiPrecision1").textContent = (m.precision_class_1 * 100).toFixed(1) + "%";
    if (document.getElementById("kpiF1")) document.getElementById("kpiF1").textContent = (m.f1_class_1 * 100).toFixed(1) + "%";
    if (document.getElementById("evalThresholdVal")) document.getElementById("evalThresholdVal").textContent = data.threshold;

    // Confusion Matrix
    const cm = data.confusion_matrix || [[0, 0], [0, 0]];
    if (document.getElementById("cmTN")) document.getElementById("cmTN").textContent = cm[0][0];
    if (document.getElementById("cmFP")) document.getElementById("cmFP").textContent = cm[0][1];
    if (document.getElementById("cmFN")) document.getElementById("cmFN").textContent = cm[1][0];
    if (document.getElementById("cmTP")) document.getElementById("cmTP").textContent = cm[1][1];

    // Baseline Comparison
    const baseM = data.baseline_comparison?.metrics || {};
    if (document.getElementById("mlpAcc")) document.getElementById("mlpAcc").textContent = (m.accuracy * 100).toFixed(1) + "%";
    if (document.getElementById("baseAcc")) document.getElementById("baseAcc").textContent = (baseM.accuracy * 100).toFixed(1) + "%";

    if (document.getElementById("mlpRec")) document.getElementById("mlpRec").textContent = (m.recall_class_1 * 100).toFixed(1) + "%";
    if (document.getElementById("baseRec")) document.getElementById("baseRec").textContent = (baseM.recall_class_1 * 100).toFixed(1) + "%";

    if (document.getElementById("mlpF1")) document.getElementById("mlpF1").textContent = (m.f1_class_1 * 100).toFixed(1) + "%";
    if (document.getElementById("baseF1")) document.getElementById("baseF1").textContent = (baseM.f1_class_1 * 100).toFixed(1) + "%";

    if (document.getElementById("mlpRoc")) document.getElementById("mlpRoc").textContent = m.roc_auc ? m.roc_auc.toFixed(4) : "-";
    if (document.getElementById("baseRoc")) document.getElementById("baseRoc").textContent = baseM.roc_auc ? baseM.roc_auc.toFixed(4) : "-";

    // Training summary
    const s = data.training_summary || {};
    if (document.getElementById("bestEpochVal")) document.getElementById("bestEpochVal").textContent = s.best_epoch || "-";
    if (document.getElementById("totalEpochsVal")) document.getElementById("totalEpochsVal").textContent = s.total_epochs_trained || "-";
    if (document.getElementById("bestValLossVal")) document.getElementById("bestValLossVal").textContent = s.best_val_loss || "-";
    if (document.getElementById("posWeightVal")) document.getElementById("posWeightVal").textContent = s.pos_weight || "-";

    // Config JSON Display
    if (document.getElementById("configJsonDisplay")) {
      document.getElementById("configJsonDisplay").textContent = JSON.stringify(
        {
          model_version: data.model_version,
          features: data.features,
          threshold: data.threshold,
          split: data.split,
          training_summary: data.training_summary
        },
        null,
        2
      );
    }
  }

  // Draw pure HTML5 Canvas Line Chart for training history
  function renderTrainingHistory(history) {
    const canvas = document.getElementById("lossChart");
    if (!canvas || !history || !Array.isArray(history) || history.length === 0) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const padLeft = 45;
    const padRight = 20;
    const padTop = 20;
    const padBottom = 30;

    const chartW = width - padLeft - padRight;
    const chartH = height - padTop - padBottom;

    // Find min and max loss
    const trainLosses = history.map(h => h.train_loss);
    const valLosses = history.map(h => h.val_loss);
    const allLosses = [...trainLosses, ...valLosses];

    const maxLoss = Math.max(...allLosses) * 1.05;
    const minLoss = Math.max(0, Math.min(...allLosses) * 0.95);

    // Draw grid lines
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.fillStyle = "#94a3b8";

    const ySteps = 4;
    for (let i = 0; i <= ySteps; i++) {
      const yVal = minLoss + (maxLoss - minLoss) * (i / ySteps);
      const yPos = padTop + chartH - (i / ySteps) * chartH;
      ctx.beginPath();
      ctx.moveTo(padLeft, yPos);
      ctx.lineTo(width - padRight, yPos);
      ctx.stroke();
      ctx.fillText(yVal.toFixed(2), 5, yPos + 3);
    }

    function getX(idx) {
      return padLeft + (idx / Math.max(1, history.length - 1)) * chartW;
    }
    function getY(loss) {
      return padTop + chartH - ((loss - minLoss) / Math.max(0.0001, maxLoss - minLoss)) * chartH;
    }

    // Draw Train Loss (blue)
    ctx.strokeStyle = "#0284c7";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    history.forEach((h, i) => {
      const x = getX(i);
      const y = getY(h.train_loss);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw Val Loss (orange)
    ctx.strokeStyle = "#ea580c";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    history.forEach((h, i) => {
      const x = getX(i);
      const y = getY(h.val_loss);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw X-axis label
    ctx.fillStyle = "#64748b";
    ctx.fillText("Epoch 1", padLeft, height - 8);
    const lastEpoch = history[history.length - 1].epoch;
    ctx.fillText(`Epoch ${lastEpoch}`, width - padRight - 55, height - 8);
  }

  // Run initialization
  loadInitialData();
});
