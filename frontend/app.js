// ═══════════════════════════════════════════════════════════
// Mint Leaf AI — Frontend Application Logic  (v1.0)
// ═══════════════════════════════════════════════════════════

// Use same origin so this works when served through FastAPI static mount.
// Change BASE_URL if you run the frontend and backend on different ports.
const BASE_URL = "";

let selectedFile       = null;
let webcamStream       = null;
let currentOriginalSrc = null;   // for XAI toggle
let xaiShowing         = true;

// ── Tab switching ────────────────────────────────────────────────────────────
function switchTab(tabName) {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => (c.style.display = "none"));

    const tabEl  = document.getElementById(`tab-${tabName}`);
    const btnEl  = document.querySelector(`button[onclick="switchTab('${tabName}')"]`);
    if (tabEl)  tabEl.style.display  = "block";
    if (btnEl)  btnEl.classList.add("active");

    if (tabName === "benchmark") loadBenchmarkTable();
}

// ── Drag-and-drop ────────────────────────────────────────────────────────────
const dropZone  = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");

dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("hover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("hover"));
dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("hover");
    if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", e => {
    if (e.target.files.length) handleFileSelect(e.target.files[0]);
});

function handleFileSelect(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        currentOriginalSrc = e.target.result;
        document.getElementById("image-preview").src = e.target.result;
        document.getElementById("drop-zone").style.display       = "none";
        document.getElementById("preview-container").style.display = "block";
        runDiagnosis(file);
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    document.getElementById("drop-zone").style.display         = "block";
    document.getElementById("preview-container").style.display = "none";
    document.getElementById("empty-state").style.display       = "block";
    document.getElementById("results-content").style.display   = "none";
    document.getElementById("loading-state").style.display     = "none";
    currentOriginalSrc = null;
    xaiShowing = true;
}

// ── Webcam ───────────────────────────────────────────────────────────────────
async function openCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        webcamStream = stream;
        document.getElementById("webcam").srcObject = stream;
        document.getElementById("camera-modal").style.display = "block";
        document.getElementById("drop-zone").style.display    = "none";
    } catch (err) {
        alert("Camera access denied: " + err.message);
    }
}

function closeCamera() {
    if (webcamStream) webcamStream.getTracks().forEach(t => t.stop());
    document.getElementById("camera-modal").style.display = "none";
    document.getElementById("drop-zone").style.display    = "block";
}

function captureWebcam() {
    const video  = document.getElementById("webcam");
    const canvas = document.createElement("canvas");
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
        const f = new File([blob], "webcam_capture.jpg", { type: "image/jpeg" });
        closeCamera();
        handleFileSelect(f);
    }, "image/jpeg");
}

// ── Diagnosis API call ───────────────────────────────────────────────────────
async function runDiagnosis(file) {
    document.getElementById("empty-state").style.display     = "none";
    document.getElementById("results-content").style.display = "none";
    document.getElementById("loading-state").style.display   = "block";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${BASE_URL}/api/diagnose`, { method: "POST", body: formData });

        if (response.status === 503) {
            const err = await response.json();
            showBanner("🔴 Model Offline: " + err.detail, "error");
            document.getElementById("loading-state").style.display = "none";
            document.getElementById("empty-state").style.display   = "block";
            return;
        }
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        renderResults(data);
    } catch (err) {
        showBanner("❌ Diagnosis error: " + err.message, "error");
        document.getElementById("loading-state").style.display = "none";
        document.getElementById("empty-state").style.display   = "block";
    }
}

// ── Render results ───────────────────────────────────────────────────────────
function renderResults(data) {
    document.getElementById("loading-state").style.display   = "none";
    document.getElementById("results-content").style.display = "block";

    const diag = data.diagnostic_result;
    const adv  = data.agronomic_advisory;
    const qual = data.quality_prescreening;

    // Header
    document.getElementById("disease-name").textContent  = diag.disease_display_name;
    document.getElementById("confidence-val").textContent = `${diag.confidence_pct}%`;
    document.getElementById("severity-badge").textContent = adv.severity_level;

    // Quality meter
    document.getElementById("q-icon").textContent = qual.is_valid ? "✅" : "⚠️";
    document.getElementById("q-text").textContent =
        `Blur score: ${qual.blur_score}  |  Brightness: ${qual.brightness_score}  |  ${qual.quality_status}`;

    // Grad-CAM heatmap
    const xaiImg    = document.getElementById("xai-image");
    const xaiNotice = document.getElementById("xai-notice");
    if (data.xai_heatmap_base64) {
        xaiImg.src                = data.xai_heatmap_base64;
        xaiImg.style.display      = "block";
        if (xaiNotice) xaiNotice.style.display = "none";
        xaiShowing = true;
    } else {
        xaiImg.style.display = "none";
        if (xaiNotice) {
            xaiNotice.textContent   = data.xai_notice || "Grad-CAM unavailable.";
            xaiNotice.style.display = "block";
        }
    }

    // Probability bars
    const container = document.getElementById("prob-bars");
    container.innerHTML = "";
    const probs = diag.class_probabilities;
    Object.entries(probs)
        .sort(([, a], [, b]) => b - a)
        .forEach(([cls, pct]) => {
            const row = document.createElement("div");
            row.className = "prob-bar-row";
            row.innerHTML = `
                <div class="prob-meta">
                    <span>${cls.replace(/_/g, " ")}</span>
                    <span>${pct.toFixed(2)}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:${Math.min(pct, 100)}%"></div>
                </div>`;
            container.appendChild(row);
        });

    // Advisory
    document.getElementById("adv-organic").textContent   = adv.organic_treatment;
    document.getElementById("adv-chemical").textContent  = adv.chemical_treatment;

    const ul = document.getElementById("adv-cultural");
    ul.innerHTML = "";
    (adv.cultural_controls || []).forEach(ctrl => {
        const li = document.createElement("li");
        li.textContent = ctrl;
        ul.appendChild(li);
    });

    // Latency
    const latEl = document.getElementById("latency-val");
    if (latEl && data.performance_metrics) {
        latEl.textContent = `${data.performance_metrics.total_request_latency_ms} ms`;
    }
}

// ── XAI toggle ───────────────────────────────────────────────────────────────
function toggleXAI() {
    const xaiImg = document.getElementById("xai-image");
    const preview = document.getElementById("image-preview");
    if (!xaiImg.src || xaiImg.src === window.location.href) return;  // no heatmap loaded

    if (xaiShowing) {
        xaiImg.src   = currentOriginalSrc;
        xaiShowing   = false;
    } else {
        const data = window._lastDiagData;
        if (data && data.xai_heatmap_base64) {
            xaiImg.src = data.xai_heatmap_base64;
        }
        xaiShowing = true;
    }
}

// ── Banner / toast ───────────────────────────────────────────────────────────
function showBanner(msg, type = "info") {
    let banner = document.getElementById("global-banner");
    if (!banner) {
        banner = document.createElement("div");
        banner.id = "global-banner";
        banner.style.cssText = (
            "position:fixed;top:16px;left:50%;transform:translateX(-50%);"
            "padding:12px 24px;border-radius:12px;font-weight:600;z-index:9999;"
            "max-width:600px;text-align:center;animation:fadein 0.3s ease;"
        );
        document.body.appendChild(banner);
    }
    banner.textContent  = msg;
    banner.style.background = type === "error" ? "rgba(255,23,68,0.9)" : "rgba(0,230,118,0.9)";
    banner.style.color  = "#fff";
    banner.style.display = "block";
    setTimeout(() => { banner.style.display = "none"; }, 6000);
}

// ── Benchmark table ──────────────────────────────────────────────────────────
async function loadBenchmarkTable() {
    const tbody = document.getElementById("benchmark-tbody");
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;color:#9ca3af;">Loading…</td></tr>`;

    try {
        const res  = await fetch(`${BASE_URL}/api/benchmark`);
        const json = await res.json();

        if (json.notice && (!json.data || json.data.length === 0)) {
            tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;color:#9ca3af;">${json.notice}</td></tr>`;
            return;
        }

        tbody.innerHTML = "";
        (json.data || json).forEach((row, idx) => {
            const acc = row.accuracy != null ? (row.accuracy * 100).toFixed(2) + "%" : (row.test_accuracy || "-");
            const f1  = row.macro_f1 != null ? parseFloat(row.macro_f1).toFixed(4) : "-";
            const ext = row.external_macro_f1 != null ? parseFloat(row.external_macro_f1).toFixed(4) : "-";
            const lat = row.inference_latency_ms || row.latency_ms || "-";
            const sz  = row.checkpoint_size_mb   || row.size_mb   || "-";
            const tier = row.performance_tier || row.tier || "-";

            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>#${idx + 1}</strong></td>
                <td>${row.model_name || row.model || "-"}</td>
                <td>${row.family || "-"}</td>
                <td><strong>${acc}</strong></td>
                <td><strong>${f1}</strong></td>
                <td>${ext}</td>
                <td>${lat} ms</td>
                <td>${sz} MB</td>
                <td><span class="badge">${tier}</span></td>`;
            tbody.appendChild(tr);
        });
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="9" style="color:#ff1744;text-align:center;">Failed to load: ${err.message}</td></tr>`;
    }
}
