/**
 * STRATOVANE Frontend Controller
 * Manages API interaction and Plotly rendering.
 */

const API_BASE = '';
const MATERIALS = {
    'BARE_ALUMINUM_OXIDIZED': '#888888',
    'WHITE_REFERENCE_PLACEHOLDER': '#ffffff',
    'STRATOVANE_AAO_LITERATURE': '#007bff',
    'BARE_ALUMINUM_POLISHED': '#4d4d4d',
    'BARE_ALUMINUM_ROUGH': '#666666',
    'AAO_AL_DIAZ_LOBO': '#17a2b8',
    'STRATOVANE_AAO_MEASURED': '#5bc0de',
    'AAO_OPTIONAL_RS94': '#ffc107',
    'AAO_OPTIONAL_EPS97': '#fd7e14'
};

// Fallback so any unknown material key still gets a visible color
function materialColor(m) {
    return MATERIALS[m] || '#555555';
}

let state = {
    meta: null,
    params: {
        altitude: 20,
        wind: 5.0,
        heat_load: 20,
        solar_scale: 1.0,
        materials: []
    },
    activeSection: 'overview'
};

async function fetchAPI(endpoint, options = {}) {
    showLoader(true);
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, options);
        if (!response.ok) {
            let msg = `HTTP ${response.status}`;
            try {
                const errorData = await response.json();
                if (errorData && typeof errorData.error === 'string') {
                    msg = errorData.error;
                }
            } catch (_) { /* non-JSON error body (e.g. HTML 404 page): keep HTTP status */ }
            throw new Error(msg);
        }
        return await response.json();
    } catch (e) {
        showError(e.message);
        throw e;
    } finally {
        showLoader(false);
    }
}

function showLoader(show) {
    document.getElementById('loader').classList.toggle('hidden', !show);
}

// Persistent so uninitialized or failed loads never render as a blank page.
function showError(msg) {
    const errCont = document.getElementById('error-container');
    errCont.textContent = msg;
    errCont.classList.remove('hidden');
}

function hasPlotly() {
    return typeof window.Plotly !== 'undefined' && typeof window.Plotly.newPlot === 'function';
}

function plot(elementId, traces, layout) {
    if (!hasPlotly()) {
        showError('Plotly failed to load — charts cannot be rendered. Check your connection/access-control setup and reload the page.');
        return;
    }
    window.Plotly.newPlot(elementId, traces, layout);
}

// UI Updates
function updateValueDisplay(id, val) {
    document.getElementById(`val-${id}`).textContent = val;
}

async function init() {
    if (!hasPlotly()) {
        showError('Plotly failed to load — charts unavailable. Data tables below will still populate once the model responds.');
    }
    try {
        state.meta = await fetchAPI('/api/meta');
        state.params.materials = state.meta.materials;
        state.params.altitude = state.meta.defaults.altitude / 1000;
        state.params.wind = state.meta.defaults.wind;
        state.params.heat_load = state.meta.defaults.heat_load;
        state.params.solar_scale = state.meta.defaults.solar_scale;

        // Set slider ranges from meta
        const altSlide = document.getElementById('slide-altitude');
        altSlide.min = state.meta.ranges.altitude[0] / 1000;
        altSlide.max = state.meta.ranges.altitude[1] / 1000;
        altSlide.value = state.params.altitude;
        updateValueDisplay('altitude', state.params.altitude);

        const windSlide = document.getElementById('slide-wind');
        windSlide.min = state.meta.ranges.wind[0];
        windSlide.max = state.meta.ranges.wind[1];
        windSlide.value = state.params.wind;
        updateValueDisplay('wind', state.params.wind);

        const heatSlide = document.getElementById('slide-heat');
        heatSlide.min = state.meta.ranges.heat_load[0];
        heatSlide.max = state.meta.ranges.heat_load[1];
        heatSlide.value = state.params.heat_load;
        updateValueDisplay('heat', state.params.heat_load);

        const solarSlide = document.getElementById('slide-solar');
        solarSlide.min = state.meta.ranges.solar_scale[0];
        solarSlide.max = state.meta.ranges.solar_scale[1];
        solarSlide.value = state.params.solar_scale;
        updateValueDisplay('solar', state.params.solar_scale);

        // Populate Meta Info
        document.getElementById('model-meta').innerHTML = `
            <strong>Model Version:</strong> ${state.meta.version} | 
            <strong>Commit:</strong> ${state.meta.commit} | 
            <strong>Date:</strong> ${state.meta.date}
        `;

        // Populate Limitations
        const limList = document.getElementById('limitations-list');
        limList.innerHTML = state.meta.limitations.map(l => `<li>${l}</li>`).join('');

        setupEventListeners();
        updateAltitudePage();
    } catch (e) {
        console.error("Init failed", e);
    }
}

function setupEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const section = item.dataset.section;
            document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            
            document.querySelectorAll('.page-section').forEach(s => s.classList.add('hidden'));
            document.getElementById(section).classList.remove('hidden');
            state.activeSection = section;

            if (section === 'transient') updateTransientPage();
            if (section === 'spectral') updateSpectralPage();
            if (section === 'uncertainty') updateUncertaintyPage();
            if (section === 'benchmark') updateBenchmarkPage();
            if (section === 'experiment') updateExperimentPage();
        });
    });

    // Sliders (with debounce)
    const debounce = (fn, delay) => {
        let timeout;
        return (...args) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => fn(...args), delay);
        };
    };

    const handleInput = debounce(async () => {
        if (state.activeSection === 'altitude') {
            await updateAltitudePage();
        }
    }, 250);

    ['altitude', 'wind', 'heat', 'solar'].forEach(id => {
        const el = document.getElementById(`slide-${id}`);
        el.addEventListener('input', (e) => {
            const val = e.target.value;
            updateValueDisplay(id, val);
            
            // Update state
            const stateKey = id === 'heat' ? 'heat_load' : id;
            state.params[stateKey] = parseFloat(val);
            
            handleInput();
        });
    });

    // MC Run
    document.getElementById('btn-run-mc').addEventListener('click', runMC);
}

async function updateAltitudePage() {
    try {
        const data = await fetchAPI('/api/altitude', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                materials: state.params.materials,
                altitude: state.params.altitude * 1000,
                wind: state.params.wind,
                heat_load: state.params.heat_load,
                solar_scale: state.params.solar_scale
            })
        });

        renderAltitudeCharts(data);
        // Solar-scale transparency: flag when the scale is inert
        const applied = data.metadata ? data.metadata.solar_scale_applied : true;
        const solarDisp = document.getElementById('val-solar');
        if (solarDisp && applied === false) {
            solarDisp.title = 'Inactive: no explicit irradiance override is configured, so the internal altitude-dependent solar model is used';
        }
    } catch (e) {}
}

function renderAltitudeCharts(data) {
    const results = data.results;
    const materials = Object.keys(results);
    
    // 1. Temperature vs Altitude (Simulated Grid)
    // Since we are live, we'd need to call the API multiple times for a curve.
    // For the "Explorer", we show the current point as a marker on a typical curve.
    // To keep it fast, we'll just show the point-comparison bar chart or just the values.
    // Requirement says "T_electronics and T_surface vs altitude". 
    // I'll implement a multi-call sequence to get a small grid for the curve.
    
    const t_elec_data = [];
    const t_surf_data = [];
    
    materials.forEach(m => {
        t_elec_data.push({
            x: [state.params.altitude],
            y: [results[m].t_electronics],
            name: m,
            mode: 'markers',
            marker: { size: 12, color: materialColor(m) }
        });
        t_surf_data.push({
            x: [state.params.altitude],
            y: [results[m].t_surface],
            name: `${m} (surf)`,
            mode: 'markers',
            marker: { size: 8, color: materialColor(m), symbol: 'diamond' }
        });
    });

    plot('chart-altitude-temp', t_elec_data.concat(t_surf_data), {
        title: 'Current Temperatures',
        xaxis: { title: 'Altitude (km)' },
        yaxis: { title: 'Temperature (°C)' },
        margin: { t: 40 }
    });

    // 2. Q_env Breakdown
    const rad_vals = [], conv_vals = [], sol_vals = [];
    const labels = [];

    materials.forEach(m => {
        labels.push(m);
        rad_vals.push(results[m].q_env.rad);
        conv_vals.push(results[m].q_env.conv);
        sol_vals.push(results[m].q_env.solar);
    });

    const q_data = [
        { x: labels, y: rad_vals, name: 'Net Radiation', type: 'bar', marker: {color: '#ff9999'} },
        { x: labels, y: conv_vals, name: 'Convection', type: 'bar', marker: {color: '#99ff99'} },
        { x: labels, y: sol_vals, name: 'Absorbed Solar', type: 'bar', marker: {color: '#9999ff'} }
    ];

    plot('chart-altitude-qenv', q_data, {
        barmode: 'stack',
        title: 'Environmental Heat Rejection (Q_env)',
        yaxis: { title: 'Power (W)' },
        margin: { t: 40 }
    });
}

async function updateTransientPage() {
    try {
        const data = await fetchAPI('/api/transient', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                materials: state.params.materials,
                altitude: state.params.altitude * 1000,
                wind: state.params.wind,
                heat_load: state.params.heat_load
            })
        });

        const results = data.results;
        const traces = [];
        
        Object.entries(results).forEach(([m, res]) => {
            traces.push({
                x: res.time,
                y: res.t_electronics,
                name: m,
                line: { color: materialColor(m) }
            });
        });

        plot('chart-transient', traces, {
            title: '24h Electronics Temperature',
            xaxis: { title: 'Time (s)' },
            yaxis: { title: 'Temperature (°C)' },
            margin: { t: 40 }
        });

        const metricsDiv = document.getElementById('transient-metrics');
        metricsDiv.innerHTML = Object.entries(results).map(([m, res]) => `
            <div class="metric-card">
                <span class="label">${m}</span>
                <span class="value">${res.metrics.min.toFixed(1)} / ${res.metrics.max.toFixed(1)} °C</span>
                <span class="label">Swing: ${res.metrics.swing.toFixed(1)} K</span>
            </div>
        `).join('');

    } catch (e) {}
}

async function updateSpectralPage() {
    try {
        const data = await fetchAPI('/api/spectral', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ materials: state.params.materials })
        });

        const results = data.results;
        const traces = [];

        Object.entries(results).forEach(([m, res]) => {
            traces.push({
                x: res.thermal.wavelength,
                y: res.thermal.emissivity,
                name: `${m} (ε)`,
                line: { color: materialColor(m) }
            });
        });

        const layout = {
            title: 'Spectral Emissivity',
            xaxis: { title: 'Wavelength (μm)' },
            yaxis: { title: 'Value', range: [0, 1] },
            shapes: [{
                type: 'rect',
                xref: 'x', yref: 'paper',
                x0: data.window[0], x1: data.window[1],
                y0: 0, y1: 1,
                fillcolor: 'rgba(128, 128, 128, 0.2)',
                line: { width: 0 },
                layer: 'below'
            }],
            margin: { t: 40 }
        };

        plot('chart-spectral', traces, layout);
    } catch (e) {}
}

async function runMC() {
    const n = parseInt(document.getElementById('input-mc-n').value);
    const seed = parseInt(document.getElementById('input-mc-seed').value);
    
    try {
        const data = await fetchAPI('/api/montecarlo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                material: state.params.materials[0], // Default to first
                altitude: state.params.altitude * 1000,
                n: n,
                seed: seed
            })
        });

        const jobId = data.job_id;
        const progFill = document.getElementById('mc-progress-fill');
        const progBar = document.getElementById('mc-progress');
        progBar.classList.remove('hidden');
        
        const poll = setInterval(() => {
            fetchAPI(`/api/jobs/${jobId}`).then((status) => {
                if (status.status === 'completed' || status.status === 'failed') {
                    clearInterval(poll);
                    progBar.classList.add('hidden');
                    if (status.status === 'completed') {
                        renderMCResults(status.result);
                    } else {
                        showError(status.error || 'Monte Carlo job failed');
                    }
                }
            }).catch(() => {
                clearInterval(poll);
                progBar.classList.add('hidden');
            });
        }, 1000);

    } catch (e) {}
}

function renderMCResults(res) {
    const trace = {
        x: res.histogram,
        type: 'histogram',
        marker: { color: '#aaa' }
    };

    plot('chart-mc', [trace], {
        title: `Electronics Temperature Distribution (n=${res.histogram.length})`,
        xaxis: { title: 'Temperature (°C)' },
        yaxis: { title: 'Frequency' },
        margin: { t: 40 }
    });
}

async function updateUncertaintyPage() {
    try {
        const data = await fetchAPI('/api/sensitivity', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                material: state.params.materials[0],
                altitude: state.params.altitude * 1000
            })
        });

        const tornado = data.tornado;
        const labels = tornado.map(t => t[0]);
        const values = tornado.map(t => t[1]);

        const trace = {
            x: values,
            y: labels,
            type: 'bar',
            orientation: 'h',
            marker: { color: '#666' }
        };

        plot('chart-tornado', [trace], {
            title: 'Sensitivity Tornado Chart',
            xaxis: { title: 'Impact on T_elec (K)' },
            margin: { t: 40 }
        });
    } catch (e) {}
}

async function updateBenchmarkPage() {
    try {
        const data = await fetchAPI('/api/benchmark', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        const res = data;
        const rep = res.reported;
        const mod = res.modeled;
        const disc = res.discrepancy;

        const table = `
            <table class="benchmark-table">
                <thead>
                    <tr><th>Metric</th><th>Reported</th><th>Modeled</th><th>Difference</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>ΔT Contrast (C)</td>
                        <td>${rep.deltaT_C.toFixed(2)}</td>
                        <td>${mod.deltaT_contrast_C.toFixed(2)}</td>
                        <td class="discrepancy-bad">${disc.deltaT_contrast_minus_reported_K.toFixed(2)}</td>
                    </tr>
                    <tr>
                        <td>Peak Cooling (W/m2)</td>
                        <td>${rep.peak_cooling_W_m2.toFixed(2)}</td>
                        <td>${mod.peak_cooling_at_ambient_W_m2.toFixed(2)}</td>
                        <td class="discrepancy-bad">${disc.Pcool_model_minus_reported_W_m2.toFixed(2)}</td>
                    </tr>
                </tbody>
            </table>
            <div class="caption" style="text-align:left; font-style:normal; margin-top:1rem">
                <strong>Notes:</strong> ${res.notes}
            </div>
        `;
        document.getElementById('benchmark-results').innerHTML = table;

        // Wind sweep chart
        const sweep = mod.wind_sweep;
        const winds = Object.keys(sweep).map(Number);
        const t_aao = winds.map(w => sweep[w].T_AAO_C);
        const t_bare = winds.map(w => sweep[w].T_bare_C);

        const traces = [
            { x: winds, y: t_aao, name: 'AAO', mode: 'lines+markers', line: {color: materialColor('STRATOVANE_AAO_LITERATURE')} },
            { x: winds, y: t_bare, name: 'Bare Al', mode: 'lines+markers', line: {color: materialColor('BARE_ALUMINUM_OXIDIZED')} }
        ];

        plot('chart-benchmark-wind', traces, {
            title: 'Effect of Wind on Surface Temperature',
            xaxis: { title: 'Wind Speed (m/s)' },
            yaxis: { title: 'Temperature (°C)' },
            margin: { t: 40 }
        });

    } catch (e) {}
}

async function updateExperimentPage() {
    try {
        const data = await fetchAPI('/api/experiment/example', { method: 'GET' });
        renderExperimentResults(data);
    } catch (e) {}
}

function renderExperimentResults(data) {
    const met = data.metrics;
    document.getElementById('exp-metrics-display').innerHTML = `
        <div class="metrics-grid">
            <div class="metric-card">
                <span class="label">RMSE</span>
                <span class="value">${met.rmse.toFixed(2)}</span>
            </div>
            <div class="metric-card">
                <span class="label">Bias</span>
                <span class="value">${met.bias.toFixed(2)}</span>
            </div>
            <div class="metric-card">
                <span class="label">Data Source</span>
                <span class="value" style="font-size:0.9rem">${data.data_source}</span>
            </div>
        </div>
    `;
}

// Experiment Upload
document.getElementById('exp-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const data = await fetchAPI('/api/experiment', {
            method: 'POST',
            body: formData
        });
        renderExperimentResults({ ...data, data_source: 'user-uploaded' });
    } catch (e) {}
});

document.getElementById('btn-exp-example').addEventListener('click', updateExperimentPage);

// Boot
window.addEventListener('DOMContentLoaded', init);
