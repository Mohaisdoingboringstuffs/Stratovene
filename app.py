from flask import Flask, request, jsonify, send_from_directory
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import uuid
import threading
from copy import deepcopy

# Import model components
from config import SimulationConfig
from materials import load_materials
from enclosure import solve_steady
from transient import simulate_24h
from spectral import thermal_wavelengths_m, solar_wavelengths_m
from uncertainty import monte_carlo, sensitivity
from validation import diaz_lobo_reproduction
from experiments import load_experiment_csv, metrics as exp_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    static_folder=str(ROOT_DIR / 'webapp' / 'static'),
    template_folder=str(ROOT_DIR / 'webapp' / 'templates'),
)

# Model Versioning
def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def get_git_commit_date():
    """Commit date of the current HEAD, so the version date is reproducible."""
    try:
        return subprocess.check_output(
            ['git', 'show', '-s', '--format=%ad', '--date=short', 'HEAD']
        ).decode('ascii').strip()
    except Exception:
        return "unknown"

MODEL_VERSION = "1.0.0"
GIT_COMMIT = get_git_commit()
MODEL_DATE = get_git_commit_date()

# Known limitations as per requirements
KNOWN_LIMITATIONS = [
    "No validation against measurements",
    "Benchmark does not reproduce the paper (peak cooling ~102 vs 175 W/m2, deltaT definition mismatch, empty-box control not modeled)",
    "Broadband sky model used in the benchmark",
    "Convection at altitude uses an assumed wind and may overstate heat transfer in thin air for a free-floating platform",
    "Electronics heat load is a scenario value, not flight data"
]

# Range validation helpers
ALLOWED_RANGES = {
    "altitude": (0, 25000),
    "wind": (0, 100),
    "heat_load": (0, 500),
    "solar_scale": (0, 2.0),
    "mc_n": (1, 10000)
}

def validate_range(name, value):
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    low, high = ALLOWED_RANGES.get(name, (-float('inf'), float('inf')))
    if not (low <= value <= high):
        raise ValueError(f"{name} out of range ({low} to {high})")

# Background Job Store for Monte Carlo.
# In-process only: cleared on restart. Capped by pruning finished jobs when
# MAX_JOBS is exceeded so long-running servers cannot leak memory unboundedly.
jobs = {}
jobs_lock = threading.Lock()
MAX_JOBS = 40

def _prune_jobs_locked():
    """Drop oldest completed/failed entries while over MAX_JOBS (insertion-ordered dict)."""
    while len(jobs) > MAX_JOBS:
        done_ids = [k for k, v in jobs.items() if v.get("status") in ("completed", "failed")]
        if not done_ids:
            break
        jobs.pop(done_ids[0])

def run_mc_job(job_id, cfg, material, altitude, n, seed):
    try:
        # Use a copy of config for thread safety
        local_cfg = deepcopy(cfg)
        local_cfg.uncertainty.seed = seed
        res = monte_carlo(local_cfg, material, altitude, n)
        result = {
            "mean": res.mean,
            "median": res.median,
            "p5": res.p5,
            "p95": res.p95,
            "std": res.std,
            "histogram": res.t_electronics_c.tolist()
        }
        with jobs_lock:
            jobs[job_id] = {"status": "completed", "result": result}
    except Exception as e:
        logger.error(f"MC Job {job_id} failed: {e}")
        with jobs_lock:
            jobs[job_id] = {"status": "failed", "error": str(e)}

@app.route('/')
def index():
    return send_from_directory(str(ROOT_DIR / 'webapp' / 'static'), 'index.html')

@app.route('/api/meta', methods=['GET'])
def api_meta():
    mats = load_materials()
    return jsonify({
        "version": MODEL_VERSION,
        "commit": GIT_COMMIT,
        "date": MODEL_DATE,
        "limitations": KNOWN_LIMITATIONS,
        "ranges": ALLOWED_RANGES,
        "materials": list(mats.keys()),
        "defaults": {
            "altitude": 20000,
            "wind": 5.0,
            "heat_load": 20.0,
            "solar_scale": 1.0
        }
    })

@app.route('/api/altitude', methods=['POST'])
def api_altitude():
    data = request.json
    try:
        mats_requested = data.get('materials', [])
        alt = data.get('altitude')
        wind = data.get('wind')
        heat = data.get('heat_load')
        solar_scale = data.get('solar_scale', 1.0)

        validate_range('altitude', alt)
        validate_range('wind', wind)
        validate_range('heat_load', heat)
        validate_range('solar_scale', solar_scale)

        cfg = SimulationConfig()
        cfg.environment.altitude_m = alt
        cfg.environment.wind_speed_m_s = wind
        cfg.electronics.heat_load_w = heat
        # solar_scale applies ONLY when an explicit irradiance override is configured.
        # With the default None the internal altitude-dependent solar model is used and
        # the scale is a no-op; report that explicitly instead of silently ignoring it.
        solar_scale_applied = False
        if cfg.environment.irradiance_direct_w_m2 is not None:
            cfg.environment.irradiance_direct_w_m2 *= solar_scale
            solar_scale_applied = True

        all_mats = load_materials()
        results = {}
        for m_id in mats_requested:
            if m_id not in all_mats:
                return jsonify({"error": f"Unknown material {m_id}"}), 400
            mat = all_mats[m_id]
            sol = solve_steady(cfg, mat, altitude_m=alt, wind=wind, heat_w=heat)
            results[m_id] = {
                "t_electronics": sol.t_electronics_c,
                "t_surface": sol.t_surface_c,
                "q_env": {
                    "rad": sol.q_env_rad_net_w,
                    "conv": sol.q_env_conv_w,
                    "solar": sol.q_env_solar_w
                },
                "residual": sol.relative_energy_error,
                "metadata": {
                    "altitude": "MODELED",
                    "wind": "ASSUMED",
                    "heat_load": "ASSUMED"
                }
            }

        return jsonify({
            "results": results,
            "metadata": {
                "version": MODEL_VERSION,
                "commit": GIT_COMMIT,
                "scenario": {"wind": wind, "heat": heat, "solar_scale": solar_scale},
                "solar_scale_applied": solar_scale_applied
            }
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Altitude API error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/transient', methods=['POST'])
def api_transient():
    data = request.json
    try:
        mats_requested = data.get('materials', [])
        alt = data.get('altitude')
        wind = data.get('wind')
        heat = data.get('heat_load')
        
        validate_range('altitude', alt)
        validate_range('wind', wind)
        validate_range('heat_load', heat)

        cfg = SimulationConfig()
        cfg.environment.altitude_m = alt
        cfg.environment.wind_speed_m_s = wind
        cfg.electronics.heat_load_w = heat
        
        all_mats = load_materials()
        results = {}
        for m_id in mats_requested:
            if m_id not in all_mats:
                return jsonify({"error": f"Unknown material {m_id}"}), 400
            mat = all_mats[m_id]
            tr = simulate_24h(cfg, mat, altitude_m=alt)
            results[m_id] = {
                "time": tr.time_s.tolist(),
                "t_electronics": (tr.t_electronics_k - 273.15).tolist(),
                "t_wall": (tr.t_wall_k - 273.15).tolist(),
                "metrics": {
                    "min": tr.t_min_c,
                    "max": tr.t_max_c,
                    "swing": tr.swing_k
                }
            }
        return jsonify({"results": results, "metadata": {"version": MODEL_VERSION, "commit": GIT_COMMIT}})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Transient API error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/spectral', methods=['POST'])
def api_spectral():
    data = request.json
    try:
        mats_requested = data.get('materials', [])
        all_mats = load_materials()
        results = {}
        
        temp_k = 293.15
        wav_solar = solar_wavelengths_m()
        wav_thermal = thermal_wavelengths_m()
        
        for m_id in mats_requested:
            if m_id not in all_mats:
                return jsonify({"error": f"Unknown material {m_id}"}), 400
            mat = all_mats[m_id]
            # Serve the model's own Kirchhoff-consistent spectral accessors
            # (interpolates the material's spectral data; falls back to
            # band-representatives when no CSV spectrum exists).
            results[m_id] = {
                "solar": {
                    "wavelength": wav_solar.tolist(),
                    "reflectance": mat.reflectance_on(wav_solar).tolist()
                },
                "thermal": {
                    "wavelength": wav_thermal.tolist(),
                    "emissivity": mat.epsilon_on(wav_thermal).tolist()
                }
            }
        return jsonify({
            "results": results,
            "window": [8.0, 13.0],
            "metadata": {"version": MODEL_VERSION}
        })
    except Exception as e:
        logger.error(f"Spectral API error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/montecarlo', methods=['POST'])
def api_montecarlo():
    data = request.json
    try:
        mat_id = data.get('material')
        alt = data.get('altitude')
        n = data.get('n', 2000)
        seed = data.get('seed', 42)
        
        validate_range('altitude', alt)
        validate_range('mc_n', n)
        
        all_mats = load_materials()
        if mat_id not in all_mats:
            return jsonify({"error": "Unknown material"}), 400
            
        job_id = str(uuid.uuid4())
        cfg = SimulationConfig()
        cfg.environment.altitude_m = alt

        with jobs_lock:
            _prune_jobs_locked()
            jobs[job_id] = {"status": "pending"}
        # daemon=True so a long MC cannot block process shutdown
        thread = threading.Thread(target=run_mc_job, args=(job_id, cfg, all_mats[mat_id], alt, n, seed), daemon=True)
        thread.start()
        
        return jsonify({"job_id": job_id, "status": "pending"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"MC Start error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/jobs/<job_id>', methods=['GET'])
def api_job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

@app.route('/api/sensitivity', methods=['POST'])
def api_sensitivity():
    data = request.json
    try:
        mat_id = data.get('material')
        alt = data.get('altitude')
        
        validate_range('altitude', alt)
        
        all_mats = load_materials()
        if mat_id not in all_mats:
            return jsonify({"error": "Unknown material"}), 400
            
        cfg = SimulationConfig()
        cfg.environment.altitude_m = alt
        res = sensitivity(cfg, all_mats[mat_id], alt)
        
        return jsonify({
            "tornado": res.tornado,
            "next_experiment": res.next_experiment,
            "metadata": {"version": MODEL_VERSION}
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Sensitivity API error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/benchmark', methods=['POST'])
def api_benchmark():
    data = request.json
    try:
        cfg = SimulationConfig()
        res = diaz_lobo_reproduction(cfg)
        return jsonify({
            "reported": res.reported,
            "modeled": res.modeled,
            "discrepancy": res.discrepancy,
            "notes": res.notes,
            "metadata": {"version": MODEL_VERSION}
        })
    except Exception as e:
        logger.error(f"Benchmark API error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/experiment/example', methods=['GET'])
def api_experiment_example():
    df = load_experiment_csv()
    cfg = SimulationConfig()
    met = exp_metrics(df, cfg=cfg)
    return jsonify({
        "metrics": met.__dict__,
        "data_source": "SYNTHETIC EXAMPLE",
        "metadata": {"version": MODEL_VERSION}
    })

@app.route('/api/experiment', methods=['POST'])
def api_experiment_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Save to a system temp dir (NOT the publicly served static folder) and
        # always delete, including on parse failure.
        fd, tmp_path = tempfile.mkstemp(suffix=".csv", prefix="stratovane_upload_")
        os.close(fd)
        try:
            file.save(tmp_path)
            df = load_experiment_csv(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        cfg = SimulationConfig()
        met = exp_metrics(df, cfg=cfg)
        return jsonify({
            "metrics": met.__dict__,
            "data_source": "user-uploaded",
            "metadata": {"version": MODEL_VERSION}
        })
    except Exception as e:
        logger.error(f"Experiment upload error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False)
