import pytest
from app import app, ALLOWED_RANGES
import json
import os
import glob
import numpy as np
import time

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_api_meta(client):
    """Verify /api/meta returns expected schema and defaults."""
    resp = client.get('/api/meta')
    assert resp.status_code == 200
    data = resp.get_json()
    assert "version" in data
    assert "limitations" in data
    assert "materials" in data
    assert "ranges" in data
    assert "defaults" in data

def test_api_altitude_correctness(client):
    """Verify results match main.py mode 10 (20km, 5m/s, 20W)."""
    payload = {
        "materials": ["BARE_ALUMINUM_OXIDIZED", "WHITE_REFERENCE_PLACEHOLDER", "STRATOVANE_AAO_LITERATURE"],
        "altitude": 20000,
        "wind": 5.0,
        "heat_load": 20.0,
        "solar_scale": 1.0
    }
    resp = client.post('/api/altitude', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    
    res = data['results']
    # Bare: 22.98, White: 1.09, AAO: -6.61
    assert abs(res['BARE_ALUMINUM_OXIDIZED']['t_electronics'] - 22.98) < 0.1
    assert abs(res['WHITE_REFERENCE_PLACEHOLDER']['t_electronics'] - 1.09) < 0.1
    assert abs(res['STRATOVANE_AAO_LITERATURE']['t_electronics'] - (-6.61)) < 0.1
    
    # Energy balance check: Q_rad + Q_conv + Q_solar = Q_internal
    for m in res:
        q = res[m]['q_env']
        total = q['rad'] + q['conv'] + q['solar']
        assert abs(total - 20.0) < 1e-3

def test_api_altitude_validation(client):
    """Verify invalid inputs return 400."""
    # Out of range altitude
    resp = client.post('/api/altitude', json={"materials": [], "altitude": 30000, "wind": 5, "heat_load": 20})
    assert resp.status_code == 400
    
    # Non-numeric
    resp = client.post('/api/altitude', json={"materials": [], "altitude": "high", "wind": 5, "heat_load": 20})
    assert resp.status_code == 400
    
    # Missing field
    resp = client.post('/api/altitude', json={"materials": [], "wind": 5})
    assert resp.status_code == 400

def test_no_hardcoded_data():
    """Scan webapp/static/ for known model output numbers (app-owned files only)."""
    forbidden_numbers = ["22.98", "-6.61", "1.09", "102.1", "14.1"]

    # Exclude third-party bundles: static/vendor/* (e.g. plotly.min.js) is not
    # app-owned and can legitimately contain arbitrary numerals.
    app_files = [p for p in glob.glob("webapp/static/**/*", recursive=True)
                 if os.path.isfile(p) and "/vendor/" not in os.path.relpath(p).replace(os.sep, "/")]

    for file_path in app_files:
        if file_path.endswith('.html') or file_path.endswith('.js') or file_path.endswith('.css'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for num in forbidden_numbers:
                    assert num not in content, f"Found hardcoded value {num} in {file_path}"

def test_mc_job_lifecycle(client):
    """Test Monte Carlo job creation and polling."""
    payload = {
        "material": "STRATOVANE_AAO_LITERATURE",
        "altitude": 20000,
        "n": 100,
        "seed": 42
    }
    resp = client.post('/api/montecarlo', json=payload)
    assert resp.status_code == 200
    job_id = resp.get_json()['job_id']
    
    # Poll until done
    for _ in range(20):
        status_resp = client.get(f'/api/jobs/{job_id}')
        status_data = status_resp.get_json()
        if status_data['status'] == 'completed':
            assert "result" in status_data
            return
        time.sleep(0.5)
    pytest.fail("MC job did not complete in time")

def test_mc_n_cap(client):
    """Ensure n cap is enforced."""
    payload = {
        "material": "stratovane_aao_literature",
        "altitude": 20000,
        "n": 20000, # Above limit
        "seed": 42
    }
    resp = client.post('/api/montecarlo', json=payload)
    assert resp.status_code == 400
