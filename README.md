# STRATOVANE Thermal Model Web Application

This is a live-simulator web application for the STRATOVANE stratospheric thermal model.

## Architecture
- **Backend**: Flask API running the real Python thermal model.
- **Frontend**: Static HTML/CSS/JS using Plotly.js for interactive visualization.
- **Data Flow**: Frontend $\rightarrow$ REST API $\rightarrow$ Model Logic $\rightarrow$ JSON $\rightarrow$ Frontend.

## Local Development

### 1. Setup Environment
```bash
# Create venv if not exists
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Server
```bash
# Using Flask built-in server
flask --app app run --port 5000

# Or using Gunicorn for production-like performance
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 3. Preview
Open `http://localhost:5000` in your browser.

## Testing
Run the full test suite, including the web API tests:
```bash
export PYTHONPATH=$PYTHONPATH:.
./venv/bin/pytest tests/
```

## Deployment Notes
This app is suitable for free hosting on platforms like:
- **Render / Fly.io**: Use the Gunicorn start command.
- **PythonAnywhere**: Configure a Flask app pointing to `app.py`.

To keep the server awake before a demo, consider using a free uptime monitoring service (e.g., UptimeRobot) to ping the `/api/meta` endpoint every 5 minutes.

## Environment Variables
- `FLASK_ENV=production` (default)
- `PORT=5000`
