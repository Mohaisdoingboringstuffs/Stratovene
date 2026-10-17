"""Static-asset serving contract for the Flask web app.

Guards against regression of the case where /style.css, /app.js and
/vendor/plotly.min.js returned 404 because index.html referenced the
assets at root-relative URLs while Flask served static files at /static.
"""
import re
from pathlib import Path

import pytest

from app import app

STATIC_DIR = Path(__file__).resolve().parent.parent / 'webapp' / 'static'
INDEX_PATH = STATIC_DIR / 'index.html'


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index_page_served(client):
    resp = client.get('/')
    assert resp.status_code == 200
    assert resp.mimetype == 'text/html'
    assert len(resp.data) > 1000


def test_style_css_served(client):
    resp = client.get('/style.css')
    assert resp.status_code == 200
    assert resp.mimetype == 'text/css'
    assert len(resp.data) > 100


def test_app_js_served(client):
    resp = client.get('/app.js')
    assert resp.status_code == 200
    assert resp.mimetype == 'text/javascript'
    assert len(resp.data) > 5000


def test_vendor_plotly_served_full_file(client):
    vendored = (STATIC_DIR / 'vendor' / 'plotly.min.js').read_bytes()
    assert len(vendored) >= 3_000_000, (
        f"vendored plotly is only {len(vendored)} bytes; expected the full "
        "~3.5 MB bundle, not a truncated copy"
    )
    assert b'plotly' in vendored[:120].lower()
    assert b'newPlot' in vendored

    resp = client.get('/vendor/plotly.min.js')
    assert resp.status_code == 200
    assert resp.mimetype == 'text/javascript'
    assert resp.data == vendored


def test_favicon_served(client):
    resp = client.get('/favicon.ico')
    assert resp.status_code == 200
    assert resp.mimetype in ('image/x-icon', 'image/vnd.microsoft.icon', 'image/x-ico')
    assert 100 <= len(resp.data) <= 5000


def test_serving_is_cwd_independent(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for url in ('/', '/app.js', '/style.css', '/vendor/plotly.min.js', '/favicon.ico'):
        resp = client.get(url)
        assert resp.status_code == 200, f"GET {url} failed when cwd is {tmp_path}"


def test_every_index_html_reference_resolves(client):
    html = INDEX_PATH.read_text(encoding='utf-8')
    refs = re.findall(r'(?:src|href)\s*=\s*"([^"]+)"', html)
    assert refs, "index.html must reference assets"

    local_urls = []
    for url in refs:
        if url.startswith(('http://', 'https://', 'data:', 'mailto:')) or url.startswith('#'):
            continue
        local_urls.append(url if url.startswith('/') else '/' + url)

    assert '/style.css' in local_urls
    assert '/app.js' in local_urls
    assert '/vendor/plotly.min.js' in local_urls
    assert '/favicon.ico' in local_urls

    for url in local_urls:
        resp = client.get(url)
        assert resp.status_code == 200, (
            f"index.html references {url!r} but GET returned {resp.status_code}"
        )


def test_no_inline_controller_copy():
    """The controller must live in app.js, not be duplicated inline in the HTML."""
    html = INDEX_PATH.read_text(encoding='utf-8')
    assert re.search(r'<link[^>]*href="style\.css"', html)
    assert re.search(r'<script[^>]*\ssrc="vendor/plotly\.min\.js"', html)
    assert re.search(r'<script[^>]*\ssrc="app\.js"', html)
    assert 'STRATOVANE Frontend Controller' not in html
    assert '<script>' not in html.replace('<script src=', 'SKIP')
