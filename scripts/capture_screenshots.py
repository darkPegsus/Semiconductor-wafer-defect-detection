"""Script to automate Chrome DevTools Protocol (CDP) and capture screenshots of WaferVision AI."""

import base64
import json
import subprocess
import time
from pathlib import Path

import requests
import websocket

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "docs" / "deployment" / "screenshots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not Path(CHROME_PATH).exists():
    CHROME_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


class CDPClient:
    """Minimal CDP Client over WebSocket."""

    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=10)
        self.msg_id = 1

    def send(self, method: str, params: dict | None = None) -> dict:
        req_id = self.msg_id
        self.msg_id += 1
        payload = {"id": req_id, "method": method, "params": params or {}}
        self.ws.send(json.dumps(payload))
        while True:
            resp = json.loads(self.ws.recv())
            if resp.get("id") == req_id:
                return resp.get("result", {})

    def evaluate(self, expression: str):
        return self.send("Runtime.evaluate", {"expression": expression, "returnByValue": True})

    def screenshot(self, output_path: Path, capture_beyond_viewport: bool = False):
        params = {"format": "png"}
        if capture_beyond_viewport:
            params["captureBeyondViewport"] = True
        res = self.send("Page.captureScreenshot", params)
        data = base64.b64decode(res["data"])
        with open(output_path, "wb") as f:
            f.write(data)
        print(f"Captured: {output_path} ({len(data):,} bytes)")

    def close(self):
        self.ws.close()


def main():
    print(f"Using browser at: {CHROME_PATH}")

    # 1. Start Streamlit server
    import os

    mpl_dir = PROJECT_ROOT / ".cache" / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["STREAMLIT_CONFIG_DIR"] = str(PROJECT_ROOT / ".streamlit")
    env["MPLCONFIGDIR"] = str(mpl_dir)
    env["PYTHONUNBUFFERED"] = "1"

    port = 8505

    st_cmd = [
        str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app" / "main.py"),
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    print(f"Launching Streamlit on port {port}...")
    st_proc = subprocess.Popen(
        st_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # 2. Wait for Streamlit server to be available
    server_ready = False
    print("Waiting for Streamlit server to initialize (PyTorch cold import takes ~15s)...")
    for i in range(60):
        time.sleep(1)
        if st_proc.poll() is not None:
            out, err = st_proc.communicate()
            print(
                f"Streamlit exited prematurely with code {st_proc.returncode}!\nSTDOUT:\n{out}\nSTDERR:\n{err}"
            )
            return
        try:
            r = requests.get(f"http://127.0.0.1:{port}/_stcore/health", timeout=1)
            if r.status_code == 200:
                server_ready = True
                print(f"Streamlit server is ready after {i + 1}s!")
                break
        except Exception:
            pass

    if not server_ready:
        print("Error: Streamlit failed to start.")
        st_proc.terminate()
        return

    # 3. Start Headless Chrome with remote debugging
    chrome_cmd = [
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--remote-allow-origins=*",
        "--window-size=1366,860",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-gpu",
        "--hide-scrollbars",
        "about:blank",
    ]
    print("Launching Chrome headless with CDP...")
    chrome_proc = subprocess.Popen(chrome_cmd)
    time.sleep(2)

    try:
        # Create a new tab target
        tab = requests.put(f"http://localhost:9222/json/new?http://127.0.0.1:{port}").json()
        ws_url = tab["webSocketDebuggerUrl"]
        cdp = CDPClient(ws_url)

        cdp.send("Page.enable")
        cdp.send("Runtime.enable")

        print(f"Navigating to http://127.0.0.1:{port} ...")

        # Wait for Streamlit app to load elements
        print("Waiting for initial app render...")
        time.sleep(5)

        # Ensure demo sample is loaded and preview is displayed
        for _ in range(10):
            res = cdp.evaluate("document.querySelector('button[kind=\"primary\"]') !== null")
            if res.get("result", {}).get("value") is True:
                break
            time.sleep(1)

        time.sleep(2)
        # Ensure scroll is at the top of the main container
        cdp.evaluate("""
        (() => {
            const el = document.querySelector('section[data-testid="stMain"]') || document.querySelector('.main') || window;
            el.scrollTo(0, 0);
        })()
        """)
        time.sleep(1)
        # Capture Screenshot 1: Inspection page before prediction
        shot1 = OUTPUT_DIR / "screenshot_1_inspection_before_prediction.png"
        cdp.screenshot(shot1)

        # Click the ANALYZE WAFER button
        print("Clicking 'ANALYZE WAFER' button...")
        click_res = cdp.evaluate("""
        (() => {
            const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('ANALYZE WAFER'));
            if (btn) {
                btn.scrollIntoView();
                btn.click();
                return 'clicked';
            }
            return 'button_not_found';
        })()
        """)
        print(f"Click result: {click_res.get('result', {}).get('value')}")

        # Wait for inference and Grad-CAM generation (check for Classification Results)
        print("Waiting for inference and Grad-CAM...")
        for _ in range(20):
            time.sleep(1)
            check = cdp.evaluate("document.body.innerText.includes('Classification Results')")
            if check.get("result", {}).get("value") is True:
                print("Classification results rendered!")
                break

        time.sleep(2)
        # Scroll down slightly to show prediction summary card and probability chart
        cdp.evaluate("""
        (() => {
            const el = Array.from(document.querySelectorAll('h4')).find(h => h.innerText.includes('Classification Results'));
            if (el) el.scrollIntoView({ behavior: 'instant', block: 'start' });
        })()
        """)
        time.sleep(1)

        # Capture Screenshot 2: Inspection page after prediction
        shot2 = OUTPUT_DIR / "screenshot_2_inspection_after_prediction.png"
        cdp.screenshot(shot2)

        # Scroll to Grad-CAM section
        print("Scrolling to Grad-CAM panel...")
        cdp.evaluate("""
        (() => {
            const el = Array.from(document.querySelectorAll('h4')).find(h => h.innerText.includes('Model Explainability'));
            if (el) el.scrollIntoView({ behavior: 'instant', block: 'start' });
        })()
        """)
        time.sleep(2)

        # Capture Screenshot 3: Grad-CAM visualization
        shot3 = OUTPUT_DIR / "screenshot_3_gradcam_visualization.png"
        cdp.screenshot(shot3)

        # Scroll back to top and navigate to "Model Performance" in sidebar
        print("Navigating to 'Model Performance' page...")
        cdp.evaluate("window.scrollTo(0, 0);")
        time.sleep(0.5)

        # Find and click the second radio button ("Model Performance")
        click_radio_js = """
        (() => {
            const labels = Array.from(document.querySelectorAll('section[data-testid="stSidebar"] div[role="radiogroup"] label'));
            const perfLabel = labels.find(l => l.innerText.includes('Model Performance'));
            if (perfLabel) {
                perfLabel.click();
                return 'clicked_performance';
            }
            return 'perf_not_found';
        })()
        """
        radio_res = cdp.evaluate(click_radio_js)
        print(f"Radio click result: {radio_res.get('result', {}).get('value')}")
        print("Waiting for Model Performance page to render...")

        for _ in range(15):
            time.sleep(1)
            check_perf = cdp.evaluate(
                "document.body.innerText.includes('Complete Per-Class Held-Out Test Breakdown')"
            )
            if check_perf.get("result", {}).get("value") is True:
                print("Performance metrics rendered!")
                break

        time.sleep(2)
        # Scroll to top of Model Performance page to show KPIs and table
        cdp.evaluate("""
        (() => {
            const el = document.querySelector('section[data-testid="stMain"]') || document.querySelector('.main') || window;
            el.scrollTo(0, 0);
        })()
        """)
        time.sleep(1)
        shot4a = OUTPUT_DIR / "screenshot_4_model_performance.png"
        cdp.screenshot(shot4a)

        # Scroll to confusion matrix
        cdp.evaluate("""
        (() => {
            const el = Array.from(document.querySelectorAll('h4')).find(h => h.innerText.includes('Confusion Matrix'));
            if (el) el.scrollIntoView({ behavior: 'instant', block: 'start' });
        })()
        """)
        time.sleep(1)
        shot4b = OUTPUT_DIR / "screenshot_4b_confusion_matrix.png"
        cdp.screenshot(shot4b)

        # Navigate to "About" in sidebar
        print("Navigating to 'About' page...")
        click_about_js = """
        (() => {
            const labels = Array.from(document.querySelectorAll('section[data-testid="stSidebar"] div[role="radiogroup"] label'));
            const aboutLabel = labels.find(l => l.innerText.includes('About'));
            if (aboutLabel) {
                aboutLabel.click();
                return 'clicked_about';
            }
            return 'about_not_found';
        })()
        """
        cdp.evaluate(click_about_js)
        time.sleep(3)
        cdp.evaluate("""
        (() => {
            const el = document.querySelector('section[data-testid="stMain"]') || document.querySelector('.main') || window;
            el.scrollTo(0, 0);
        })()
        """)
        time.sleep(1)
        shot5 = OUTPUT_DIR / "screenshot_5_about_page.png"
        cdp.screenshot(shot5)

        cdp.close()
        print("All screenshots successfully captured!")

    finally:
        print("Cleaning up processes...")
        chrome_proc.terminate()
        st_proc.terminate()


if __name__ == "__main__":
    main()
