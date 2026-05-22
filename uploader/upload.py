import os
import json
from playwright.sync_api import sync_playwright

# --- CONFIGURABLE VARIABLES ---
URL = "https://platform-challenges.philab.esa.int/geoai/submissions"  # Replace with the upload page URL
FILE_PATH = "/mnt/head/users/bassam/src/geofmchal/runs/2A_alpha_ts1_ts2/submission_2A_alpha_ts1_ts2.zip"
USE_COOKIES = True
COOKIES_FILE = "cookies.json"

# Targets (Discovered from discover.py)
UPLOAD_INPUT_SELECTOR = "input[type='file']" 
SUBMIT_BUTTON_SELECTOR = "button#submit"  
TIMEOUT_MS = 10 * 60 * 1000  # 10 minutes timeout for massive file uploads


def upload_submission():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: Submission file not found at {FILE_PATH}")
        return

    with sync_playwright() as p:
        print("Launching headless Chromium...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()

        # Cookie Injection for SSO Bypass
        if USE_COOKIES:
            if os.path.exists(COOKIES_FILE):
                print(f"Loading session cookies from {COOKIES_FILE}...")
                with open(COOKIES_FILE, "r") as f:
                    cookies = json.load(f)
                context.add_cookies(cookies)
            else:
                print(f"⚠️ Warning: USE_COOKIES is True but {COOKIES_FILE} is missing. Continuing without cookies.")
        else:
            print("Cookie injection disabled. Manual login required.")
            # --- Placeholder for Manual Login ---
            # page = context.new_page()
            # page.goto("https://example.com/login", wait_until="networkidle")
            # page.fill("input#username", "my_user")
            # page.fill("input#password", "my_pass")
            # page.click("button#login")
            # page.wait_for_load_state("networkidle")

        page = context.new_page()
        print(f"Navigating to {URL}...")
        
        try:
            page.goto(URL, wait_until="networkidle")
        except Exception as e:
            print(f"❌ Failed to load URL: {e}")
            browser.close()
            return

        print("Attaching file stream...")
        try:
            # Playwright bypasses UI clicks/React hidden overlays by setting files directly on the input
            page.set_input_files(UPLOAD_INPUT_SELECTOR, FILE_PATH)
            print("File attached successfully.")
        except Exception as e:
            print(f"❌ Failed to attach file to selector '{UPLOAD_INPUT_SELECTOR}': {e}")
            browser.close()
            return

        print(f"Triggering final submission with {TIMEOUT_MS/1000/60:.1f} minute timeout...")
        try:
            # Wrap submission click with generous timeout to accommodate backbone bandwidth streaming
            page.click(SUBMIT_BUTTON_SELECTOR, timeout=TIMEOUT_MS)
            
            # Wait for the network to idle indicating upload transmission has completed
            print("Waiting for network idle to confirm transmission completion...")
            page.wait_for_load_state("networkidle", timeout=TIMEOUT_MS)
            
            # Optional: Add an explicit assertion for a success element here if known
            # page.wait_for_selector("text='Upload Successful'", timeout=TIMEOUT_MS)
            
            print("✅ Upload transmitted successfully!")
        except Exception as e:
            print(f"❌ Upload process timed out or failed: {e}")
            
            # Save a debug screenshot if it fails
            page.screenshot(path="upload_failure_debug.png", full_page=True)
            print("Saved debug screenshot to 'upload_failure_debug.png'")
        finally:
            print("Closing browser context.")
            browser.close()

if __name__ == "__main__":
    upload_submission()
