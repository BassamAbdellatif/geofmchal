import os
import sys
import json
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from playwright.sync_api import sync_playwright

# --- CONFIGURABLE VARIABLES ---
URL = "https://platform-challenges.philab.esa.int/geoai/submissions"
USE_COOKIES = True
COOKIES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookies.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Upload a competition submission zip directly to the ESA platform.")
    parser.add_argument("--experiment-name", type=str, required=True,
                        help="Name of the experiment (must match the run folder and zip created by submit.py).")
    parser.add_argument("--tta", action="store_true",
                        help="Upload the TTA zip (submission_<name>_tta.zip) instead of the base zip.")
    parser.add_argument("--zip-file", type=str, default=None,
                        help="Optional explicit path to the zip file to upload. "
                             "Overrides the default naming derived from --experiment-name and --tta.")
    return parser.parse_args()


def upload_submission():
    args = parse_args()
    exp_dir = os.path.join(config.SHARED_RUNS_DIR, args.experiment_name)

    if args.zip_file:
        FILE_PATH = args.zip_file
    else:
        suffix = "_tta" if args.tta else ""
        FILE_PATH = os.path.join(exp_dir, f"submission_{args.experiment_name}{suffix}.zip")

    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: Submission file not found at {FILE_PATH}")
        print(f"   → Run submit.py --experiment-name {args.experiment_name} first.")
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
                # Playwright requires sameSite to be exactly "Strict", "Lax", or "None".
                # EditThisCookie exports lowercase ("lax") and "unspecified" — normalise in-memory.
                _samesite_map = {"strict": "Strict", "lax": "Lax", "none": "None", "unspecified": "Lax"}
                for cookie in cookies:
                    raw = cookie.get("sameSite", "Lax")
                    cookie["sameSite"] = _samesite_map.get(raw.lower(), "Lax")
                context.add_cookies(cookies)
            else:
                print(f"⚠️ Warning: USE_COOKIES is True but {COOKIES_FILE} is missing. Continuing without cookies.")

        page = context.new_page()
        print(f"Navigating to {URL}...")

        try:
            page.goto(URL, wait_until="networkidle", timeout=60000)
            print("Page loaded. Giving React 5 seconds to render...")
            page.wait_for_timeout(5000)
        except Exception as e:
            print(f"❌ Failed to load URL: {e}")
            page.screenshot(path="error_timeout.png", full_page=True)
            browser.close()
            return

        # --- Session guard: detect if cookies have expired ---
        if page.locator("text='SIGN IN'").count() > 0 and page.locator("text='SIGN OUT'").count() == 0:
            print("❌ Session expired! The page shows SIGN IN. Please export fresh cookies.json and retry.")
            page.screenshot(path="session_expired.png", full_page=True)
            browser.close()
            return
        print("✅ Session is authenticated (SIGN OUT found in header).")

        # --- Step 1: Click the NEW button ---
        # Use a precise button selector to avoid accidentally clicking page text like 'New'
        print("🔍 Searching for the 'NEW' submission button...")
        try:
            # Target strictly a <button> or <a> element whose exact visible text is 'NEW'
            new_btn = page.locator("button:has-text('NEW'), a:has-text('NEW')").first
            new_btn.wait_for(state="visible", timeout=30000)
            new_btn.click(force=True, timeout=15000)
            print("✅ Clicked 'NEW' button!")
        except Exception as e:
            print(f"❌ Could not find or click the 'NEW' button: {e}")
            try:
                page.screenshot(path="dashboard_error.png", full_page=True)
            except Exception:
                pass
            browser.close()
            return

        # Give the modal animation time to complete
        page.wait_for_timeout(3000)
        page.screenshot(path="after_new_click.png", full_page=True)
        print("📸 Saved 'after_new_click.png' for debugging.")

        # --- Step 2: Fill Name and Description using JavaScript ---
        # Playwright found the elements but reported them as "not visible".
        # We bypass this entirely with JS after confirming the input exists in DOM.
        print("📝 Waiting for modal inputs to appear in DOM...")
        
        # Wait until the Name input is actually in the DOM (not just the button click)
        try:
            page.wait_for_selector('input[placeholder="Name"]', state="attached", timeout=15000)
            print("✅ Modal inputs found in DOM.")
        except Exception:
            print("⚠️  Name input not found via selector, attempting JS anyway...")

        exp_name = os.path.basename(FILE_PATH).replace(".zip", "")
        # Description: strip the 'submission_' prefix to get just the experiment ID
        short_name = exp_name.replace("submission_", "", 1)
        desc_text = short_name
        print(f"📝 Filling Name: '{exp_name}' | Description: '{desc_text}'")

        fill_result = page.evaluate("""(args) => {
            const [name, desc] = args;
            const results = [];

            // Helper: set value on an input and fire React-compatible events
            function fillInput(el, value, label) {
                if (!el) { results.push(label + ': NOT FOUND'); return; }
                const proto = el.tagName === 'TEXTAREA'
                    ? window.HTMLTextAreaElement.prototype
                    : window.HTMLInputElement.prototype;
                const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
                setter.call(el, value);
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                results.push(label + ': OK');
            }

            // Target by placeholder (confirmed from Playwright's own log)
            fillInput(
                document.querySelector('input[placeholder="Name"]'),
                name, 'Name'
            );
            fillInput(
                document.querySelector('input[placeholder="Description"]')
                || document.querySelector('textarea[placeholder="Description"]')
                || document.querySelector('textarea'),
                desc, 'Description'
            );

            return results;
        }""", [exp_name, desc_text])

        for r in fill_result:
            print(f"  → {r}")

        page.wait_for_timeout(1000)

        # --- Step 3: Attach the file ---
        print("📎 Attaching file stream...")
        try:
            page.set_input_files("input[type='file']", FILE_PATH)
            print("✅ File attached successfully.")
        except Exception as e:
            print(f"❌ Failed to attach file: {e}")
            page.screenshot(path="file_attach_error.png", full_page=True)
            browser.close()
            return

        # Give React time to validate the form and reveal the Submit button
        page.wait_for_timeout(2000)
        page.screenshot(path="before_submit.png", full_page=True)
        print("📸 Saved 'before_submit.png' for debugging.")

        # --- Step 4: Click Submit and intercept the network response ---
        print("🚀 Triggering final submission...")
        try:
            submit_btn = page.locator("button[type='submit'], button:has-text('Submit')").first
            
            print("\n" + "=" * 50)
            print("⏳ UPLOAD IN PROGRESS... DO NOT CLOSE SCRIPT ⏳")
            print("=" * 50)
            print("Watching network for server confirmation of upload...")

            # Use Playwright's expect_response to intercept the API call.
            # We give it a VERY generous timeout (90 minutes) to stream the 890MB file.
            # The lambda matches any POST-like response from the platform's API.
            with page.expect_response(
                lambda r: "philab.esa.int" in r.url and r.request.method in ("POST", "PUT"),
                timeout=90 * 60 * 1000  # 90 minutes
            ) as response_info:
                submit_btn.click(force=True, timeout=15000)
                print("✅ Submit clicked! Streaming file to ESA servers...")

            # We land here only after the server responded!
            response = response_info.value
            print(f"\n📡 Server responded: HTTP {response.status} from {response.url}")

            if response.status in (200, 201, 204):
                print("✅ SUCCESS! Server confirmed the submission was received!")
                page.screenshot(path="upload_success.png", full_page=True)
                print("📸 Saved 'upload_success.png'")
            else:
                print(f"⚠️  Unexpected status {response.status}. Check 'upload_response_error.png'")
                page.screenshot(path="upload_response_error.png", full_page=True)

            # Now click "Close" to cleanly dismiss the dialog
            print("Closing the submission dialog...")
            try:
                close_btn = page.locator("button:has-text('Close')").first
                close_btn.click(force=True, timeout=10000)
                print("✅ Dialog closed.")
            except Exception:
                print("⚠️  Could not find Close button, dialog may have closed itself.")

        except Exception as e:
            print(f"\n❌ Upload process failed: {e}")
            page.screenshot(path="upload_failure_debug.png", full_page=True)
            print("Saved debug screenshot to 'upload_failure_debug.png'")
        finally:
            print("Closing browser context.")
            browser.close()


if __name__ == "__main__":
    upload_submission()

