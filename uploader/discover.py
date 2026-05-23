from playwright.sync_api import sync_playwright
import json

TARGET_URL = "https://platform-challenges.philab.esa.int/geoai/submissions"  

def discover():
    with sync_playwright() as p:
        print("Launching headless Chromium...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        
        print("Loading session cookies...")
        with open("uploader/cookies.json", "r") as f:
            cookies = json.load(f)
        context.add_cookies(cookies)
        
        page = context.new_page()

        print(f"Navigating to {TARGET_URL}...")
        try:
            page.goto(TARGET_URL, wait_until="networkidle", timeout=60000)
        except Exception as e:
            print(f"Failed to load URL: {e}")
            page.screenshot(path="error_preview.png", full_page=True)
            browser.close()
            return
        
        print("\n--- INPUT ELEMENTS ---")
        inputs = page.locator("input").all()
        for idx, element in enumerate(inputs):
            name = element.get_attribute("name") or "None"
            eid = element.get_attribute("id") or "None"
            etype = element.get_attribute("type") or "None"
            placeholder = element.get_attribute("placeholder") or "None"
            print(f"[{idx}] id: {eid} | name: {name} | type: {etype} | placeholder: {placeholder}")
            
        print("\n--- LINK ELEMENTS (<a> tags) ---")
        links = page.locator("a").all()
        for idx, element in enumerate(links):
            text = element.inner_text().strip() or "None"
            href = element.get_attribute("href") or "None"
            if text != "None" and href != "None":
                print(f"[{idx}] text: {text} | href: {href}")

        print("\n--- BUTTON ELEMENTS ---")
        buttons = page.locator("button").all()
        for idx, element in enumerate(buttons):
            text = element.inner_text().strip() or "None"
            if not text:
                text = element.get_attribute("value") or "None"
            eid = element.get_attribute("id") or "None"
            print(f"[{idx}] id: {eid} | text: {text}")
            
        print("\n--- IFRAME CHECK ---")
        iframes = page.locator("iframe").all()
        if not iframes:
            print("No iframes found.")
        else:
            print(f"WARNING: Found {len(iframes)} iframes. Upload inputs might be hidden inside them!")
            for idx, element in enumerate(iframes):
                src = element.get_attribute("src") or "None"
                name = element.get_attribute("name") or "None"
                print(f"[{idx}] src: {src} | name: {name}")

        print("\nCapturing visual screenshot...")
        page.screenshot(path="page_preview.png", full_page=True)
        print("Saved 'page_preview.png'. You can download this to verify the headless render.")
        
        print("\nClosing browser.")
        browser.close()

if __name__ == "__main__":
    discover()
