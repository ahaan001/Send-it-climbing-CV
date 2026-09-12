"""Drive the running Streamlit app with Playwright: screenshot every tab and
exercise the hold editor (click-select a hold, rate it). Produces backup
screenshots in docs/screenshots/.

    streamlit run app.py &      # on :8501
    python3 scripts/ui_screenshots.py
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "screenshots")
URL = os.environ.get("SENDIT_URL", "http://localhost:8501")


def wait_idle(page, t=2.5):
    page.wait_for_timeout(int(t * 1000))
    try:
        page.wait_for_selector("[data-testid='stStatusWidget']", state="detached", timeout=60000)
    except Exception:
        pass
    page.wait_for_timeout(600)


def shot(page, name, full=True):
    path = os.path.join(OUT, name)
    page.screenshot(path=path, full_page=full)
    print("saved", path)


def click_tab(page, label):
    page.get_by_role("tab", name=label).click()
    wait_idle(page, 1.5)


def main():
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1)
        page.goto(URL, wait_until="domcontentloaded")
        wait_idle(page, 6)
        errors = page.locator("[data-testid='stException']").count()
        print("exceptions on load:", errors)
        shot(page, "01_route_holds.png")
        click_tab(page, "2 · Climber")
        shot(page, "02_climber.png")
        click_tab(page, "3 · Optimize")
        shot(page, "03_optimize.png")
        click_tab(page, "4 · Personalize")
        shot(page, "04_personalize.png")
        click_tab(page, "Method")
        shot(page, "05_method.png")

        # --- hold editor interaction: select a hold by clicking the canvas
        click_tab(page, "1 · Route & holds")
        canvas = page.locator("iframe[title*='streamlit_image_coordinates']").first
        box = canvas.bounding_box()
        print("canvas box:", box)
        if box:
            # click near the centre of the displayed image; the app selects the nearest hold if close enough
            page.mouse.click(box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.60)
            wait_idle(page, 2)
            sel_text = page.get_by_text("Hold ", exact=False).first
            print("selected panel visible:", sel_text.is_visible() if sel_text else None)
            shot(page, "06_hold_selected.png")
        # --- switch demo to Kilter
        page.get_by_role("combobox").first.click()
        wait_idle(page, 0.5)
        page.get_by_text("Kilter-style LED board", exact=False).first.click()
        wait_idle(page, 0.5)
        page.get_by_role("button", name="Load", exact=True).click()
        wait_idle(page, 4)
        click_tab(page, "3 · Optimize")
        shot(page, "07_kilter_optimize.png")
        click_tab(page, "4 · Personalize")
        shot(page, "08_kilter_personalize.png")
        print("total exceptions now:", page.locator("[data-testid='stException']").count())

        # --- presentation mode + demo presets (spray wall)
        page.goto(URL + "?present=1", wait_until="domcontentloaded")
        wait_idle(page, 6)
        click_tab(page, "3 · Optimize")
        shot(page, "10_present_optimize.png", full=False)
        page.get_by_role("button", name="Rate H25 terrible", exact=True).click()
        wait_idle(page, 3)
        shot(page, "11_present_rate_h25.png", full=False)
        page.get_by_role("button", name="Reset ratings", exact=True).click()
        wait_idle(page, 2)
        page.get_by_role("button", name="Simulate 85 % climber", exact=True).click()
        wait_idle(page, 3)
        shot(page, "12_present_simulated.png")
        page.get_by_role("button", name="Back to measured", exact=True).click()
        wait_idle(page, 2)
        page.get_by_role("button", name="Show full-body plan", exact=True).click()
        wait_idle(page, 4)
        shot(page, "13_present_fullbody.png")
        print("presentation-mode exceptions:", page.locator("[data-testid='stException']").count())

        # --- live upload path through the UI (optional, ~30 s)
        if os.environ.get("SENDIT_TEST_UPLOAD"):
            page.get_by_text("Upload video", exact=True).click()
            wait_idle(page, 1)
            page.locator("input[type='file']").set_input_files(os.path.join(ROOT, "demo_assets", "skeleton_overlay.mp4"))
            wait_idle(page, 2)
            page.get_by_role("button", name="Analyze", exact=True).click()
            t0 = time.time()
            wait_idle(page, 5)
            for _ in range(60):
                if page.locator("[data-testid='stException']").count() or page.get_by_text("live analysis", exact=False).count():
                    break
                page.wait_for_timeout(2000)
            print(f"upload analysis finished in {time.time() - t0:.0f}s; exceptions:", page.locator("[data-testid='stException']").count())
            click_tab(page, "3 · Optimize")
            shot(page, "09_upload_live_optimize.png")
        browser.close()


if __name__ == "__main__":
    sys.exit(main())
