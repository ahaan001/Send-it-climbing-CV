"""Drive the running Streamlit app with Playwright: screenshot the three steps and
exercise the hold editor and the Explore presets. Produces docs/screenshots/.

    streamlit run app.py &      # on :8501
    python3 scripts/ui_screenshots.py
    SENDIT_TEST_UPLOAD=1 python3 scripts/ui_screenshots.py   # also runs the photo + video upload path (~1 min)
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "screenshots")
URL = os.environ.get("SENDIT_URL", "http://localhost:8501")
TAB_ROUTE, TAB_CLIMB, TAB_EXPLORE = "1 Route", "2 Climb", "3 Explore"


def wait_idle(page, t=2.5):
    page.wait_for_timeout(int(t * 1000))
    try:
        page.wait_for_selector("[data-testid='stStatusWidget']", state="detached", timeout=90000)
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


def exceptions(page):
    return page.locator("[data-testid='stException']").count()


def main():
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.goto(URL, wait_until="domcontentloaded")
        wait_idle(page, 6)
        print("exceptions on load:", exceptions(page))
        shot(page, "01_route.png")
        click_tab(page, TAB_CLIMB)
        shot(page, "02_climb.png")
        click_tab(page, TAB_EXPLORE)
        shot(page, "03_explore.png")

        # --- hold editor: pick a hold by clicking the canvas
        click_tab(page, TAB_ROUTE)
        canvas = page.locator("iframe[title*='streamlit_image_coordinates']").first
        box = canvas.bounding_box()
        print("canvas box:", box)
        if box:
            page.mouse.click(box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.60)
            wait_idle(page, 2)
            shot(page, "04_hold_selected.png")

        # --- explore presets
        click_tab(page, TAB_EXPLORE)
        page.get_by_role("button", name="What if I were 15 % shorter?", exact=True).click()
        wait_idle(page, 3)
        shot(page, "05_explore_shorter.png")
        page.get_by_role("button", name="Show feet too", exact=True).click()
        wait_idle(page, 6)
        shot(page, "06_explore_feet.png")
        print("exceptions after presets:", exceptions(page))

        # --- photo + video upload path through the UI (optional)
        if os.environ.get("SENDIT_TEST_UPLOAD"):
            page.get_by_text("Use my own route", exact=True).click()
            wait_idle(page, 1.5)
            click_tab(page, TAB_ROUTE)
            page.locator("input[type='file']").nth(0).set_input_files(os.path.join(ROOT, "demo_assets", "kilter", "route_photo.png"))
            wait_idle(page, 3)
            shot(page, "07_upload_route.png")
            click_tab(page, TAB_CLIMB)
            page.locator("input[type='file']").nth(1).set_input_files(os.path.join(ROOT, "demo_assets", "kilter", "kilter_climb.mp4"))
            wait_idle(page, 2)
            page.get_by_role("button", name="Analyze my climb", exact=True).click()
            t0 = time.time()
            for _ in range(90):
                page.wait_for_timeout(2000)
                if exceptions(page) or page.get_by_text("matched", exact=False).count() or page.get_by_text("Mark the board corners", exact=False).count():
                    break
            wait_idle(page, 3)
            print(f"upload analysis finished in {time.time() - t0:.0f}s; exceptions:", exceptions(page))
            shot(page, "08_upload_climb.png")
        browser.close()


if __name__ == "__main__":
    sys.exit(main())
