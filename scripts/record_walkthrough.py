"""Record a silent walkthrough of the running app (presentation mode) with
Playwright's built-in video recording -> docs/walkthrough.webm (plays in Chrome).

    streamlit run app.py &   # on :8501
    python3 scripts/record_walkthrough.py
"""
import os
import shutil
import sys
import tempfile

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ui_screenshots import wait_idle, click_tab, URL  # noqa: E402


def main():
    tmp = tempfile.mkdtemp()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1920, "height": 1000}, record_video_dir=tmp,
                                  record_video_size={"width": 1920, "height": 1000})
        page = ctx.new_page()
        page.goto(URL + "?present=1", wait_until="domcontentloaded")
        wait_idle(page, 6)
        click_tab(page, "1 · Route & holds"); page.wait_for_timeout(2500)
        click_tab(page, "2 · Climber"); page.wait_for_timeout(3000)
        click_tab(page, "3 · Optimize"); page.wait_for_timeout(4000)
        page.get_by_role("button", name="Rate H25 terrible", exact=True).click(); wait_idle(page, 3); page.wait_for_timeout(2500)
        page.get_by_role("button", name="Reset ratings", exact=True).click(); wait_idle(page, 2); page.wait_for_timeout(1000)
        page.get_by_role("button", name="Simulate 85 % climber", exact=True).click(); wait_idle(page, 3)
        page.mouse.wheel(0, 700); page.wait_for_timeout(3500)
        page.mouse.wheel(0, -700); page.wait_for_timeout(800)
        page.get_by_role("button", name="Back to measured", exact=True).click(); wait_idle(page, 2)
        page.get_by_role("button", name="Show full-body plan", exact=True).click(); wait_idle(page, 4)
        page.mouse.wheel(0, 900); page.wait_for_timeout(4000)
        page.mouse.wheel(0, 900); page.wait_for_timeout(3000)
        click_tab(page, "4 · Personalize"); page.wait_for_timeout(3500)
        video = page.video
        ctx.close()
        path = video.path()
        browser.close()
    out = os.path.join(ROOT, "docs", "walkthrough.webm")
    shutil.move(path, out)
    print("saved", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
