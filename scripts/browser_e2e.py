"""Real-browser end-to-end checks for the local CitationGuard app.

Drives the running app (python src/app.py) with Playwright, asserts the
user-visible behaviour against official SciFact dev annotations, and saves
screenshots used in the Phase 5 report.

    python -m pip install -r requirements-dev.txt
    python -m playwright install chromium     # or pass --channel chrome / msedge
    python src/app.py --port 8765             # in another terminal
    python scripts/browser_e2e.py --url http://127.0.0.1:8765

Exit code 0 means every check passed. Nothing is sent outside 127.0.0.1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "figures"
passed: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"FAIL {message}")
    passed.append(message)
    print(f"PASS {message}")


def run_claim(page: Page, claim: str, doc_id: str = "") -> None:
    expect(page.locator("#verify")).to_be_enabled(timeout=180_000)
    page.fill("#claim", claim)
    if doc_id or page.locator("#doc-id").is_visible():
        if not page.locator("#doc-id").is_visible():
            page.click(".scope summary")
        page.fill("#doc-id", doc_id)
    page.click("#verify")


def heading(page: Page) -> str:
    return page.locator("#result-heading").inner_text()


def desktop_checks(page: Page, base: str, out_dir: Path) -> None:
    page.goto(base)
    expect(page.locator("#verify")).to_be_enabled(timeout=180_000)
    check("5,183" in page.locator("#model-status").inner_text(), "real model ready with 5,183 papers")
    page.screenshot(path=str(out_dir / "phase5_app_ready.png"))

    # Oracle: official SciFact dev claim 53 is SUPPORT with evidence in document 45638119.
    run_claim(page, "ALDH1 expression is associated with poorer prognosis in breast cancer.")
    expect(page.locator("#results")).to_be_visible(timeout=60_000)
    check(heading(page) == "Potential support", "official SUPPORT claim (dev 53) renders a tentative support verdict")
    check("45638119" in page.locator("#evidence-list").inner_text(), "gold source document 45638119 is cited")
    check(page.locator("#evidence-list .evidence-card").count() == 2
          and page.locator("#context-list .evidence-card").count() == 3,
          "two sentences used for the judgment, three kept as context")
    check(not page.locator("#additional-evidence").evaluate("e => e.open")
          and not page.locator("#scores").evaluate("e => e.open"), "context and scores start collapsed")
    check(page.evaluate("document.activeElement.id") == "results", "result panel receives keyboard focus")
    page.click("#additional-evidence summary")
    page.click("#scores summary")
    scores = page.locator("#score-values").inner_text()
    check("threshold: 0.76" in scores and "Source 5 retrieval score" in scores, "scores and threshold are disclosed")
    page.screenshot(path=str(out_dir / "phase5_app_support.png"), full_page=True)

    # JSON export contains the same verdict and provenance as the page.
    with page.expect_download() as info:
        page.click("#download")
    report = json.loads(Path(info.value.path()).read_text(encoding="utf-8"))
    check(report["label"] == "SUPPORT" and report["evidence"][0]["doc_id"] == 45638119
          and len(report["model"]["corpus_sha256"]) == 64, "downloaded JSON report matches the displayed result")

    # Oracle: official dev claim 249 is CONTRADICT (document 1568684).
    run_claim(page, "Chenodeosycholic acid treatment reduces whole-body energy expenditure.")
    expect(page.locator("#result-heading")).to_have_text("Potential contradiction", timeout=60_000)
    check(True, "official CONTRADICT claim (dev 249) renders a contradiction verdict")
    page.screenshot(path=str(out_dir / "phase5_app_contradict.png"))

    # Oracle: official dev claim 1 has no gold evidence -> must abstain.
    run_claim(page, "0-dimensional biomaterials show inductive properties.")
    expect(page.locator("#result-heading")).to_have_text("Insufficient evidence", timeout=60_000)
    check(page.locator(".evidence-card").count() > 0 and page.locator("#empty-results").is_hidden(),
          "related-but-insufficient evidence (dev 1) abstains instead of a verdict")
    page.screenshot(path=str(out_dir / "phase5_app_abstain.png"))

    run_claim(page, "Qzxvplmn qzzxrrt.")
    expect(page.locator("#empty-results")).to_be_visible(timeout=60_000)
    check(heading(page) == "Insufficient evidence" and page.locator(".evidence-card").count() == 0,
          "unknown vocabulary abstains with no invented evidence")

    # Document scope and validation.
    run_claim(page, "Aspirin inhibits platelet aggregation.", doc_id="9999999999")
    expect(page.locator("#form-error")).to_be_visible(timeout=60_000)
    check("not in this corpus" in page.locator("#form-error").inner_text() and page.locator("#results").is_hidden(),
          "unknown document ID gives an actionable error and hides stale results")
    run_claim(page, "Microstructural development of newborn cerebral white matter.", doc_id="4983")
    expect(page.locator("#results")).to_be_visible(timeout=60_000)
    metas = page.locator(".source-meta").all_text_contents()  # includes collapsed context cards
    check(bool(metas) and all("document 4983" in m for m in metas), "document-restricted search shows only that paper")
    expect(page.locator("#verify")).to_be_enabled()
    page.click("#clear")
    check(page.input_value("#claim") == "" and page.input_value("#doc-id") == "" and page.locator("#results").is_hidden()
          and page.evaluate("document.activeElement.id") == "claim", "Clear resets inputs, results and focus")
    page.click("[data-example]")
    check(page.input_value("#claim").startswith("Vitamin D") and page.locator("#results").is_hidden(),
          "example button fills a claim without inventing a result")

    # Keyboard-only flow: focus textarea, type, Tab to the button, Enter.
    page.click("#clear")
    page.keyboard.type("Aspirin inhibits platelet aggregation.")
    for _ in range(6):
        if page.evaluate("document.activeElement.id") == "verify":
            break
        page.keyboard.press("Tab")
    check(page.evaluate("document.activeElement.id") == "verify", "Check button reachable by keyboard")
    page.keyboard.press("Enter")
    expect(page.locator("#results")).to_be_visible(timeout=60_000)
    check(True, "claim can be submitted with the keyboard only")


def mobile_checks(page: Page, base: str, out_dir: Path) -> None:
    page.goto(base)
    expect(page.locator("#verify")).to_be_enabled(timeout=180_000)
    run_claim(page, "ALDH1 expression is associated with poorer prognosis in breast cancer.")
    expect(page.locator("#results")).to_be_visible(timeout=60_000)
    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    check(overflow <= 0, f"no horizontal scrolling at 375 px width (overflow={overflow}px)")
    page.screenshot(path=str(out_dir / "phase5_app_mobile.png"))


def composite(out_dir: Path) -> None:
    """Report figure: desktop contradiction result + phone-width support result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    desktop = plt.imread(out_dir / "phase5_app_contradict.png")
    phone = plt.imread(out_dir / "phase5_app_mobile.png")[:1300]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1280 / 900, 750 / 1300]})
    for ax, img, title in ((axes[0], desktop, "Desktop: official CONTRADICT claim (dev 249)"),
                           (axes[1], phone, "Phone width (375 px): dev 53")):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "phase5_app_screens.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--channel", default=None, help="Installed browser channel, e.g. chrome or msedge")
    parser.add_argument("--out", type=Path, default=SHOTS, help="Directory for screenshots")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=True)
        try:
            desktop = browser.new_context(viewport={"width": 1280, "height": 900}, accept_downloads=True)
            desktop_checks(desktop.new_page(), args.url, args.out)
            mobile = browser.new_context(viewport={"width": 375, "height": 812}, device_scale_factor=2, is_mobile=True)
            mobile_checks(mobile.new_page(), args.url, args.out)
        finally:
            browser.close()
    composite(args.out)
    print(f"\nAll {len(passed)} browser checks passed. Screenshots in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
