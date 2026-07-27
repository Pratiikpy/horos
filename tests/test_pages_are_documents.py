"""The served pages must be complete HTML documents, not fragments.

`/scorecard` and `/proof` began at `<title>` — no doctype, no `<html>`, no `<head>`, and no viewport
meta. Browsers forgive the missing tags. They do not forgive the missing viewport: a phone lays the
page out at roughly 980px and zooms out to fit, so the coverage table — the part carrying the actual
argument — arrives unreadable on the device a reviewer is most likely holding. No doctype also drops
the browser into quirks mode, where the box model differs from the one this CSS was written against.

Found by auditing the served markup rather than by calling the API, which is the only way this class
of defect surfaces: every endpoint check passed, and the page looked correct on a desktop.
"""
from __future__ import annotations

import pytest

import proof
import scorecard_page


def _pages():
    return {"/scorecard": scorecard_page.page({}, [], True, []) if False else None}


@pytest.fixture(scope="module")
def rendered():
    from fastapi.testclient import TestClient

    import server
    c = TestClient(server.app)
    return {p: c.get(p) for p in ("/scorecard", "/proof")}


@pytest.mark.parametrize("path", ["/scorecard", "/proof"])
def test_the_page_is_served_as_html(rendered, path):
    r = rendered[path]
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


@pytest.mark.parametrize("path", ["/scorecard", "/proof"])
def test_the_page_is_a_whole_document(rendered, path):
    html = rendered[path].text.lstrip()
    assert html.lower().startswith("<!doctype html>"), "no doctype means quirks mode"
    for tag in ("<html", "<head", "<body", "</html>"):
        assert tag in html.lower(), f"{path} is missing {tag}"


@pytest.mark.parametrize("path", ["/scorecard", "/proof"])
def test_the_page_scales_on_a_phone(rendered, path):
    """The one that is not cosmetic: without this a phone renders at ~980px and zooms out."""
    html = rendered[path].text.lower()
    assert 'name="viewport"' in html, f"{path} has no viewport meta — unreadable on a phone"
    assert "width=device-width" in html


@pytest.mark.parametrize("path", ["/scorecard", "/proof"])
def test_the_page_declares_a_language_and_charset(rendered, path):
    html = rendered[path].text.lower()
    assert 'lang="en"' in html
    assert 'charset="utf-8"' in html or "charset=utf-8" in html


@pytest.mark.parametrize("path", ["/scorecard", "/proof"])
def test_wide_content_scrolls_inside_its_own_container(rendered, path):
    """A table wider than a phone must scroll itself rather than scrolling the page body."""
    html = rendered[path].text
    if "<table" in html:
        assert "overflow-x:auto" in html.replace(" ", ""), \
            f"{path} has a table with no horizontal scroll container"
