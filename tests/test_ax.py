from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from grounder.extract import ax

RAW = {
    "viewport": [800, 600], "doc_size": [800, 600], "title": "Docs",
    "elements": [{"id": 0, "role": "button", "text": " Save ", "context": "Docs", "tag": "AXButton",
                  "box": [0.5, 0.5, 0.1, 0.05], "abs_box": [500, 350, 80, 30], "in_viewport": True,
                  "enabled": True, "conf": 1.0, "source": "ax", "href": ""}],
}


def fake_binary(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "axdump"
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_page_from_dump_validates_elements():
    page = ax.page_from_dump(RAW, "ax-textedit", "textedit")
    assert page.viewport == [800, 600] and page.elements[0].text == "Save" and page.elements[0].source == "ax"


def test_dump_page_runs_the_binary_with_the_right_arguments(tmp_path):
    binary = fake_binary(tmp_path, f"[ \"$1 $2\" = \"--app Finder\" ] || exit 9\ncat <<'EOF'\n{json.dumps(RAW)}\nEOF\n")
    page = ax.dump_page(app="Finder", binary=binary)
    assert page.site == "finder" and len(page.elements) == 1


def test_dump_page_reports_the_binarys_message(tmp_path):
    binary = fake_binary(tmp_path, "echo 'axdump needs the Accessibility permission.' >&2\nexit 2\n")
    with pytest.raises(ax.AxdumpError, match="Accessibility permission"):
        ax.dump_page(pid=1, binary=binary)


def test_dump_page_missing_binary_and_bad_arguments(tmp_path):
    with pytest.raises(ax.AxdumpError, match="swift build"):
        ax.dump_page(app="Finder", binary=tmp_path / "nope" / "a" / "b" / "axdump")
    with pytest.raises(ValueError):
        ax.dump_page()
