from __future__ import annotations

import io
import struct
import zipfile

from tools import agentnet, groundcua


def _central_directory(zip_bytes: bytes) -> bytes:
    end = zip_bytes.rfind(b"PK\x05\x06")
    size, offset = struct.unpack("<II", zip_bytes[end + 12:end + 20])
    return zip_bytes[offset:offset + size]


def test_central_directory_gives_offsets_that_hold_the_files():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("images/a.png", b"first" * 100)
        z.writestr("images/b.png", b"second" * 50)
    raw = buf.getvalue()
    entries = agentnet.parse_central_directory(_central_directory(raw))
    assert set(entries) == {"a.png", "b.png"}
    disk, off, csz, meth = entries["b.png"]
    assert disk == 0 and meth == 8 and raw[off:off + 4] == b"PK\x03\x04"


def test_click_usable_only_when_next_step_confirms_it():
    traj = [
        {"image": "1.png", "value": {"code": "pyautogui.click(x=0.5, y=0.25)", "action": "Click File. Then"}},
        {"image": "2.png", "value": {"code": "pyautogui.write('x')", "last_step_correct": True}},
        {"image": "3.png", "value": {"code": "pyautogui.doubleClick(x=0.1, y=0.2)"}},
        {"image": "4.png", "value": {"code": "pyautogui.click(x=0.3, y=0.3)", "last_step_correct": False}},
    ]
    steps = agentnet.parse_steps(traj, completed=True)
    assert [(s.kind, s.usable) for s in steps] == [("click", True), (None, False), ("doubleClick", False),
                                                   ("click", True)]
    assert steps[0].x == 0.5 and steps[0].y == 0.25


def test_test_apps_are_excluded_by_app_or_instruction():
    assert agentnet.uses_test_app({"applications": ["VLC media player"]})
    assert agentnet.uses_test_app({"applications": [], "instruction": "Open Logic Pro and add a track"})
    assert not agentnet.uses_test_app({"applications": ["Safari"], "instruction": "search flights"})


def test_click_hits_smallest_text_line():
    page = agentnet.build_page("agentnet-x", "x", 1000, 500, [
        {"text": "Toolbar File Edit", "box": [0, 0, 400, 30]},
        {"text": "File", "box": [100, 5, 40, 20]},
    ])
    assert [page.elements[i].text for i in agentnet.hit_ids(page, 110, 10)] == ["File", "Toolbar File Edit"]
    assert agentnet.hit_ids(page, 900, 400) == []


def test_groundcua_page_drops_cursor_and_maps_roles():
    page = groundcua.build_page("Some App", "ab" * 32, 200, 100, [
        {"bbox": [10, 10, 30, 20], "text": "Save", "category": "Button"},
        {"bbox": [50, 50, 60, 60], "text": "cursor", "category": "Others"},
        {"bbox": [0, 0, 200, 100], "text": "", "category": ""},
    ])
    assert [(e.role, e.text) for e in page.elements] == [("button", "Save"), ("other", "")]
    assert page.page_id == f"groundcua-some-app-{'ab' * 6}"
    assert page.elements[0].box == [0.05, 0.1, 0.1, 0.1]
    assert groundcua.element_at(page, 15, 15) == 0  # smallest containing element wins over the full-window one


def test_groundcua_excludes_held_out_vlc():
    assert "VLC Media Player" in groundcua.EXCLUDED_APPS
