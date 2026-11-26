from __future__ import annotations

import json
import threading
from pathlib import Path
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from grounder.data.pages import write_page
from grounder.extract import ax
from grounder.records import Example
from tests.test_ax_collect import raw_dump
from tools.mac_goals import (
    examples_for_label, parse_json_object, validate_label, vague_offending_tokens, walk_prompt,
)
from grounder.data.pages import load_pages_from


def make_goal_walk(tmp_path):
    raw1 = raw_dump("Calculator")
    raw2 = raw_dump("Calculator", words="one two three four five six seven eight")
    pages_dir = tmp_path / "pages"
    page1 = ax.page_from_dump(raw1, "calculator-a", "calculator")
    page2 = ax.page_from_dump(raw2, "calculator-b", "calculator")
    write_page(page1, pages_dir)
    write_page(page2, pages_dir)
    element = page1.elements[2]
    walk = {
        "walk_id": "calculator-train-000", "app": "Calculator", "split": "train",
        "steps": [{
            "index": 0, "pre_page_id": page1.page_id, "post_page_id": page2.page_id,
            "action": {"type": "click", "target_id": element.id, "role": element.role, "label": element.text,
                       "tag": element.tag, "typed_text": ""},
            "target": {"id": element.id, "equiv_ids": [element.id], "visible_text": element.text, "nonvisual_text": ""},
        }, {
            "index": 1, "pre_page_id": page2.page_id, "post_page_id": page2.page_id,
            "action": {"type": "click", "target_id": page2.elements[1].id, "role": "button",
                       "label": page2.elements[1].text, "tag": "AXButton", "typed_text": ""},
            "target": {"id": page2.elements[1].id, "equiv_ids": [page2.elements[1].id],
                       "visible_text": page2.elements[1].text, "nonvisual_text": ""},
        }],
    }
    return walk, {page1.page_id: page1, page2.page_id: page2}


def test_goal_json_parsing_and_validation():
    raw = parse_json_object('noise {"coherent":true,"goal_literal":"Open settings","goal_vague":"adjust preferences","stop_step":1} tail')
    label = validate_label(raw, 2)
    assert label.coherent and label.stop_step == 1 and label.goal_vague == "adjust preferences"
    with pytest.raises(ValueError, match="stop_step"):
        validate_label({**raw, "stop_step": 2}, 2)
    with pytest.raises(ValueError, match="empty"):
        parse_json_object("")


def test_goal_vague_target_token_overlap_is_mechanically_detected(tmp_path):
    walk, _pages = make_goal_walk(tmp_path)
    label_word = walk["steps"][0]["action"]["label"].split()[0]
    assert label_word in vague_offending_tokens(f"press {label_word} now", walk, 0)
    assert vague_offending_tokens("adjust the device controls", walk, 0) == []


def test_examples_use_custom_nextstep_schema_walk_source_and_item_tags(tmp_path):
    walk, pages = make_goal_walk(tmp_path)
    label = validate_label({"coherent": True, "goal_literal": "Use the controls",
                            "goal_vague": "operate the device", "stop_step": 0}, 2)
    examples = examples_for_label(walk, label, pages)
    assert len(examples) == 1
    example = examples[0]
    assert isinstance(example, Example)
    assert example.kind == example.mode == "goal" and example.query == "operate the device"
    assert example.source == "calculator-train-000" and example.history == []
    assert example.target_id == walk["steps"][0]["target"]["id"]
    assert example.equiv_ids == walk["steps"][0]["target"]["equiv_ids"]
    assert set(example.tags) <= {"literal", "goal-only", "icon"}


def test_walk_prompt_contains_app_actions_and_state_summaries(tmp_path):
    walk, pages = make_goal_walk(tmp_path)
    prompt = walk_prompt(walk, pages)
    assert "App: Calculator" in prompt and "Step 0" in prompt and "target role='button'" in prompt
    assert "before:" in prompt and "after:" in prompt



def test_synthesize_and_write_builds_examples_pages_report_and_manifest(tmp_path, monkeypatch):
    import tools.mac_goals as mg

    walk_root = tmp_path / "walks-root"
    walk, pages = make_goal_walk(walk_root)
    # make_goal_walk wrote the two state pages directly in the root; move them into the collector layout.
    pages_dir = walk_root / "train" / "pages" / "calculator"
    pages_dir.mkdir(parents=True)
    for page in pages.values():
        write_page(page, walk_root / "train" / "pages")
    walk_dir = walk_root / "train" / "walks"
    walk_dir.mkdir(parents=True)
    (walk_dir / "calculator.jsonl").write_text(json.dumps(walk) + "\n")
    payload = {"coherent": True, "goal_literal": "Use the controls", "goal_vague": "operate the device", "stop_step": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
        def do_POST(self):
            length = int(self.headers.get("content-length", "0")); self.rfile.read(length)
            body = json.dumps({"choices": [{"message": {"content": json.dumps(payload)}}]}).encode()
            self.send_response(200); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, fmt, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake_gguf = tmp_path / "fake.gguf"
    fake_gguf.write_bytes(b"gemma-test")
    monkeypatch.setattr(mg, "find_model", lambda pattern: fake_gguf)
    out = tmp_path / "nextstep"
    args = Namespace(walk_root=walk_root, policy=Path("configs/apps_policy.yaml"), split="train", out=out,
                     name="mac_train_v2", endpoint=f"http://127.0.0.1:{server.server_port}",
                     server_bin="/unused", model_glob="unused", no_start_server=True, concurrency=1, max_tokens=400,
                     request_timeout=5.0, min_steps=1, max_walks=0)
    try:
        report = mg.synthesize_and_write(args)
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
    assert report["counts"]["examples"] == 1 and report["counts"]["coherent_walks"] == 1
    rows = [json.loads(line) for line in (out / "mac_train_v2.jsonl").read_text().splitlines()]
    assert rows[0]["query"] == "operate the device" and rows[0]["kind"] == "goal"
    assert rows[0]["source"] == walk["walk_id"] and rows[0]["tags"]
    manifest = json.loads((out / "mac_train_v2.manifest.json").read_text())
    assert manifest["status"] == "frozen" and manifest["outputs"]
    assert all(len(item["sha256"]) == 64 for item in manifest["outputs"])
    assert report["model"]["gguf_sha256"] == mg.sha256_file(fake_gguf)
    assert (out / "mac_train_v2-build.json").exists()
