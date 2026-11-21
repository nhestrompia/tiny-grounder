"""Planner loop: one Claude model drives a browser through tools, in three element-access configurations.

    full   `snapshot()` returns every element on the page
    top20  `snapshot()` returns the grounder's top 20 elements in goal mode (query = goal, history = last actions)
    find   `find(query)` returns the grounder's top 5 elements with scores in do mode; there is no snapshot

All configurations share `click(id)`, `type(id, text, press_enter)`, `go_back()` and `done(answer)`. Element ids
refer to the most recent snapshot / find. Both the planner (`ClaudePlanner`) and the ranker are injectable.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol, Sequence

from demo.browser import BrowserSession, Task, ToolError
from grounder.baselines.bm25 import rank_elements
from grounder.records import Element, Mode, action_repr

CONFIGS: tuple[str, ...] = ("full", "top20", "find")
TOP_N = 20
FIND_K = 5
HISTORY_LEN = 3
MAX_TOKENS = 16000
ACTIONS = ("click", "type", "go_back")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Planner calls allowed per step of the task budget; snapshot / find calls do not count as steps.
CALLS_PER_STEP = 3

SYSTEM_BASE = """You operate a web browser to accomplish a goal on a public website. The browser is already on the start page.

Each element is shown as `id|role|text|context`. Element ids refer to the most recent element list you asked for; after \
the page changes, ask again before using ids. Use click(id) to follow links and press buttons, and type(id, text, \
press_enter) to fill a text field (set press_enter to search). Some clicks are refused for safety (buttons that submit \
forms, or that mention paying, buying, deleting, sending...); pick another way. You can only read and navigate; never \
try to log in or to change anything.

Work efficiently. When the goal is reached, call done(answer) with a short answer; if it cannot be reached, call \
done with the reason."""

SYSTEM_BY_CONFIG: dict[str, str] = {
    "full": SYSTEM_BASE + "\n\nCall snapshot() to list every element on the current page.",
    "top20": SYSTEM_BASE + "\n\nCall snapshot() to list the 20 elements most relevant to your goal on the current page, "
    "most relevant first. If the element you need is not listed, navigate elsewhere and snapshot again.",
    "find": SYSTEM_BASE + "\n\nYou cannot see the whole page. Call find(query) with a short description of the element you "
    "want (for example 'search box' or 'link to the download page'); it returns the 5 best matches with a score between "
    "0 and 1, best first.",
}

_ID = {"type": "integer", "description": "Element id from the most recent element list."}
TOOLS: dict[str, dict[str, Any]] = {
    "snapshot": {
        "name": "snapshot",
        "description": "List the elements of the current page. Replaces the element ids you had before.",
        "input_schema": {"type": "object", "properties": {}},
    },
    "find": {
        "name": "find",
        "description": "Find the elements of the current page that best match a description. Replaces the element ids "
        "you had before.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What the element is or does."}},
            "required": ["query"],
        },
    },
    "click": {
        "name": "click",
        "description": "Click an element. Returns the page address and title afterwards.",
        "input_schema": {"type": "object", "properties": {"id": _ID}, "required": ["id"]},
    },
    "type": {
        "name": "type",
        "description": "Click a text field and type into it, optionally pressing Enter (for example to run a search).",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": _ID,
                "text": {"type": "string"},
                "press_enter": {"type": "boolean", "description": "Press Enter after typing. Default false."},
            },
            "required": ["id", "text"],
        },
    },
    "go_back": {
        "name": "go_back",
        "description": "Go back to the previous page.",
        "input_schema": {"type": "object", "properties": {}},
    },
    "done": {
        "name": "done",
        "description": "Finish the task.",
        "input_schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
    },
}
TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "full": ("snapshot", "click", "type", "go_back", "done"),
    "top20": ("snapshot", "click", "type", "go_back", "done"),
    "find": ("find", "click", "type", "go_back", "done"),
}


class Ranker(Protocol):
    """What the demo needs from a grounder; `grounder.runtime.Grounder` satisfies it."""

    def rank(
        self, query: str, elements: Sequence[Element], k: int = 5, mode: Mode = "do", history: list[str] | None = None
    ) -> dict[str, Any]:
        """Return {"results": [{"id", "score", ...}, ...], "null_prob": float}, best first."""


class BM25Ranker:
    """Fallback ranker for running the demo before a model exists; ignores the mode and the history."""

    def rank(
        self, query: str, elements: Sequence[Element], k: int = 5, mode: Mode = "do", history: list[str] | None = None
    ) -> dict[str, Any]:
        ranked = rank_elements(query, list(elements))
        results = [{"id": i, "score": s} for i, s in zip(ranked.ids[:k], ranked.scores[:k])]
        return {"results": results, "null_prob": ranked.null_prob}


class Planner(Protocol):
    def create(self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> Any:
        """One model call; returns an object with `content`, `stop_reason` and `usage` like an API message."""


class ClaudePlanner:
    """Claude Messages API (beta endpoint, for server-side refusal fallback). Pass `client` only in tests."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", client: Any = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client, self.model, self.effort = client, model, effort

    def create(self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> Any:
        # No thinking param (adaptive), no temperature, tool_choice left at auto. The breakpoint on the system block
        # caches tools + system; top-level `cache_control` (automatic caching) also caches the growing conversation,
        # as any real agent would, so configurations are compared at realistic cost. `fallbacks="default"` re-runs a request the safety classifiers decline on
        # the recommended fallback model server-side; a refusal that survives comes back as stop_reason "refusal".
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=messages,
            cache_control={"type": "ephemeral"},
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )


def element_line(e: Element) -> str:
    return f"{e.id}|{e.role}|{e.text}|{e.context}"


class ToolBox:
    """Executes the planner's tool calls against a browser session; keeps the action history for goal mode."""

    def __init__(self, config: str, session: BrowserSession, ranker: Ranker, goal: str) -> None:
        if config not in CONFIGS:
            raise ValueError(f"unknown config {config!r}; choose from {', '.join(CONFIGS)}")
        self.config, self.session, self.ranker, self.goal = config, session, ranker, goal
        self.history: list[str] = []
        self.extract_ms = self.rank_ms = 0.0
        self.tools = [TOOLS[name] for name in TOOL_NAMES[config]]
        self.system = SYSTEM_BY_CONFIG[config]

    def call(self, name: str, args: dict[str, Any]) -> str:
        """Run one tool; raises ToolError for a failed or refused call."""
        if name not in TOOL_NAMES[self.config]:
            raise ToolError(f"unknown tool {name!r}")
        try:
            return getattr(self, f"_{name}")(**args)
        except TypeError as exc:
            raise ToolError(f"invalid arguments: {exc}") from exc

    def _header(self, count: int) -> str:
        return f"{self.session.status()}\n{count} elements shown (id|role|text|context):"

    def _ranked(self, query: str, k: int, mode: Mode, history: list[str] | None) -> list[dict[str, Any]]:
        started = time.perf_counter()
        elements = self.session.extract().elements
        self.extract_ms += (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        try:
            return self.ranker.rank(query, elements, k=k, mode=mode, history=history)["results"]
        finally:
            self.rank_ms += (time.perf_counter() - started) * 1000

    def _snapshot(self) -> str:
        if self.config == "full":
            started = time.perf_counter()
            shown = self.session.extract().elements
            self.extract_ms += (time.perf_counter() - started) * 1000
        else:
            ranked = self._ranked(self.goal, TOP_N, "goal", self.history[-HISTORY_LEN:])
            by_id = {e.id: e for e in self.session.elements}
            shown = [by_id[r["id"]] for r in ranked]
        return "\n".join([self._header(len(shown)), *map(element_line, shown)])

    def _find(self, query: str) -> str:
        ranked = self._ranked(query, FIND_K, "do", None)
        by_id = {e.id: e for e in self.session.elements}
        lines = [f"{element_line(by_id[r['id']])}|score={r['score']:.3f}" for r in ranked]
        return "\n".join([self._header(len(lines)), *lines])

    # History entries use Mind2Web's action format, which goal mode is trained on; go_back has no Mind2Web
    # equivalent and is not recorded.
    def _click(self, id: int) -> str:
        element = self.session.click(id)
        self.history.append(action_repr(element))
        return f"clicked {element.role} {element.text!r}\n{self.session.status()}"

    def _type(self, id: int, text: str, press_enter: bool = False) -> str:
        element = self.session.type_text(id, text, press_enter)
        self.history.append(action_repr(element, text))
        enter = " and pressed Enter" if press_enter else ""
        return f"typed into {element.role} {element.text!r}{enter}\n{self.session.status()}"

    def _go_back(self) -> str:
        self.session.go_back()
        return f"went back\n{self.session.status()}"

    def _done(self, answer: str) -> str:
        return "ok"


@dataclass
class RunResult:
    """One task run. Token fields sum the API usage of every call; `tokens_in` includes cached tokens."""

    task_id: str
    config: str
    repeat: int
    success: bool = False
    done: bool = False
    stop: str = ""
    answer: str = ""
    error: str = ""
    final_url: str = ""
    steps: int = 0
    calls: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    @property
    def tokens_in(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "tokens_in": self.tokens_in}

    def add_usage(self, usage: Any) -> None:
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0


def as_dict(obj: Any) -> dict[str, Any]:
    """JSON-able view of an API object (pydantic model) or of a plain test double."""
    return obj.model_dump(exclude_none=True) if hasattr(obj, "model_dump") else dict(vars(obj))


def run_task(
    task: Task, config: str, planner: Planner, ranker: Ranker, log: Callable[[dict[str, Any]], None], repeat: int = 0
) -> RunResult:
    """Run `task` once in `config`. `log` receives one JSON-able event per model response and per tool call.

    The run ends when the planner calls `done`, stops without a tool call, is refused or truncated, or spends the
    step budget (click / type / go_back calls) or the call budget.
    """
    result = RunResult(task.id, config, repeat)
    with BrowserSession(task.start_url) as session:
        box = ToolBox(config, session, ranker, task.goal)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"Goal: {task.goal}\nThe browser is on {session.url}"}
        ]
        start = time.perf_counter()
        while not result.done and not result.stop:
            if result.steps >= task.max_steps or result.calls >= CALLS_PER_STEP * task.max_steps:
                result.stop = "max_steps"
                break
            response = planner.create(box.system, box.tools, messages)
            result.calls += 1
            result.add_usage(response.usage)
            log({"type": "response", "call": result.calls, "stop_reason": response.stop_reason,
                 "usage": as_dict(response.usage), "content": [as_dict(b) for b in response.content]})
            # Never run tools of a refused or truncated turn: their input may be cut off.
            if response.stop_reason in ("refusal", "max_tokens"):
                result.stop = response.stop_reason
                break
            messages.append({"role": "assistant", "content": response.content})
            calls = [b for b in response.content if b.type == "tool_use"]
            if not calls:
                result.stop = "end_turn"
                result.answer = " ".join(b.text for b in response.content if b.type == "text")
                break
            outputs = []
            for call in calls:
                try:
                    output, is_error = box.call(call.name, dict(call.input)), False
                except ToolError as exc:
                    output, is_error = str(exc), True
                log({"type": "tool", "call": result.calls, "name": call.name, "input": dict(call.input),
                     "result": output, "is_error": is_error})
                outputs.append({"type": "tool_result", "tool_use_id": call.id, "content": output,
                                **({"is_error": True} if is_error else {})})
                result.steps += call.name in ACTIONS
                if call.name == "done" and not is_error:
                    result.done, result.answer = True, str(call.input.get("answer", ""))
            messages.append({"role": "user", "content": outputs})
        result.seconds = time.perf_counter() - start
        result.final_url = session.url
        result.success = session.succeeded(task.check)
    return result
