"""Load and enforce the macOS app collection policy in configs/apps_policy.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

POLICY_PATH = Path(__file__).resolve().parents[1] / "configs" / "apps_policy.yaml"
SET_TRAIN, SET_TEST_V2, SET_FROZEN_V1, SET_EXCLUDED = "train", "test_v2", "frozen_test_v1", "excluded"
COLLECTION_SETS = {SET_TRAIN, SET_TEST_V2}
ALL_SETS = {SET_TRAIN, SET_TEST_V2, SET_FROZEN_V1, SET_EXCLUDED}


@dataclass(frozen=True)
class AppPolicy:
    name: str
    set_name: str
    launch_name: str
    max_steps: int
    state_budget: int
    walk_budget: int
    click_roles: frozenset[str]
    typing: str
    launch_args: tuple[str, ...]
    start_keys: tuple[str, ...]
    deny_labels: tuple[str, ...]
    allow_labels: tuple[str, ...]
    deny_views: tuple[str, ...]
    throwaway_project: bool = False


@dataclass(frozen=True)
class AppsPolicy:
    path: Path
    apps: dict[str, AppPolicy]
    set_members: dict[str, frozenset[str]]
    defaults: dict[str, Any] = field(default_factory=dict)

    def canonical(self, app: str) -> str:
        """Resolve exact or case-insensitive policy app names."""
        if app in self.apps:
            return app
        folded = app.casefold()
        matches = [name for name in self.apps if name.casefold() == folded]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(f"unknown app in collection policy: {app}")

    def policy_for(self, app: str) -> AppPolicy:
        return self.apps[self.canonical(app)]

    def apps_for_split(self, split: str) -> list[str]:
        set_name = SET_TEST_V2 if split == "test" else SET_TRAIN
        return sorted(self.set_members[set_name], key=str.casefold)

    def refusal(self, app: str, split: str) -> str:
        """Why `app` may not be collected in `split`, or ''."""
        try:
            policy = self.policy_for(app)
        except KeyError:
            return "not in the app policy"
        if policy.set_name == SET_EXCLUDED:
            return "excluded by policy"
        if policy.set_name == SET_FROZEN_V1:
            return "frozen test_v1 app"
        wanted = SET_TEST_V2 if split == "test" else SET_TRAIN
        if policy.set_name != wanted:
            return f"belongs to {policy.set_name}, not {wanted}"
        return ""


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"expected a list of strings, got {value!r}")
    return value


def load_policy(path: str | Path = POLICY_PATH) -> AppsPolicy:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported app policy schema")
    sets_raw = raw.get("sets") or {}
    sets = {name: frozenset(_list(sets_raw.get(name))) for name in ALL_SETS}
    missing = ALL_SETS - set(sets_raw)
    if missing:
        raise ValueError(f"{path}: missing policy sets {sorted(missing)}")
    overlap = {name: sets[name] & union for name in ALL_SETS if (union := frozenset().union(*(sets[o] for o in ALL_SETS if o != name)))}
    overlap = {name: apps for name, apps in overlap.items() if apps}
    if overlap:
        raise ValueError(f"{path}: apps appear in multiple policy sets: {overlap}")

    defaults = dict(raw.get("defaults") or {})
    default_roles = frozenset(_list(defaults.get("click_roles")))
    if not default_roles:
        raise ValueError(f"{path}: defaults.click_roles must not be empty")
    base_deny = tuple(_list(defaults.get("deny_labels")))
    base_views = tuple(_list(defaults.get("deny_views")))
    apps: dict[str, AppPolicy] = {}
    overrides = raw.get("apps") or {}
    for set_name in (SET_TRAIN, SET_TEST_V2, SET_FROZEN_V1, SET_EXCLUDED):
        for name in sorted(sets[set_name], key=str.casefold):
            raw_app = dict(overrides.get(name) or {})
            if raw_app and raw_app.get("set", set_name) != set_name:
                raise ValueError(f"{path}: {name} override disagrees with its set")
            typing = str(raw_app.get("typing", defaults.get("typing", "search")))
            if typing not in {"search", "throwaway", "never"}:
                raise ValueError(f"{path}: {name} has unknown typing mode {typing!r}")
            walk_default = defaults.get("test_walk_budget" if set_name == SET_TEST_V2 else "walk_budget", 1)
            apps[name] = AppPolicy(
                name=name,
                set_name=set_name,
                launch_name=str(raw_app.get("launch_name", name)),
                max_steps=int(raw_app.get("max_steps", defaults.get("max_steps", 6))),
                state_budget=int(raw_app.get("state_budget", defaults.get("state_budget", 40))),
                walk_budget=int(raw_app.get("walk_budget", walk_default)),
                click_roles=frozenset(_list(raw_app.get("click_roles")) or default_roles),
                typing=typing,
                launch_args=tuple(_list(raw_app.get("launch_args")) or _list(defaults.get("launch_args"))),
                start_keys=tuple(_list(raw_app.get("start_keys")) or _list(defaults.get("start_keys"))),
                deny_labels=base_deny + tuple(_list(raw_app.get("deny_labels"))),
                allow_labels=tuple(_list(raw_app.get("allow_labels"))),
                deny_views=base_views + tuple(_list(raw_app.get("deny_views"))),
                throwaway_project=bool(raw_app.get("throwaway_project", False)),
            )
    unknown_overrides = sorted(set(overrides) - set(apps))
    if unknown_overrides:
        raise ValueError(f"{path}: overrides for apps absent from every set: {unknown_overrides}")
    return AppsPolicy(path, apps, sets, defaults)
