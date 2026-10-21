"""Where the user is: a state and (optionally) a district, resolved against data/states/.

Only a district/state id is ever saved, never a query. The saved file is a convenience so
`synaptix ask` can show the right district offices without a flag each time.
"""

import difflib
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Location:
    state: str | None = None
    district: str | None = None
    label: str = ""

    def __bool__(self) -> bool:
        return self.state is not None


class UnknownLocation(ValueError):
    def __init__(self, name: str, suggestions: list[str] | None = None):
        super().__init__(f"Unknown location '{name}'")
        self.name, self.suggestions = name, suggestions or []


def normalize(s: str) -> str:
    s = s.strip().lower()
    for ch in " -_.":
        s = s.replace(ch, "")
    return s


def _names(d: dict) -> list[str]:
    return [d["id"], d.get("name", ""), d.get("name_kn", ""), *(d.get("aliases") or [])]


def resolve(kb, district: str | None = None, state: str | None = None) -> Location:
    st = None
    if state:
        n = normalize(state)
        for sid, s in kb.states.items():
            if n in {normalize(x) for x in _names(s["meta"]) if x}:
                st = sid
        if st is None:
            raise UnknownLocation(state, difflib.get_close_matches(
                n, [normalize(sid) for sid in kb.states], n=3, cutoff=0.5))
    if not district:
        return Location(st, None, kb.states[st]["meta"]["name"]) if st else Location()
    n = normalize(district)
    pool = [(sid, d) for sid, s in kb.states.items() if st in (None, sid) for d in s["districts"]]
    for sid, d in pool:
        if n in {normalize(x) for x in _names(d) if x}:
            return Location(sid, d["id"], d["name"])
    names = {normalize(x): x for _, d in pool for x in [d["name"], *(d.get("aliases") or [])] if x}
    close = difflib.get_close_matches(n, list(names), n=3, cutoff=0.6)
    raise UnknownLocation(district, [names[c] for c in close])


def config_path() -> Path:
    if os.environ.get("SYNAPTIX_CONFIG"):
        return Path(os.environ["SYNAPTIX_CONFIG"])
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "synaptix" / "config.yaml"


def load_config(kb) -> Location:
    try:
        with open(config_path(), encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return resolve(kb, district=data.get("district"), state=data.get("state"))
    except (OSError, yaml.YAMLError, UnknownLocation, AttributeError, TypeError):
        return Location()


def save_config(loc: Location) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"state": loc.state, "district": loc.district}, f)
    return path


def clear_config() -> None:
    config_path().unlink(missing_ok=True)
