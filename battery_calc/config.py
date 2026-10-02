"""Configuration loading. All paths, column names and tariffs come from config.yaml."""
from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass

import yaml

DEFAULT_CONFIG = "config.yaml"


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    path = path or DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    cfg["_base_dir"] = base
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve(cfg: dict, path: str | None) -> str | None:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(cfg.get("_base_dir", "."), path)


@dataclass(frozen=True)
class Connection:
    phases: int
    amps: int
    margin: float
    volts: float = 230.0

    @classmethod
    def parse(cls, text: str, margin: float = 0.2, volts: float = 230.0) -> "Connection":
        m = re.fullmatch(r"\s*([13])\s*[xX]\s*(\d+)\s*[aA]?\s*", text or "")
        if not m:
            raise ValueError(f"Connection must look like 1x25 or 3x25, got {text!r}")
        return cls(int(m.group(1)), int(m.group(2)), float(margin), float(volts))

    @property
    def label(self) -> str:
        return f"{self.phases}x{self.amps}"

    @property
    def phase_kw(self) -> float:
        return self.amps * self.volts / 1000.0

    @property
    def total_kw(self) -> float:
        return self.phase_kw * self.phases

    @property
    def battery_phase_cap_w(self) -> float:
        """Per-phase cap for battery power after the safety margin."""
        return self.amps * self.volts * (1.0 - self.margin)

    def battery_cap_w(self, battery_phases: int) -> float | None:
        """Total AC power cap for a battery, or None if it cannot be connected."""
        if battery_phases == 3 and self.phases == 1:
            return None
        return self.battery_phase_cap_w * (3 if battery_phases == 3 else 1)

    def describe(self) -> str:
        return (f"Connection {self.label} A: {self.phase_kw:.2f} kW per phase, "
                f"{self.total_kw:.2f} kW total; battery cap {self.battery_phase_cap_w/1000:.2f} kW "
                f"per phase after {self.margin:.0%} margin")
