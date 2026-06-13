"""Top-level pipeline runner. Backed by LangGraph since v0.0.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from .engines import build_engine
from .graph import build_graph, run_source
from .ingest import FileSource
from .judges import build_judge
from .output import build_sink
from .types import AudioSource


@dataclass
class Pipeline:
    graph: Any  # compiled langgraph

    def run(self, source: AudioSource, language_hint: str | None = None) -> None:
        run_source(self.graph, source, language_hint=language_hint)


def load_pipeline(config_path: str) -> Pipeline:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    engines = [build_engine(s) for s in cfg["stt"]]
    judge = build_judge(cfg.get("judge", {"strategy": "confidence"}))
    sink = build_sink(cfg.get("output", {"sink": "console"}))
    graph = build_graph(engines=engines, judge=judge, sink=sink)
    return Pipeline(graph=graph)


def source_from_arg(audio_path: str) -> AudioSource:
    return FileSource(audio_path)
