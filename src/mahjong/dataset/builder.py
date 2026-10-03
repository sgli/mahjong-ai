"""Streaming Decision Dataset builder.

Pipeline (DATA_SPEC section 9)::

    raw .mjai.json -> parse -> replay -> decision extraction -> chunk -> parquet

Only one game file's events/states are materialised at a time; samples are
buffered up to ``chunk_size`` rows per split and then flushed to a numbered
Parquet file, so the whole dataset never has to fit in memory.

Splitting is per complete game (assigned before processing from the sorted file
list, deterministically from ``seed``), so no game's samples span two splits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from ..decision.action import ACTION_SCHEMA_VERSION
from ..decision.decision import DecisionSample
from ..decision.extractor import DecisionExtractor
from ..parser import MjaiParseError, MjaiParser
from ..replay import ReplayEngine, ReplayInconsistency
from .manifest import build_manifest, write_manifest
from .schema import DATASET_VERSION, FEATURE_VERSION, parquet_schema, sample_to_row
from .split import SPLIT_NAMES, assign_splits

log = logging.getLogger("dataset.builder")


class DatasetBuilder:
    """Build a versioned, chunked, game-split Parquet decision dataset."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        chunk_size: int = 100_000,
        ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 0,
        dataset_version: str = DATASET_VERSION,
        action_schema_version: str = ACTION_SCHEMA_VERSION,
        feature_version: str = FEATURE_VERSION,
        source_dir: str | Path | None = None,
        limit: int = 0,
        git_commit_hash: str | None = None,
        progress_interval: int = 0,
        processed_games: set[str] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.ratios = ratios
        self.seed = seed
        self.dataset_version = dataset_version
        self.action_schema_version = action_schema_version
        self.feature_version = feature_version
        self.source_dir = source_dir
        self.limit = limit
        self.git_commit_hash = git_commit_hash
        self.progress_interval = progress_interval
        self.processed_games = set(processed_games or ())
        self.new_processed_games: set[str] = set()

        self._split_assignment: dict[str, str] = {}
        self._buffers = {name: [] for name in SPLIT_NAMES}
        self._part_counters = {name: 0 for name in SPLIT_NAMES}
        self.errors: list[str] = []
        self._counts = {
            "files_requested": 0,
            "files_processed": 0,
            "files_error": 0,
            "games": 0,
            "rounds": 0,
            "samples": 0,
            "samples_skipped": 0,
            "splits": {name: {"games": 0, "samples": 0, "files": 0} for name in SPLIT_NAMES},
        }
        self._rounds_seen: set[str] = set()

    # -- split assignment -------------------------------------------------------
    def assign_splits(self, game_ids: Sequence[str]) -> None:
        self._split_assignment = assign_splits(game_ids, self.ratios, self.seed)

    # -- writing ----------------------------------------------------------------
    def _split_dir(self, split: str) -> Path:
        return self.output_dir / split

    def _flush(self, split: str) -> None:
        rows = self._buffers[split]
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=parquet_schema())
        out_dir = self._split_dir(split)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._part_counters[split] += 1
        path = out_dir / f"{split}-{self._part_counters[split]:05d}.parquet"
        pq.write_table(table, path)
        self._counts["splits"][split]["files"] += 1
        self._buffers[split].clear()

    def add_game_samples(self, game_id: str, samples: Iterable[DecisionSample]) -> None:
        """Append all samples of one game to the game's assigned split."""
        split = self._split_assignment.get(game_id)
        if split is None:
            raise KeyError(f"game {game_id!r} was not assigned a split (call assign_splits first)")

        added = 0
        round_ids: set[str] = set()
        for sample in samples:
            # Invalid for supervised learning: human action not among the legal
            # actions.  Count and skip (DATA_SPEC section 11 quality check).
            if sample.action not in sample.legal_actions:
                self._counts["samples_skipped"] += 1
                continue
            self._buffers[split].append(sample_to_row(sample))
            added += 1
            round_ids.add(sample.round_id)
            if len(self._buffers[split]) >= self.chunk_size:
                self._flush(split)

        if added > 0:
            self._counts["samples"] += added
            self._counts["games"] += 1
            self._counts["rounds"] += len(round_ids)
            self._counts["splits"][split]["games"] += 1
            self._counts["splits"][split]["samples"] += added

    # -- streaming build over real files ----------------------------------------
    def build(self, files: Sequence[str | Path]) -> dict:
        """Run the full pipeline over ``files`` and write the manifest."""
        paths = [Path(f) for f in files]
        game_ids = [p.stem for p in paths]
        self.assign_splits(game_ids)
        self._counts["files_requested"] = len(paths)

        parser = MjaiParser()
        for idx, path in enumerate(paths, 1):
            game_id = path.stem
            if game_id in self.processed_games:
                continue  # 断点续传：跳过已完成的 game

            if self.progress_interval and idx % self.progress_interval == 0:
                log.info(
                    "progress %d/%d files (processed %d, games %d, samples %d, skipped %d)",
                    idx,
                    len(paths),
                    self._counts["files_processed"],
                    self._counts["games"],
                    self._counts["samples"],
                    self._counts["samples_skipped"],
                )

            events = []
            try:
                for line_no, line in enumerate(Path(path).open("r", encoding="utf-8"), 1):
                    try:
                        events.append(parser.parse_line(line, line_no))
                    except MjaiParseError as exc:
                        raise MjaiParseError(str(exc), line_no=exc.line_no) from exc
            except (MjaiParseError, OSError) as exc:
                self._record_error(path, f"parse: {exc}")
                continue

            try:
                states = list(ReplayEngine(game_id=game_id).replay(events))
            except ReplayInconsistency as exc:
                self._record_error(
                    path, f"replay: [{exc.event_type} @step {exc.step}] {exc}"
                )
                continue

            extractor = DecisionExtractor(game_id=game_id)
            try:
                samples = list(extractor.extract(states, events))
            except Exception as exc:  # rule/extractor bug must not abort the run
                self._record_error(path, f"extract: {type(exc).__name__}: {exc}")
                continue

            self.add_game_samples(game_id, samples)
            self._counts["files_processed"] += 1
            self.new_processed_games.add(game_id)

        return self.finish()

    def _record_error(self, path: Path, message: str) -> None:
        self._counts["files_error"] += 1
        self.errors.append(f"{path.name}: {message}")

    # -- manifest ---------------------------------------------------------------
    def finish(self) -> dict:
        for split in SPLIT_NAMES:
            self._flush(split)
        config = {
            "chunk_size": self.chunk_size,
            "split_ratios": {name: self.ratios[i] for i, name in enumerate(SPLIT_NAMES)},
            "seed": self.seed,
            "source_dir": str(self.source_dir) if self.source_dir else None,
            "limit": self.limit,
        }
        manifest = build_manifest(
            dataset_version=self.dataset_version,
            action_schema_version=self.action_schema_version,
            feature_version=self.feature_version,
            config=config,
            counts=self._counts,
            git_commit_hash=self.git_commit_hash,
        )
        write_manifest(self.output_dir / "manifest.json", manifest)
        return manifest
