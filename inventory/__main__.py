"""Data inventory of both cohorts. Run from the repo root:

    python -m inventory                                     # every visit
    python -m inventory --visits L03_Ramp L12_PRBS G11_Hard # a subset
    python -m inventory --no-plots

Writes to outputs/inventory/ (gitignored): CSVs, summary.md, plots/<visit_id>.png, inventory.log.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: files only, before anything imports pyplot

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402

from inventory.annotations import annotation_for, load_annotations  # noqa: E402
from inventory.catalog import OUTPUT_ROOT, VisitRef, enumerate_visits  # noqa: E402
from inventory.metrics import signal_rows, visit_row  # noqa: E402
from inventory.plots import plot_visit  # noqa: E402
from inventory.report import identity, write_reports  # noqa: E402
from inventory.sensors import Loaded, load_participant_fitdays, load_visit  # noqa: E402

logger = logging.getLogger("inventory")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m inventory", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visits", nargs="+", metavar="VISIT_ID", help="only these visit_ids, e.g. G11_Hard")
    parser.add_argument("--no-plots", action="store_true", help="skip the per-visit PNGs")
    return parser.parse_args(argv)


def _configure_logging() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_ROOT / "inventory.log", mode="w")],
    )


def _select(refs: list[VisitRef], wanted: list[str] | None) -> list[VisitRef]:
    if not wanted:
        return refs
    known = {ref.visit_id for ref in refs}
    unknown = sorted(set(wanted) - known)
    if unknown:
        raise SystemExit(f"Unknown visit_id(s): {', '.join(unknown)}\nKnown: {', '.join(sorted(known))}")
    return [ref for ref in refs if ref.visit_id in wanted]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging()
    refs = _select(enumerate_visits(), args.visits)
    annotations = load_annotations()
    fitdays_by_participant: dict[str, Loaded] = {}
    visit_rows, all_signal_rows, people = [], [], {}
    for number, ref in enumerate(refs, start=1):
        logger.info("[%d/%d] %s (%s %s/%s)", number, len(refs), ref.visit_id, ref.cohort, ref.source_participant, ref.source_visit)
        participant_dir = str(ref.path.parent)
        if participant_dir not in fitdays_by_participant:
            fitdays_by_participant[participant_dir] = load_participant_fitdays(ref.path.parent)
        visit = load_visit(ref, annotation_for(ref, annotations), fitdays_by_participant[participant_dir])
        signals = signal_rows(visit)
        visit_rows.append(visit_row(visit, signals))
        all_signal_rows.extend(signals)
        if identity(visit) is not None:
            people.setdefault(ref.participant_id, identity(visit))
        if not args.no_plots:
            try:
                plot_visit(visit, OUTPUT_ROOT / "plots")
            except Exception:  # a broken plot must not lose the CSVs; logged with traceback
                logger.exception("%s: plot failed", ref.visit_id)
    write_reports(people, visit_rows, all_signal_rows, OUTPUT_ROOT)
    logger.info("Wrote %s", OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
