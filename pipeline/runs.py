"""Run registry: frozen `Run` dataclasses picked in `main.py` by commenting/uncommenting. No YAML, no CLI.

The primary training setting is one shared literal (lr 5e-5 x 30 epochs); a training-setting sensitivity reruns
the A1 runs at lr 5e-4 x 1 epoch (dev, 9 Oct 2026; replaces the pilot's choice). Every A1 run trains all 46
configurations x 5 seeds on LOSO folds of its cohorts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from pipeline.domains import check_nirs_exclusive, generate_configurations, parse_configuration
from pipeline.prepare.settings import PrepSettings

COHORTS = ("Gies", "Lucas")
SEEDS = (0, 1, 2, 3, 4)
SHARED_MASK = "shared"
PER_CONFIG_MASK = "per_config"  # sensitivity: each configuration uses every unit valid for its own groups
SUPPORTED_MASKS = (SHARED_MASK, PER_CONFIG_MASK)


@dataclass(frozen=True)
class TrainingSettings:
    learning_rate: float = 5e-5
    epochs: int = 30
    batch_size: int = 32


A1_TRAINING = TrainingSettings()


@dataclass(frozen=True)
class Run:
    run_id: str
    cohorts: tuple[str, ...]  # data pool: every participant of these cohorts trains or is tested
    configurations: tuple[str, ...]  # configuration IDs, e.g. ("WR",)
    seeds: tuple[int, ...] = SEEDS
    mask: str = SHARED_MASK
    training: TrainingSettings = A1_TRAINING
    prep: PrepSettings = field(default_factory=PrepSettings)
    fold_ids: tuple[str, ...] | None = None  # the run's folds (test participant IDs); None = LOSO over all
    nirs_exclusive: bool = True  # TSI and HBDIFF never in one configuration (§7 flag; False allows it)

    def __post_init__(self) -> None:
        if self.mask not in SUPPORTED_MASKS:
            raise ValueError(f"{self.run_id}: mask {self.mask!r} not supported (have {SUPPORTED_MASKS})")
        unknown = set(self.cohorts) - set(COHORTS)
        if unknown or not self.cohorts:
            raise ValueError(f"{self.run_id}: cohorts {self.cohorts}, expected a subset of {COHORTS}")
        if not self.configurations or not self.seeds:
            raise ValueError(f"{self.run_id}: needs at least one configuration and one seed")
        self._check_configurations()

    def _check_configurations(self) -> None:
        """Valid, unique IDs; under `nirs_exclusive` none combines TSI and HBDIFF."""
        if len(set(self.configurations)) != len(self.configurations):
            raise ValueError(f"{self.run_id}: repeated configuration in {self.configurations}")
        for config_id in self.configurations:
            config = parse_configuration(config_id)
            if self.nirs_exclusive:
                check_nirs_exclusive(config)


A1_CONFIGURATIONS = generate_configurations()
A1_POOLED = Run("A1_pooled", COHORTS, A1_CONFIGURATIONS)
A1_GIES = Run("A1_gies", ("Gies",), A1_CONFIGURATIONS)
A1_LUCAS = Run("A1_lucas", ("Lucas",), A1_CONFIGURATIONS)
A1_RUNS = (A1_POOLED, A1_GIES, A1_LUCAS)

# `per_config` sensitivity runs (§2.8): same data pools, configurations and settings, own run_ids.
A1_POOLED_PER_CONFIG = replace(A1_POOLED, run_id="A1_pooled_per_config", mask=PER_CONFIG_MASK)
A1_GIES_PER_CONFIG = replace(A1_GIES, run_id="A1_gies_per_config", mask=PER_CONFIG_MASK)
A1_LUCAS_PER_CONFIG = replace(A1_LUCAS, run_id="A1_lucas_per_config", mask=PER_CONFIG_MASK)
A1_PER_CONFIG_RUNS = (A1_POOLED_PER_CONFIG, A1_GIES_PER_CONFIG, A1_LUCAS_PER_CONFIG)

# Training-setting sensitivity runs (dev, 9 Oct 2026): pilot arm C (lr 5e-4, checkpoint at epoch 1), otherwise
# identical to the primary runs. No LR schedule, so a 1-epoch run follows the arm's procedure (dropout RNG differs).
A1_SHORT_TRAINING = TrainingSettings(learning_rate=5e-4, epochs=1)
A1_POOLED_SHORT = replace(A1_POOLED, run_id="A1_pooled_lr5e-4_ep1", training=A1_SHORT_TRAINING)
A1_GIES_SHORT = replace(A1_GIES, run_id="A1_gies_lr5e-4_ep1", training=A1_SHORT_TRAINING)
A1_LUCAS_SHORT = replace(A1_LUCAS, run_id="A1_lucas_lr5e-4_ep1", training=A1_SHORT_TRAINING)
A1_SHORT_RUNS = (A1_POOLED_SHORT, A1_GIES_SHORT, A1_LUCAS_SHORT)

# Plumbing check before the pilot (CONTEXT.md "Smoke run"); its results are not used.
SMOKE_EPOCHS = 3
SMOKE = Run("smoke", COHORTS, ("WR", "WR+HR+VENT+TSI+ANTHRO"), seeds=(0,),
            training=replace(A1_TRAINING, epochs=SMOKE_EPOCHS), fold_ids=("G07", "L04"))


# --- Training-settings pilot (§3.0, ticket 09) ------------------------------------------------------------------
# LOSO (dev, 9 Oct 2026; departs from §3.0's validation splits): every family holds out each of its participants
# once, like the A1 folds, so the held-out participant is an A1 test participant. `pipeline.pilot` trains and
# tabulates; the dev picks the setting from the table.

POOLED_FAMILY = "pooled"
PILOT_FAMILIES = (POOLED_FAMILY, *COHORTS)


@dataclass(frozen=True)
class PilotSplit:
    split_id: str
    family: str  # "pooled" trains on both cohorts; "Gies"/"Lucas" train on the rest of that cohort
    validation: tuple[str, ...]  # held-out participant IDs

    @property
    def cohorts(self) -> tuple[str, ...]:
        return COHORTS if self.family == POOLED_FAMILY else (self.family,)


@dataclass(frozen=True)
class PilotTrajectory:
    trajectory_id: str  # output folder name
    training: TrainingSettings


@dataclass(frozen=True)
class Pilot:
    pilot_id: str  # outputs/<pilot_id>/<split_id>/<trajectory_id>/
    splits: tuple[PilotSplit, ...]
    trajectories: tuple[PilotTrajectory, ...]
    configurations: tuple[str, ...]
    seeds: tuple[int, ...]
    prep: PrepSettings = field(default_factory=PrepSettings)
    is_loso: bool = False  # every participant of each family held out exactly once (checked against the cache)

    def __post_init__(self) -> None:
        if not (self.splits and self.trajectories and self.configurations and self.seeds):
            raise ValueError(f"{self.pilot_id}: needs splits, trajectories, configurations and seeds")
        for label, ids in (("split", [s.split_id for s in self.splits]),
                           ("trajectory", [t.trajectory_id for t in self.trajectories]),
                           ("configuration", list(self.configurations)), ("seed", list(self.seeds))):
            if len(set(ids)) != len(ids):
                raise ValueError(f"{self.pilot_id}: repeated {label} in {ids}")
        bad = [s.split_id for s in self.splits if s.family not in PILOT_FAMILIES or not s.validation]
        if bad:
            raise ValueError(f"{self.pilot_id}: splits {bad} need a family in {PILOT_FAMILIES} and participants")
        for config_id in self.configurations:
            check_nirs_exclusive(parse_configuration(config_id))


# Participant IDs per cohort (CONTEXT.md); `pipeline.pilot.check_loso_coverage` checks them against the cache.
PARTICIPANTS = {
    "Gies": ("G01", *(f"G{n:02d}" for n in range(3, 16))),
    "Lucas": (*(f"L{n:02d}" for n in range(1, 11)), "L12", "L13", "L14"),
}


def loso_pilot_splits() -> tuple[PilotSplit, ...]:
    """One split per held-out participant: pooled (27), then Gies (14), then Lucas (13)."""
    pooled = tuple(PilotSplit(f"{POOLED_FAMILY}_{p}", POOLED_FAMILY, (p,)) for c in COHORTS for p in PARTICIPANTS[c])
    within = tuple(PilotSplit(f"{c.lower()}_{p}", c, (p,)) for c in COHORTS for p in PARTICIPANTS[c])
    return pooled + within


PILOT_SPLITS = loso_pilot_splits()
# Arms R/A read off the first trajectory, B the second, D/C the third (`pipeline.pilot.ARMS`).
PILOT_TRAJECTORY_R, PILOT_TRAJECTORY_B, PILOT_TRAJECTORY_D = "lr5e-5_ep30", "lr2e-4_ep10", "lr5e-4_ep3"
PILOT_TRAJECTORIES = (
    PilotTrajectory(PILOT_TRAJECTORY_R, TrainingSettings(learning_rate=5e-5, epochs=30)),
    PilotTrajectory(PILOT_TRAJECTORY_B, TrainingSettings(learning_rate=2e-4, epochs=10)),
    PilotTrajectory(PILOT_TRAJECTORY_D, TrainingSettings(learning_rate=5e-4, epochs=3)),
)
PILOT_CONFIGURATIONS = ("WR", "WR+HR+VENT+TSI+ANTHRO", "WR+HR+VENT+HBDIFF+ANTHRO", "HR+VENT+TSI+ANTHRO",
                        "HR+VENT+HBDIFF+ANTHRO")  # replaces §3.0's three (dev, 9 Oct 2026)
PILOT_SEEDS = (0, 1)
PILOT = Pilot("pilot", PILOT_SPLITS, PILOT_TRAJECTORIES, PILOT_CONFIGURATIONS, PILOT_SEEDS, is_loso=True)

# Pilot smoke (CONTEXT.md): checks the held-out logging path only; picks no setting.
PILOT_SMOKE = Pilot("pilot_smoke", PILOT_SPLITS[:1], PILOT_TRAJECTORIES[2:], ("WR", "WR+HR+VENT+TSI+ANTHRO"), (0,))
