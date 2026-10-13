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


# --- Aim 2 (METHODOLOGY Appendix B; approved 13 Oct 2026) --------------------------------------------------------
# Each spec wraps a plain `Run` (data pool, configurations, explicit test folds) and says which visits train each
# fold; `pipeline.aim2` trains it and `evaluate(spec.run)` summarises it like an A1 run. Never pass `spec.run` to
# `train.run`: that would train plain LOSO folds into the Aim 2 run folder.

A2_CONFIGURATIONS = ("WR", "HR+VENT+TSI+ANTHRO", "WR+VENT")  # carry-forward rule (METHODOLOGY §12)
A2A_STUDY, A2B_STUDY, A2C_STUDY = "2a", "2b", "2c"


@dataclass(frozen=True)
class LeftOutArm:
    """2a/2b, one arm for one left-out condition (Appendix B.3-B.4).

    Unseen arm: each fold trains on the other participants' `pool` visits except `left_out`. Seen arm: on all their
    `pool` visits, subsampled to exactly the unseen arm's number of training targets in that fold.
    """

    study: str
    run: Run  # one cohort; fold_ids = the participants with `left_out` targets
    pool: tuple[str, ...]  # protocols that may train (2a: the PRBS levels only, so never the Gies Ramp)
    left_out: str
    is_seen: bool

    def __post_init__(self) -> None:
        if len(self.run.cohorts) != 1 or self.run.mask != SHARED_MASK or not self.run.fold_ids:
            raise ValueError(f"{self.run.run_id}: needs one cohort, the shared mask and explicit fold_ids")
        if self.left_out not in self.pool or len(self.pool) < 2:
            raise ValueError(f"{self.run.run_id}: left_out {self.left_out!r} must be one of a pool {self.pool}")


@dataclass(frozen=True)
class CohortTransfer:
    """2c (Appendix B.5): one training on every visit of `train_cohort`, tested on the other cohort's participants
    (`run.fold_ids`); compared with that test cohort's within-cohort A1 run, `reference_run_id`."""

    run: Run  # cohorts = both
    train_cohort: str
    reference_run_id: str

    def __post_init__(self) -> None:
        if set(self.run.cohorts) != set(COHORTS) or self.train_cohort not in COHORTS:
            raise ValueError(f"{self.run.run_id}: pool must be both cohorts; train cohort {self.train_cohort!r}")
        if self.run.mask != SHARED_MASK or not self.run.fold_ids:
            raise ValueError(f"{self.run.run_id}: needs the shared mask and explicit fold_ids (test participants)")

    @property
    def test_cohort(self) -> str:
        return next(c for c in COHORTS if c != self.train_cohort)


A2A_POOL = ("Easy", "Medium", "Hard")
A2B_POOL = ("Ramp", "PRBS", "Interval", "Step")
# Participants with targets in each Gies level under the shared mask (A1_pooled targets.csv; checked against the
# cache by `aim2.check_test_participants` and the real-data test).
A2A_TEST_PARTICIPANTS = {
    "Easy": ("G01", "G03", "G04", "G05", "G06", "G07", "G09", "G10", "G11", "G12", "G14", "G15"),
    "Medium": ("G01", "G03", "G04", "G05", "G06", "G07", "G08", "G09", "G10", "G11", "G12", "G13", "G15"),
    "Hard": ("G01", "G03", "G04", "G05", "G06", "G08", "G09", "G10", "G11", "G12", "G13", "G14", "G15"),
}
A2B_TEST_PARTICIPANTS = {protocol: PARTICIPANTS["Lucas"] for protocol in A2B_POOL}


def left_out_arms(study: str, cohort: str, pool: tuple[str, ...],
                  test_participants: dict[str, tuple[str, ...]]) -> tuple[LeftOutArm, ...]:
    """Unseen then seen arm per left-out condition; run IDs `A<study>_<condition>_<arm>`."""
    return tuple(
        LeftOutArm(study, Run(f"A{study}_{condition.lower()}_{'seen' if is_seen else 'unseen'}", (cohort,),
                              A2_CONFIGURATIONS, fold_ids=test_participants[condition]),
                   pool, condition, is_seen)
        for condition in pool for is_seen in (False, True)
    )


A2A_ARMS = left_out_arms(A2A_STUDY, "Gies", A2A_POOL, A2A_TEST_PARTICIPANTS)
A2B_ARMS = left_out_arms(A2B_STUDY, "Lucas", A2B_POOL, A2B_TEST_PARTICIPANTS)
A2C_TRANSFERS = (
    CohortTransfer(Run("A2c_gies_to_lucas", COHORTS, A2_CONFIGURATIONS, fold_ids=PARTICIPANTS["Lucas"]),
                   "Gies", A1_LUCAS.run_id),
    CohortTransfer(Run("A2c_lucas_to_gies", COHORTS, A2_CONFIGURATIONS, fold_ids=PARTICIPANTS["Gies"]),
                   "Lucas", A1_GIES.run_id),
)

# Aim 2 smoke (plumbing only, results unused): 2a Hard, both arms, on fold G01, and 2c Gies -> Lucas; seed 0, 3 epochs.
A2_SMOKE_TRAINING = replace(A1_TRAINING, epochs=SMOKE_EPOCHS)
A2_SMOKE_ARMS = tuple(
    replace(arm, run=replace(arm.run, run_id=f"smoke_{arm.run.run_id}", seeds=(0,), training=A2_SMOKE_TRAINING,
                             fold_ids=("G01",)))
    for arm in A2A_ARMS if arm.left_out == "Hard"
)
A2_SMOKE_TRANSFER = replace(A2C_TRANSFERS[0], run=replace(A2C_TRANSFERS[0].run, run_id="smoke_A2c_gies_to_lucas",
                                                          seeds=(0,), training=A2_SMOKE_TRAINING))
