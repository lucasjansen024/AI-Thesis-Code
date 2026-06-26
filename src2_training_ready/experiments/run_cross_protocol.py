from dataclasses import dataclass
from itertools import product

from src2_training_ready.constants import VisitSelection, ParticipantSelection
from src2_training_ready.processing.processors import (
    ALL_SIGNALS,
    VO2_OR_WR,
    VO2_OR_HR_HRR,
    VO2_OR_WR_HR_HRR,
    VO2_OR_WR_HR_VE_median_TSI_norm_,
    VO2_OR_WR_VE_OR_median,
    VO2_OR_WR_Hb_diff3_norm,
    VO2_OR_WR_HRR_VE_OR_median_TSI_norm,
    VO2_OR_WR_HRR_VE_OR_median,
    VO2_OR_WR_HR_HRR_VE_OR_median,
    VO2_OR_Hb_diff_norm,
    VO2_OR_WR_gender_smm,
    VO2_OR_WR_HRR_Hb_diff,
    VO2_OR_WR_HRR_VE_OR_median,
    VO2_WR_HRR_VE_Hb_diff_gender_smm,
    VO2_OR_WR_HRR_VE_OR_median_Hb_diff_norm_gender_smm,
    VO2_OR_WR_HR_HRR_VE_OR_median_Hb_diff_norm,
    VO2_OR_WR_HRR_VE_OR_median_Hb_diff_norm,
    VO2_OR_WR_HRR_Hb_diff_gender_smm,
    VO2_OR_WR_HRR_Hb_diff,
    VO2_OR_WR_HRR_TSI_norm,
    VO2_OR_WR_HR_HRR_Hb_diff_norm_gender_smm,
    VO2_OR_TSI_norm,
    VO2_OR_HR,
    VO2_OR_HRR,
    VO2_OR_VE_OR_median,
    VO2_OR_Hb_diff_norm,
    VO2_OR_WR_HRR,
    VO2_OR_WR_HR,
    VO2_OR_WR_TSI_norm,
    VO2_OR_WR_gender,
    VO2_OR_WR_smm,
    VO2_OR_WR_Hb_diff,   
    VO2_OR_WR_VE_OR_median_Hb_diff_norm, 
    VO2_OR_WR_HHb_avg,
    VO2_OR_HHb_avg,
    VO2_OR_WR_gender_Hb_diff_norm,
    VO2_OR_WR_gender_TSI_norm,
    VO2_OR_WR_gender_smm_Hb_diff_norm,
    VO2_OR_WR_gender_smm_TSI_norm,
    VO2_OR_WR_smm_Hb_diff_norm,
    VO2_OR_WR_smm_TSI_norm,
    VO2_OR_WR_gender_Hb_diff_norm_VE_OR_median,
    VO2_OR_WR_gender_smm_Hb_diff_norm_VE_OR_median,
    VO2_OR_WR_gender_Hb_diff_norm_HRR,
    VO2_OR_WR_gender_smm_Hb_diff_norm_HRR,
    VO2_OR_WR_gender_Hb_diff_norm_HRR_VE_OR_median,
    VO2_OR_WR_gender_smm_Hb_diff_norm_HRR_VE_OR_median,
    VO2_OR_WR_HR_HRR_Hb_diff_norm_gender_smm,
    VO2_OR_WR_gender_Hb_diff3_norm,
    VO2_OR_WR_gender_smm_Hb_diff3_norm,
    VO2_OR_WR_gender_Hb_diff_norm_TSI_norm,
    VO2_OR_WR_gender_smm_Hb_diff_norm_TSI_norm,
    VO2_OR_WR_VE_OR_median_TSI_norm,
    VO2_OR_WR_TSI_norm_Hb_diff_norm,
    VO2_OR_WR_HR_VE_OR_median,
    VO2_OR_WR_VE_OR_median_TSI_norm_Hb_diff_norm,
    VO2_OR_WR_VE_OR_median_gender,
    VO2_OR_WR_VE_OR_median_smm,   
    VO2_OR_Hb_diff_norm_gender,
    VO2_OR_Hb_diff_norm_smm,
    VO2_OR_Hb_diff_norm_gender_smm,
    VO2_OR_Hb_diff_norm_VE_OR_median,
    VO2_OR_Hb_diff_norm_VE_OR_median_gender_smm,
    VO2_OR_Hb_diff_norm_VE_OR_median_HR_gender_smm,
    VO2_OR_Hb_diff_norm_HR,
    VO2_OR_Hb_diff_norm_VE_OR_median_HR, 
    VO2_OR_Hb_diff_norm_VE_OR_median_HRR_gender_smm,
    VO2_OR_Hb_diff_norm_VE_OR_median_HRR,
    VO2_OR_VE_OR_median_HRR,
    VO2_OR_Hb_diff_norm_HRR,
    VO2_OR_VE_OR_median_HR,
    VO2_OR_Hb_diff_norm_VE_OR_median_gender,
    VO2_OR_Hb_diff_norm_VE_OR_median_smm,
    VO2_OR_Hb_diff_norm_VE_OR_median_TSI_norm,
    VO2_OR_Hb_diff_norm_VE_OR_median_TSI_norm_smm,
    VO2_OR_Hb_diff_norm_VE_OR_median_TSI_norm_gender,
    VO2_OR_Hb_diff_norm_TSI_norm_smm,
)
from src2_training_ready.train_testing.manager import Manager, ManagerConfig
from src2_training_ready.models.models import (
    TCN_30, GRU_30,
    TCN_SHORT_DO00, TCN_SHORT_DO02, TCN_SHORT_DO04,
    TCN_BASE_DO00, TCN_BASE_DO02, TCN_BASE_DO04,
    TCN_LONG_DO00, TCN_LONG_DO02, TCN_LONG_DO04,
    GRU_SHORT_DO00, GRU_SHORT_DO02, GRU_SHORT_DO04,
    GRU_BASE_DO00, GRU_BASE_DO02, GRU_BASE_DO04,
    GRU_LONG_DO00, GRU_LONG_DO02, GRU_LONG_DO04,
    TCN_BASE_DO02_5,
)


@dataclass
class ExperimentSpec:
    name: str
    participant_selection: ParticipantSelection
    train_selection: VisitSelection
    test_selection: VisitSelection


def run_experiment(spec: ExperimentSpec, model, processor, logging=True, plot=True, force=False, test_best=False):
    manager = Manager(
        config=ManagerConfig(
            visit_selection=VisitSelection.SUBMAX_PLUS_RAMP,
            participant_selection=spec.participant_selection,
            model=model,
            train_visit_selection=spec.train_selection,
            test_visit_selection=spec.test_selection,
        ),
        logging=logging,
        plot_MNG=False,
    )
    manager.set_processor(processor)

    model_folder = manager.train_LOOCV()
    manager.test(model_folder, plot=plot, force=force)

    if test_best:
        manager.test_best(plot=plot)

    return model_folder


# -----------------------------------------------------------------------------
# Legacy experiment sets (kept for compatibility with older runs)
# -----------------------------------------------------------------------------
WITHIN_PROTOCOL = [
    ExperimentSpec(
        "within_ramp",
        ParticipantSelection.RAMP,
        VisitSelection.ONLY_RAMP,
        VisitSelection.ONLY_RAMP,
    ),
    ExperimentSpec(
        "within_prbs",
        ParticipantSelection.PRBS,
        VisitSelection.ONLY_PRBS,
        VisitSelection.ONLY_PRBS,
    ),
    ExperimentSpec(
        "within_interval",
        ParticipantSelection.INTERVAL,
        VisitSelection.ONLY_INTERVAL,
        VisitSelection.ONLY_INTERVAL,
    ),
    ExperimentSpec(
        "within_step",
        ParticipantSelection.STEP,
        VisitSelection.ONLY_STEP,
        VisitSelection.ONLY_STEP,
    ),
]

BETWEEN_PROTOCOL = [
    ExperimentSpec(
        "ramp_to_prbs",
        ParticipantSelection.RAMP_PRBS,
        VisitSelection.ONLY_RAMP,
        VisitSelection.ONLY_PRBS,
    ),
    ExperimentSpec(
        "prbs_to_ramp",
        ParticipantSelection.RAMP_PRBS,
        VisitSelection.ONLY_PRBS,
        VisitSelection.ONLY_RAMP,
    ),
    ExperimentSpec(
        "ramp_to_interval",
        ParticipantSelection.RAMP_INTERVAL,
        VisitSelection.ONLY_RAMP,
        VisitSelection.ONLY_INTERVAL,
    ),
    ExperimentSpec(
        "interval_to_ramp",
        ParticipantSelection.RAMP_INTERVAL,
        VisitSelection.ONLY_INTERVAL,
        VisitSelection.ONLY_RAMP,
    ),
    ExperimentSpec(
        "ramp_to_step",
        ParticipantSelection.RAMP_STEP,
        VisitSelection.ONLY_RAMP,
        VisitSelection.ONLY_STEP,
    ),
    ExperimentSpec(
        "step_to_ramp",
        ParticipantSelection.RAMP_STEP,
        VisitSelection.ONLY_STEP,
        VisitSelection.ONLY_RAMP,
    ),
    ExperimentSpec(
        "prbs_to_interval",
        ParticipantSelection.PRBS_INTERVAL,
        VisitSelection.ONLY_PRBS,
        VisitSelection.ONLY_INTERVAL,
    ),
    ExperimentSpec(
        "interval_to_prbs",
        ParticipantSelection.PRBS_INTERVAL,
        VisitSelection.ONLY_INTERVAL,
        VisitSelection.ONLY_PRBS,
    ),
    ExperimentSpec(
        "prbs_to_step",
        ParticipantSelection.PRBS_STEP,
        VisitSelection.ONLY_PRBS,
        VisitSelection.ONLY_STEP,
    ),
    ExperimentSpec(
        "step_to_prbs",
        ParticipantSelection.PRBS_STEP,
        VisitSelection.ONLY_STEP,
        VisitSelection.ONLY_PRBS,
    ),
    ExperimentSpec(
        "interval_to_step",
        ParticipantSelection.INTERVAL_STEP,
        VisitSelection.ONLY_INTERVAL,
        VisitSelection.ONLY_STEP,
    ),
    ExperimentSpec(
        "step_to_interval",
        ParticipantSelection.INTERVAL_STEP,
        VisitSelection.ONLY_STEP,
        VisitSelection.ONLY_INTERVAL,
    ),
]

LOPO_PROTOCOL = [
    ExperimentSpec(
        "interval_step_to_prbs",
        ParticipantSelection.PRBS_INTERVAL_STEP,
        VisitSelection.INTERVAL_STEP,
        VisitSelection.ONLY_PRBS,
    ),
    ExperimentSpec(
        "prbs_step_to_interval",
        ParticipantSelection.PRBS_INTERVAL_STEP,
        VisitSelection.PRBS_STEP,
        VisitSelection.ONLY_INTERVAL,
    ),
    ExperimentSpec(
        "prbs_interval_to_step",
        ParticipantSelection.PRBS_INTERVAL_STEP,
        VisitSelection.PRBS_INTERVAL,
        VisitSelection.ONLY_STEP,
    ),
]


# -----------------------------------------------------------------------------
# Thesis experiment sets
# Protocol-inclusive = train on all 4 protocols, test on one target protocol
# LOPO               = train on all protocols except the target protocol
# -----------------------------------------------------------------------------
PROTOCOL_INCLUSIVE = [
    ExperimentSpec(
        "all_to_interval",
        ParticipantSelection.COMPLETE_4_PROTOCOLS,
        VisitSelection.ALL_PROTOCOLS,
        VisitSelection.ONLY_INTERVAL,
    ),
    ExperimentSpec(
        "all_to_step",
        ParticipantSelection.COMPLETE_4_PROTOCOLS,
        VisitSelection.ALL_PROTOCOLS,
        VisitSelection.ONLY_STEP,
    ),
    ExperimentSpec(
        "all_to_prbs",
        ParticipantSelection.COMPLETE_4_PROTOCOLS,
        VisitSelection.ALL_PROTOCOLS,
        VisitSelection.ONLY_PRBS,
    ),
    ExperimentSpec(
        "all_to_ramp",
        ParticipantSelection.COMPLETE_4_PROTOCOLS,
        VisitSelection.ALL_PROTOCOLS,
        VisitSelection.ONLY_RAMP,
    ),
]

THESIS_LOPO = [
    ExperimentSpec(
        "ramp_prbs_step_to_interval",
        ParticipantSelection.INTERVAL,
        VisitSelection.ALL_EXCEPT_INTERVAL,
        VisitSelection.ONLY_INTERVAL,
    ),
    ExperimentSpec(
        "ramp_prbs_interval_to_step",
        ParticipantSelection.STEP,
        VisitSelection.ALL_EXCEPT_STEP,
        VisitSelection.ONLY_STEP,
    ),
    ExperimentSpec(
        "ramp_interval_step_to_prbs",
        ParticipantSelection.PRBS,
        VisitSelection.ALL_EXCEPT_PRBS,
        VisitSelection.ONLY_PRBS,
    ),
    ExperimentSpec(
        "prbs_interval_step_to_ramp",
        ParticipantSelection.RAMP,
        VisitSelection.ALL_EXCEPT_RAMP,
        VisitSelection.ONLY_RAMP,
    ),
]

POOLED_DEVELOPMENT = [
    ExperimentSpec(
        "all_to_all_pooled_dev",
        ParticipantSelection.COMPLETE_4_PROTOCOLS,
        VisitSelection.ALL_PROTOCOLS,
        VisitSelection.ALL_PROTOCOLS,
    ),
]

THESIS_ALL_EXPERIMENTS = [*PROTOCOL_INCLUSIVE, *THESIS_LOPO]


BALANCED_PROTOCOL_CONTROL = []

train_sets = [
    ("no_ramp", VisitSelection.ALL_EXCEPT_RAMP),
    ("no_prbs", VisitSelection.ALL_EXCEPT_PRBS),
    ("no_interval", VisitSelection.ALL_EXCEPT_INTERVAL),
    ("no_step", VisitSelection.ALL_EXCEPT_STEP),
]

target_sets = [
    ("interval", VisitSelection.ONLY_INTERVAL),
    ("step", VisitSelection.ONLY_STEP),
    ("prbs", VisitSelection.ONLY_PRBS),
    ("ramp", VisitSelection.ONLY_RAMP),
]

for target_name, test_selection in target_sets:
    for train_name, train_selection in train_sets:
        BALANCED_PROTOCOL_CONTROL.append(
            ExperimentSpec(
                name=f"{train_name}_to_{target_name}",
                participant_selection=ParticipantSelection.COMPLETE_4_PROTOCOLS,
                train_selection=train_selection,
                test_selection=test_selection,
            )
        )
                
if __name__ == "__main__":
    spec = POOLED_DEVELOPMENT[0]

    # Best-performing input combination:
    # WR + gender + SMM + normalized Hb-diff
    processor = VO2_OR_WR_gender_smm_Hb_diff_norm

    models_to_run = [
        TCN_BASE_DO02,
        GRU_BASE_DO02,
    ]

    for model in models_to_run:
        model_name = getattr(model, "name", model.__class__.__name__)

        print(
            f"Running {spec.name} with {model_name} "
            f"and processor {processor.get_folder_name()}"
        )

        run_experiment(
            spec=spec,
            model=model,
            processor=processor,
            plot=True,
            force=True,
            test_best=False,
        )
