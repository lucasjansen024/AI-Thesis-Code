from enum import StrEnum
from itertools import product, combinations
from typing import Optional, Dict

from .function import ProcessingFunction
from .processor import Processor
from .functions import *
from ..data_types.base import Domain


_domain_map: Dict[Domain, List[ProcessingFunction]] = {
        Domain.WR: [Process_WR()],
        Domain.HR: [
            Process_HR(),
            Process_HRR(),
            Process_HR_HRR(),
        ],
        Domain.VE: [
            Process_VE_OR_median(),
            Process_VE_BR_OR_median(),
            Process_VE_measured_OR_median(),
            # Process_VE_OR(),
        ],
        Domain.NIRS: [
            Process_Hb_diff3_norm(),
            # Process_TSI_norm(),
        ],
        Domain.Fit: [
            Process_gender(),
            Process_smm(),
            Process_gender_smm(),
        ],
    }

class GetProcessorsMode(StrEnum):
    SINGLE = "single"
    SINGLE_WITH = "single_with"
    COMBINATIONS = "combinations"

def get_processors(
    mode: GetProcessorsMode = GetProcessorsMode.COMBINATIONS,
    include_domains: Optional[List[Domain]] = None
) -> List['Processor']:
    VO2_f = Process_VO2_OR_median()

    def functions_to_processor(functions: List[ProcessingFunction]) -> Processor:
        processor = Processor()
        processor.add(VO2_f)

        unique_funcs = []
        for f in functions:
            if all(f != seen_f for seen_f in unique_funcs):
                unique_funcs.append(f)

        for f in unique_funcs:
            processor.add(f)

        return processor

    processors = []

    if mode == GetProcessorsMode.SINGLE:
        # Each processor individually, including WR
        for domain in Domain:
            if domain == Domain.Fit:
                continue
            for proc in _domain_map[domain]:
                processors.append(functions_to_processor([proc]))

    elif mode == GetProcessorsMode.SINGLE_WITH:
        if not include_domains:
            raise ValueError("include_domains must be specified for Mode.SINGLE_WITH")

        # Create all combinations of one processor from each included domain
        included_proc_lists = [_domain_map[domain] for domain in include_domains]
        included_combos = list(product(*included_proc_lists))

        for base_domain in Domain:
            if base_domain in include_domains or base_domain == Domain.Fit:
                continue

            for base_proc in _domain_map[base_domain]:
                for combo in included_combos:
                    processors.append(functions_to_processor([base_proc] + list(combo)))

    elif mode == GetProcessorsMode.COMBINATIONS:
        if include_domains is None:
            include_domains = list(Domain)  # Now WR is not special-cased

        domain_procs_lists = [_domain_map[domain] for domain in include_domains]
        for combo in product(*domain_procs_lists):
            processors.append(functions_to_processor(list(combo)))

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return processors


def get_all_domain_combinations():
    all_domain_combinations = []

    # Loop over all combinations of domains
    for r in range(2, len(Domain) + 1):
        for domain_subset in combinations(Domain, r):
            domain_list = list(domain_subset)

            all_domain_combinations.append(domain_list)

    return all_domain_combinations


def get_all_processors() -> List[Processor]:
    all_processors = []

    # 1. Add processors from SINGLE mode (each processor on its own, including WR)
    all_processors.extend(get_processors(mode=GetProcessorsMode.SINGLE))

    # 2. Add processors from SINGLE_WITH_WR mode (each domain processor with WR)
    all_processors.extend(get_processors(mode=GetProcessorsMode.SINGLE_WITH, include_domains=[Domain.WR]))
    all_processors.extend(get_processors(mode=GetProcessorsMode.SINGLE_WITH, include_domains=[Domain.WR, Domain.Fit]))
    all_processors.extend(get_processors(mode=GetProcessorsMode.SINGLE_WITH, include_domains=[Domain.Fit]))

    # 3. Add processors from COMBINATIONS mode for all non-empty domain subsets
    all_domain_combinations = get_all_domain_combinations()
    for domain_subset in all_domain_combinations:
        processors = get_processors(mode=GetProcessorsMode.COMBINATIONS, include_domains=domain_subset)
        all_processors.extend(processors)

    return all_processors


VO2 = (Processor()
        .add(Process_VO2_Raw())
   )

VO2_WR = (Processor()
        .add(Process_VO2_Raw())
        .add(Process_WR())
    )

VO2_OR_WR = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_WR())
      )

VO2_OR_HR = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_HR())
      )

VO2_OR_HRR = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_HRR())
      )

VO2_OR_HR_HRR = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_HR_HRR())
      )
      
VO2_OR_WR_Hb_diff = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_WR())
      .add(Process_Hb_diff_norm())
      )

VO2_OR_VE_OR = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_VE_OR())
      )

VO2_OR_VE_BR_OR = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_VE_BR_OR())
      )

VO2_OR_VE_OR_median = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_VE_OR_median())
    )

VO2_OR_VE_BR_OR_median = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_VE_BR_OR_median())
    )

VO2_OR_VE_measured_OR = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_VE_measured_OR())
      )

VO2_OR_TSI_norm = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_TSI_norm())
      )

VO2_OR_Hb_diff = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_Hb_diff())
    )

VO2_OR_Hb_diff_norm = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_Hb_diff_norm())
    )

VO2_OR_Hb_diff3_norm = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_Hb_diff3_norm())
    )

VO2_OR_WR_HR_HRR_VE_OR_median = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_HR())
    .add(Process_HRR())
    .add(Process_VE_OR_median())
)

VO2_OR_HHb1_norm = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_HHb1_norm())
      )

VO2_OR_HHb2_norm = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_HHb2_norm())
      )

VO2_OR_HHb3_norm = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_HHb3_norm())
      )

VO2_OR_HHb_avg = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_HHb_avg())
      )

single_input_processors = [
    VO2_OR_WR,
    VO2_OR_HR,
    VO2_OR_HRR,
    VO2_OR_HR_HRR,
    VO2_OR_VE_OR,
    VO2_OR_VE_BR_OR,
    VO2_OR_VE_OR_median,
    VO2_OR_VE_BR_OR_median,
    VO2_OR_VE_measured_OR,
    VO2_OR_TSI_norm,
    VO2_OR_Hb_diff3_norm
    # VO2_OR_HHb1_norm,
    # VO2_OR_HHb2_norm,
    # VO2_OR_HHb3_norm,
    # VO2_OR_HHb_avg,
]

VO2_OR_WR_HR = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_WR())
      .add(Process_HR())
      )

VO2_OR_WR_HR_VE_median_TSI_norm_ = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_HR())
    .add(Process_TSI_norm())
    .add(Process_VE_OR_median())


)

VO2_OR_WR_HRR = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_WR())
      .add(Process_HRR())
      )

VO2_OR_WR_HR_HRR = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_WR())
      .add(Process_HR_HRR())
      )

VO2_OR_WR_VE_OR = (Processor()
      .add(Process_VO2_OR_median())
.add(Process_WR())
      .add(Process_VE_OR())
      )

VO2_OR_WR_VE_BR_OR = (Processor()
      .add(Process_VO2_OR_median())
.add(Process_WR())
      .add(Process_VE_BR_OR())
      )

VO2_OR_WR_VE_OR_median = (Processor()
        .add(Process_VO2_OR_median())
.add(Process_WR())
        .add(Process_VE_OR_median())
    )

VO2_OR_WR_VE_BR_OR_median = (Processor()
      .add(Process_VO2_OR_median())
.add(Process_WR())
      .add(Process_VE_BR_OR_median())
      )

VO2_OR_WR_VE_measured_OR = (Processor()
      .add(Process_VO2_OR_median())
        .add(Process_WR())
      .add(Process_VE_measured_OR())
      )

VO2_OR_WR_TSI_norm = (Processor()
      .add(Process_VO2_OR_median())
.add(Process_WR())
      .add(Process_TSI_norm())
      )

VO2_OR_WR_Hb_diff3_norm = (Processor()
      .add(Process_VO2_OR_median())
.add(Process_WR())
      .add(Process_Hb_diff3_norm())
      )

VO2_OR_WR_HHb1_norm = (Processor()
      .add(Process_VO2_OR_median())
.add(Process_WR())
      .add(Process_HHb1_norm())
      )

VO2_OR_WR_HHb2_norm = (Processor()
      .add(Process_VO2_OR_median())
.add(Process_WR())
      .add(Process_HHb2_norm())
      )

VO2_OR_WR_HHb3_norm = (Processor()
      .add(Process_VO2_OR_median())
.add(Process_WR())
      .add(Process_HHb3_norm())
      )

VO2_OR_WR_HHb_avg = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_WR())
      .add(Process_HHb_avg())
      )

VO2_OR_WR_gender = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_WR())
      .add(Process_gender())
      )
VO2_OR_WR_smm = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_WR())
      .add(Process_smm())
      )
VO2_OR_WR_weight = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_WR())
      .add(Process_weight())
      )
VO2_OR_WR_gender_smm = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_WR())
      .add(Process_gender_smm())
      )
VO2_OR_WR_gender_weight = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_WR())
      .add(Process_gender_weight())
      )
VO2_OR_WR_bf_smm = (Processor()
      .add(Process_VO2_OR_median())
      .add(Process_WR())
      .add(Process_bf_smm())
      )

single_input_processors_with_WR = [
    VO2_OR_WR_HR,
    VO2_OR_WR_HRR,
    VO2_OR_WR_HR_HRR,
    VO2_OR_WR_VE_OR,
    VO2_OR_WR_VE_BR_OR,
    VO2_OR_WR_VE_OR_median,
    VO2_OR_WR_VE_BR_OR_median,
    VO2_OR_WR_VE_measured_OR,
    VO2_OR_WR_TSI_norm,
    # VO2_OR_WR_HHb1_norm,
    # VO2_OR_WR_HHb2_norm,
    # VO2_OR_WR_HHb3_norm,
    # VO2_OR_WR_HHb_avg,
    VO2_OR_WR_gender,
    # VO2_OR_WR_smm,
    # VO2_OR_WR_gender_smm,
    # VO2_OR_WR_bf_smm,
]

VO2_OR_WR_HRR_VE_OR_median = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_VE_OR_median())
    )

VO2_OR_WR_HRR_TSI_norm = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_TSI_norm())
    )

VO2_OR_WR_HRR_gender = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_gender())
    )

VO2_OR_WR_VE_OR_TSI_norm = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_TSI_norm())
        .add(Process_VE_OR())
    )

VO2_OR_WR_VE_OR_gender = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_gender())
        .add(Process_VE_OR())
    )

VO2_OR_WR_HRR_VE_OR_TSI_norm = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_VE_OR())
        .add(Process_TSI_norm())
    )

VO2_OR_WR_HRR_VE_OR_median_TSI_norm = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_VE_OR_median())
        .add(Process_TSI_norm())
    )

VO2_OR_WR_HRR_VE_OR_gender = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_VE_OR())
        .add(Process_gender())
    )

VO2_OR_WR_HRR_VE_OR_median = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_VE_OR_median())
    )


VO2_WR_HRR_VE_Hb_diff_gender_smm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_HRR())
    .add(Process_VE_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_gender_smm())
)

VO2_OR_WR_HRR_VE_OR_TSI_norm_gender = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_VE_OR())
        .add(Process_TSI_norm())
        .add(Process_gender())
    )

VO2_OR_WR_HRR_VE_OR_median_TSI_norm_gender = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_VE_OR_median())
        .add(Process_TSI_norm())
        .add(Process_gender())
    )

VO2_OR_WR_HRR_VE_OR_median_TSI_norm_smm = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_VE_OR_median())
        .add(Process_TSI_norm())
        .add(Process_smm())
    )

VO2_OR_WR_HRR_VE_OR_median_TSI_norm_gender_smm = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_VE_OR_median())
        .add(Process_TSI_norm())
        .add(Process_gender())
        .add(Process_smm())
    )

VO2_OR_WR_HRR_VE_OR_median_Hb_diff3_norm_gender_smm = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_VE_OR_median())
        .add(Process_Hb_diff3_norm())
        .add(Process_gender())
        .add(Process_smm())
    )

VO2_OR_WR_HRR_VE_OR_median_Hb_diff_norm_gender_smm = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_VE_OR_median())
        .add(Process_Hb_diff_norm())
        .add(Process_gender())
        .add(Process_smm())
    )

VO2_OR_WR_HRR_VE_OR_median_Hb_diff_norm = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_VE_OR_median())
        .add(Process_Hb_diff_norm())
    )
    
    

VO2_OR_WR_VE_OR_median_Hb_diff_norm = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_VE_OR_median())
        .add(Process_Hb_diff_norm())
    )

VO2_OR_WR_HR_HRR_VE_OR_median_Hb_diff_norm = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HR_HRR())
        .add(Process_VE_OR_median())
        .add(Process_Hb_diff_norm())
    )



VO2_OR_WR_HR_HRR_Hb_diff_norm_gender_smm = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HR_HRR())
        .add(Process_Hb_diff_norm())
        .add(Process_gender())
        .add(Process_smm())
    )


VO2_OR_WR_HRR_Hb_diff_gender_smm = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_Hb_diff_norm())
        .add(Process_gender())
        .add(Process_smm())
    )

VO2_OR_WR_HRR_Hb_diff = (Processor()
        .add(Process_VO2_OR_median())
        .add(Process_WR())
        .add(Process_HRR())
        .add(Process_Hb_diff_norm())
    )

combined_input_processors = [
    VO2_OR_WR_HRR_VE_OR_median,
    VO2_OR_WR_HRR_TSI_norm,
    VO2_OR_WR_HRR_gender,
    VO2_OR_WR_VE_OR_TSI_norm,
    VO2_OR_WR_VE_OR_gender,
    VO2_OR_WR_HRR_VE_OR_TSI_norm,
    VO2_OR_WR_HRR_VE_OR_median_TSI_norm_smm,
    VO2_OR_WR_HRR_VE_OR_gender,
    VO2_OR_WR_HRR_VE_OR_TSI_norm_gender,
    VO2_OR_WR_HRR_VE_OR_median_TSI_norm_gender,
    VO2_OR_WR_HRR_VE_OR_median_TSI_norm_smm,
    VO2_OR_WR_HRR_VE_OR_median_TSI_norm_gender_smm,
    VO2_OR_WR_HR_HRR_VE_OR_median, 
]

ALL_SIGNALS = (Processor()
      .add(Process_VO2_OR_median())   # target
      .add(Process_WR())
      .add(Process_HR_HRR())
      .add(Process_VE_OR_median())
      .add(Process_Hb_diff3_norm())
      )
      

# WR + gender + Hb_diff
VO2_OR_WR_gender_Hb_diff_norm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_gender())
    .add(Process_Hb_diff_norm())
)

# WR + gender + TSI
VO2_OR_WR_gender_TSI_norm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_gender())
    .add(Process_TSI_norm())
)

# WR + gender + smm + Hb_diff
VO2_OR_WR_gender_smm_Hb_diff_norm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_gender())
    .add(Process_smm())
    .add(Process_Hb_diff_norm())
)

# WR + gender + smm + TSI
VO2_OR_WR_gender_smm_TSI_norm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_gender())
    .add(Process_smm())
    .add(Process_TSI_norm())
)

# WR + smm + Hb_diff
VO2_OR_WR_smm_Hb_diff_norm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_smm())
    .add(Process_Hb_diff_norm())
)

# WR + smm + TSI
VO2_OR_WR_smm_TSI_norm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_smm())
    .add(Process_TSI_norm())
)      


# WR + gender + Hb_diff_norm + VE
VO2_OR_WR_gender_Hb_diff_norm_VE_OR_median = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_gender())
    .add(Process_Hb_diff_norm())
    .add(Process_VE_OR_median())
)

# WR + gender_smm + Hb_diff_norm + VE
VO2_OR_WR_gender_smm_Hb_diff_norm_VE_OR_median = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_gender())
    .add(Process_smm())
    .add(Process_Hb_diff_norm())
    .add(Process_VE_OR_median())
)

# WR + gender + Hb_diff_norm + HRR
VO2_OR_WR_gender_Hb_diff_norm_HRR = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_gender())
    .add(Process_Hb_diff_norm())
    .add(Process_HRR())
)

# WR + gender_smm + Hb_diff_norm + HRR
VO2_OR_WR_gender_smm_Hb_diff_norm_HRR = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_gender())
    .add(Process_smm())
    .add(Process_Hb_diff_norm())
    .add(Process_HRR())
)

# WR + gender + Hb_diff_norm + VE + HRR
VO2_OR_WR_gender_Hb_diff_norm_HRR_VE_OR_median = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_gender())
    .add(Process_Hb_diff_norm())
    .add(Process_HRR())
    .add(Process_VE_OR_median())
)

# WR + gender_smm + Hb_diff_norm + VE + HRR
VO2_OR_WR_gender_smm_Hb_diff_norm_HRR_VE_OR_median = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_gender())
    .add(Process_smm())
    .add(Process_Hb_diff_norm())
    .add(Process_HRR())
    .add(Process_VE_OR_median())
)

# WR + gender + Hb_diff3_norm
VO2_OR_WR_gender_Hb_diff3_norm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_gender())
    .add(Process_Hb_diff3_norm())
)

# WR + gender + smm + Hb_diff3_norm
VO2_OR_WR_gender_smm_Hb_diff3_norm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_gender())
    .add(Process_smm())
    .add(Process_Hb_diff3_norm())
)

# WR + gender + Hb_diff_norm + TSI_norm
VO2_OR_WR_gender_Hb_diff_norm_TSI_norm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_gender())
    .add(Process_Hb_diff_norm())
    .add(Process_TSI_norm())
)

# WR + gender + smm + Hb_diff_norm + TSI_norm
VO2_OR_WR_gender_smm_Hb_diff_norm_TSI_norm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_gender())
    .add(Process_smm())
    .add(Process_Hb_diff_norm())
    .add(Process_TSI_norm())
)

# WR + VE_OR_median + TSI_norm
VO2_OR_WR_VE_OR_median_TSI_norm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_VE_OR_median())
    .add(Process_TSI_norm())
)

# WR + TSI_norm + Hb_diff_norm
VO2_OR_WR_TSI_norm_Hb_diff_norm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_TSI_norm())
    .add(Process_Hb_diff_norm())
)

# WR + HR + VE_OR_median
VO2_OR_WR_HR_VE_OR_median = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_HR())
    .add(Process_VE_OR_median())
)

# WR + VE_OR_median + TSI_norm + Hb_diff_norm
VO2_OR_WR_VE_OR_median_TSI_norm_Hb_diff_norm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_VE_OR_median())
    .add(Process_TSI_norm())
    .add(Process_Hb_diff_norm())
)


# WR + VE_OR_median + gender
VO2_OR_WR_VE_OR_median_gender = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_VE_OR_median())
    .add(Process_gender())
)

# WR + VE_OR_median + smm
VO2_OR_WR_VE_OR_median_smm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_WR())
    .add(Process_VE_OR_median())
    .add(Process_smm())
)

VO2_OR_Hb_diff_norm_gender = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_gender())
)

VO2_OR_Hb_diff_norm_smm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_smm())
)

VO2_OR_Hb_diff_norm_gender_smm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_gender())
    .add(Process_smm())
)

VO2_OR_Hb_diff_norm_VE_OR_median = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_VE_OR_median())
)

VO2_OR_Hb_diff_norm_VE_OR_median_gender_smm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_VE_OR_median())
    .add(Process_gender())
    .add(Process_smm())
)

VO2_OR_Hb_diff_norm_VE_OR_median_HR_gender_smm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_VE_OR_median())
    .add(Process_HR())
    .add(Process_gender())
    .add(Process_smm())
)

VO2_OR_Hb_diff_norm_HR = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_HR())
)

VO2_OR_Hb_diff_norm_VE_OR_median_HR = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_VE_OR_median())
    .add(Process_HR())
)

# Hb-diff + VE + gender
VO2_OR_Hb_diff_norm_VE_OR_median_gender = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_VE_OR_median())
    .add(Process_gender())
)

# Hb-diff + VE + SMM
VO2_OR_Hb_diff_norm_VE_OR_median_smm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_VE_OR_median())
    .add(Process_smm())
)

# Hb-diff + VE + HRR + gender + SMM
VO2_OR_Hb_diff_norm_VE_OR_median_HRR_gender_smm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_VE_OR_median())
    .add(Process_HRR())
    .add(Process_gender())
    .add(Process_smm())
)

# Hb-diff + VE + HRR
VO2_OR_Hb_diff_norm_VE_OR_median_HRR = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_VE_OR_median())
    .add(Process_HRR())
)

# VE + HRR
VO2_OR_VE_OR_median_HRR = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_VE_OR_median())
    .add(Process_HRR())
)

# Hb-diff + HRR
VO2_OR_Hb_diff_norm_HRR = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_HRR())
)

# VE + HR
VO2_OR_VE_OR_median_HR = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_VE_OR_median())
    .add(Process_HR())
)

# Hb-diff + VE + TSI
VO2_OR_Hb_diff_norm_VE_OR_median_TSI_norm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_VE_OR_median())
    .add(Process_TSI_norm())
)

# Hb-diff + VE + TSI + SMM
VO2_OR_Hb_diff_norm_VE_OR_median_TSI_norm_smm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_VE_OR_median())
    .add(Process_TSI_norm())
    .add(Process_smm())
)

# Hb-diff + VE + TSI + gender
VO2_OR_Hb_diff_norm_VE_OR_median_TSI_norm_gender = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_VE_OR_median())
    .add(Process_TSI_norm())
    .add(Process_gender())
)

# Hb-diff + TSI + SMM
VO2_OR_Hb_diff_norm_TSI_norm_smm = (Processor()
    .add(Process_VO2_OR_median())
    .add(Process_Hb_diff_norm())
    .add(Process_TSI_norm())
    .add(Process_smm())
)


# region VO2 norm

# VO2_OR_norm_WR = (Processor()
#         .add(Process_VO2_OR_median_norm())
#         .add(Process_WR())
#         )
#
# VO2_OR_norm_HR = (Processor()
#         .add(Process_VO2_OR_median_norm())
#         .add(Process_HR())
#         )
#
# VO2_OR_norm_HR_HRR = (Processor()
#         .add(Process_VO2_OR_median_norm())
#         .add(Process_HR_HRR())
#         )
#
# VO2_OR_norm_VE_OR = (Processor()
#       .add(Process_VO2_OR_median_norm())
#       .add(Process_VE_OR())
#       )
#
# VO2_OR_norm_VE_measured_OR = (Processor()
#       .add(Process_VO2_OR_median_norm())
#       .add(Process_WR())
#       .add(Process_VE_measured_OR())
#       )
#
# VO2_OR_norm_TSI_norm = (Processor()
#       .add(Process_VO2_OR_median_norm())
#       .add(Process_TSI_norm())
#       )
#
# VO2_OR_norm_WR_HR_HRR = (Processor()
#         .add(Process_VO2_OR_median_norm())
#         .add(Process_WR())
#         .add(Process_HR_HRR())
#         )
#
# VO2_OR_norm_WR_VE_OR = (Processor()
#       .add(Process_VO2_OR_median_norm())
#       .add(Process_WR())
#       .add(Process_VE_OR())
#       )
#
# VO2_OR_norm_WR_VE_measured_OR = (Processor()
#       .add(Process_VO2_OR_median_norm())
#       .add(Process_WR())
#       .add(Process_VE_measured_OR())
#       )
#
# VO2_OR_norm_WR_TSI_norm = (Processor()
#       .add(Process_VO2_OR_median_norm())
#       .add(Process_WR())
#       .add(Process_TSI_norm())
#       )
#
# VO2_OR_norm_WR_gender_smm = (Processor()
#       .add(Process_VO2_OR_median_norm())
#       .add(Process_WR())
#       .add(Process_gender_smm())
#       )
#
# single_input_processors_VO2_norm = [
#     VO2_OR_norm_WR,
#     VO2_OR_norm_HR,
#     VO2_OR_norm_HR_HRR,
#     VO2_OR_norm_VE_OR,
#     VO2_OR_norm_VE_measured_OR,
#     VO2_OR_norm_TSI_norm,
# ]
#
# single_input_processors_with_WR_VO2_norm = [
#     VO2_OR_norm_WR_HR_HRR,
#     VO2_OR_norm_WR_TSI_norm,
#     VO2_OR_norm_WR_VE_OR,
#     VO2_OR_norm_WR_VE_measured_OR,
#     VO2_OR_norm_WR_gender_smm
# ]

# endregion

