from ..data_types.base import Participant, Visit
from ..data_types.visit_config import Config, NIRSRepetition, HRSource
from ..data_types.visit_config import NIRSRepetition, HRSource

config_lucas: Config = {

    'P01': {
        "Ramp": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 13079,
                'end': 15000
            },
            'NIRS_repetition': NIRSRepetition.Both,
            'ramp_rate': 30,
        },
        "Interval": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23699,
                'end': 26069
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "Step": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23956,
                'end': 26151
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "PRBS": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 25198,
                'end': 27493
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
    },

    'P02': {
        "Ramp": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 13972,
                'end': 19336
            },
            'NIRS_repetition': NIRSRepetition.Both,
            'ramp_rate': 20,
        },
        "Interval": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 24370,
                'end': 26191
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "Step": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23736,
                'end': 25448
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "PRBS": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 24066,
                'end': 25924
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
    },

    'P03': {
        "Ramp": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 12627,
                'end': 17613
            },
            'NIRS_repetition': NIRSRepetition.Both,
            'ramp_rate': 25,
        },
        "Interval": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23410,
                'end': 27681
            },
            'NIRS_repetition': NIRSRepetition.Both,
        },
        "Step": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23444,
                'end': 27433
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "PRBS": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23630,
                'end': 28071
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
    },

    'P04': {
        "Ramp": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 11496,
                'end': 13568
            },
            'NIRS_repetition': NIRSRepetition.Both,
            'ramp_rate': 20,
        },
        "Interval": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23389,
                'end': 27176
            },
            'NIRS_repetition': NIRSRepetition.Both,
        },
        "Step": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23407,
                'end': 25912
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "PRBS": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23879,
                'end': 27605
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
    },

    'P05': {
        "Ramp": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 12344,
                'end': 16209
            },
            'NIRS_repetition': NIRSRepetition.Both,
            'ramp_rate': 30,
        },
        "Interval": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23934,
                'end': 26718
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "Step": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23518,
                'end': 26286
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "PRBS": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23441,
                'end': 25864
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
    },

    'P06': {
        "Ramp": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 10162,
                'end': 13564
            },
            'NIRS_repetition': NIRSRepetition.Both,
            'ramp_rate': 20,
        },
        "Interval": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23707,
                'end': 27082
            },
            'NIRS_repetition': NIRSRepetition.Both,
        },
        "Step": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23577,
                'end': 27260
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "PRBS": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23872,
                'end': 26886
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
    },

    'P07': {
        "Ramp": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 11305,
                'end': 14763
            },
            'NIRS_repetition': NIRSRepetition.Both,
            'ramp_rate': 25,
        },
        "Interval": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23463,
                'end': 27524
            },
            'NIRS_repetition': NIRSRepetition.Both,
        },
        "Step": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23879,
                'end': 26740
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "PRBS": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23164,
                'end': 26870
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
    },

    'P08': {
        "Ramp": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 13890,
                'end': 16180
            },
            'NIRS_repetition': NIRSRepetition.Both,
            'ramp_rate': 30,
        },
        "Interval": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 25499,
                'end': 28865
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "Step": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23508,
                'end': 27354
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "PRBS": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23543,
                'end': 26929
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
    },

    'P09': {
        "Ramp": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 11446,
                'end': 14344
            },
            'NIRS_repetition': NIRSRepetition.Both,
            'ramp_rate': 20,
        },
        "Interval": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23974,
                'end': 27730
            },
            'NIRS_repetition': NIRSRepetition.Both,
        },
        "Step": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23467,
                'end': 26835
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "PRBS": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 24318,
                'end': 27420
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
    },

    'P10': {
        "Ramp": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 10810,
                'end': 13556
            },
            'NIRS_repetition': NIRSRepetition.Both,
            'ramp_rate': 25,
        },
        "Interval": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23404,
                'end': 27897
            },
            'NIRS_repetition': NIRSRepetition.Both,
        },
        "Step": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23500,
                'end': 27414
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "PRBS": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23917,
                'end': 26738
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
    },

    'P14': {
        "Ramp": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 26108,
                'end': 29739
            },
            'NIRS_repetition': NIRSRepetition.Both,
            'ramp_rate': 30,
        },
        "Interval": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23404,
                'end': 27897
            },
            'NIRS_repetition': NIRSRepetition.Both,
        },
        "Step": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 24038,
                'end': 27917
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "PRBS": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 25475,
                'end': 29282
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
    },

    'P12': {
        "Ramp": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 13941,
                'end': 18859
            },
            'NIRS_repetition': NIRSRepetition.Both,
            'ramp_rate': 30,
        },
        "Interval": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23404,
                'end': 27897
            },
            'NIRS_repetition': NIRSRepetition.Both,
        },
        "Step": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23656,
                'end': 28133
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "PRBS": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23917,
                'end': 26738
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
    },

    'P13': {
        "Ramp": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 12921,
                'end': 16484
            },
            'NIRS_repetition': NIRSRepetition.Both,
            'ramp_rate': 25,
        },
        "Interval": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23404,
                'end': 27897
            },
            'NIRS_repetition': NIRSRepetition.Both,
        },
        "Step": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23459,
                'end': 26975
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
        "PRBS": {
            'HR_source': HRSource.HR,
            'Occlusion': {
                'start': 23762,
                'end': 27444
            },
            'NIRS_repetition': NIRSRepetition.Both
        },
    },
}