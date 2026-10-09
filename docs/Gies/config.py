import numpy as np
from data_types import *

# Should include for all participants and visits
# VE_lag is the number of seconds to shift the VE file to match the VO2 file
# HR_source is the source of the HR data, either 'TW', 'VO2' or None
# Occlusion: None or { start, end, notes? } in NIRS datapoints from the raw file

config: Config = {
    'P1': {
        'Ramp': {
            'VE_lag': 60,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 26980,
                'end': 31290,
            },
            'NIRS_repetition': 'Both', # End of TSI is not soo nice
        },
        'Easy': {
            'VE_lag': 60,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 25340,
                'end': 29240,
            },
            'NIRS_repetition': 'Both',
        },
        'Medium': {
            'VE_lag': 59,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 23480,
                'end': 27570,
            },
            'NIRS_repetition': 'Both',
        },
        'Hard': {
            'VE_lag': 60,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 26830,
                'end': 31430,
            },
            'NIRS_repetition': 'Both',
        },
    },
    'P3': {
        'Ramp': {
            'VE_lag': 27,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 13860,
                'end': 18790,
            },
            'NIRS_repetition': 'Both',
        },
        'Easy': {
            'VE_lag': 29,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 27360,
                'end': 31000,
            },
            'NIRS_repetition': 'Both',
        },
        'Medium': {
            'VE_lag': 37,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 27970,
                'end': 31830,
            },
            'NIRS_repetition': 'Both',
        },
        'Hard': {
            'VE_lag': 30,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 28730,
                'end': 32760,
            },
            'NIRS_repetition': 'Both',
        },
    },
    'P4': {
        'Ramp': {
            'VE_lag': 29,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 16990,
                'end': 23230,
            },
            'NIRS_repetition': 'Both',
        },
        'Easy': {
            'VE_lag': 38,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 25670,
                'end': 29770,
            },
            'NIRS_repetition': 'Both',
        },
        'Medium': {
            'VE_lag': 29,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 24560,
                'end': 28500,
            },
            'NIRS_repetition': 'Both',
        },
        'Hard': {
            'VE_lag': 30,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 29030,
                'end': 33510,
            },
            'NIRS_repetition': 'Both',
        },
    },
    'P5': {
        'Ramp': {
            'VE_lag': 28,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 15830,
                'end': 20800,
            },
            'NIRS_repetition': 'Both',
        },
        'Easy': {
            'VE_lag': 30,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 25110,
                'end': 29110,
            },
            'NIRS_repetition': 'Both',
        },
        'Medium': {
            'VE_lag': 30,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 27070,
                'end': 31170,
            },
            'NIRS_repetition': 'Both',
        },
        'Hard': {
            'VE_lag': 30,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 25790,
                'end': 29770,
            },
            'NIRS_repetition': 'Both',
        },
    },
    'P6': {
        'Ramp': {
            'VE_lag': np.nan, # Missing
            'HR_source': 'VO2',
            'Occlusion': {
                'start': 20280,
                'end': 22360,
            },
            'NIRS_repetition': 'Both',
        },
        'Easy': {
            'VE_lag': 694,
            'HR_source': 'VO2',
            'Occlusion': {
                'start': 32800,
                'end': 37160,
                'notes': 'No clear max.'
            },
            'NIRS_repetition': 'Both',
        },
        'Medium': {
            'VE_lag': 379,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 26030,
                'end': 30200,
            },
            'NIRS_repetition': 'Both',
        },
        'Hard': {
            'VE_lag': 448,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 27190,
                'end': 31530,
            },
            'NIRS_repetition': 'Both',
        },
    },
    'P7': {
        'Ramp': {
            'VE_lag': np.nan, # Missing
            'HR_source': None,
            'Occlusion': None,
        },
        'Easy': {
            'VE_lag': 29,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 36790,
                'end': 40730,
            },
            'NIRS_repetition': 'Both',
        },
        'Medium': {
            'VE_lag': 631,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 25880,
                'end': 28880,
            },
            'notes': 'Fit factor drops during occlusion and only Tx2 seems representative',
            'NIRS_repetition': None,
        },
        'Hard': {
            'VE_lag': 440,
            'HR_source': None,
            'Occlusion': {
                'start': 30140,
                'end': 34200,
            },
            'NIRS_repetition': 'One',
        },
    },
    'P8': {
        'Ramp': {
            'VE_lag': 1042,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 16840,
                'end': 23000,
            },
            'NIRS_repetition': 'Both',
        },
        'Easy': {
            'VE_lag': 557,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 30810,
                'end': 34530,

            },
            'notes': 'High blood volume shifts. Max not clear.',
            'NIRS_repetition': None,
        },
        'Medium': {
            'VE_lag': 2423,
            'HR_source': 'TW',
            'Occlusion': None,
            'NIRS_repetition': None,

        },
        'Medium2': {
            'VE_lag': 498,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 28780,
                'end': 32700,
            },
            'NIRS_repetition': 'Both',
        },
        'Hard': {
            'VE_lag': 1727,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 35670,
                'end': 39840,
            },
            'NIRS_repetition': 'Both',
        },
    },
    'P9': {
        'Ramp': {
            'VE_lag': np.nan, # Missing
            'HR_source': 'VO2',
            'Occlusion': {
                'start': 16490,
                'end': 18400,
            },
            'notes': 'Weird blood volume changes towards end of test.',
            'NIRS_repetition': 'Both',
        },
        'Easy': {
            'VE_lag': 245,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 25120,
                'end': 29160,
            },
            'notes': 'Blood volume changes during 1st rep, fit factor not ideal during 2nd rep.',
            'NIRS_repetition': 'Both',
        },
        'Medium': {
            'VE_lag': 52,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 25860,
                'end': 29680,
            },
            'notes': 'Blood volume changes during 1st rep.',
            'NIRS_repetition': 'Both',
        },
        'Hard': {
            'VE_lag': 51,
            'HR_source': 'TW',
            'Occlusion': None,
            'NIRS_repetition': None,
        },
        'Hard2': {
            'VE_lag': 719,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 27060,
                'end': 31510,
            },
            'notes': 'Blood volume changes during 1st rep.',
            'NIRS_repetition': 'Both',
        }
    },
    'P10': {
        'Ramp': {
            'VE_lag': 510,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 13580,
                'end': 18590,
            },
            'NIRS_repetition': 'Both',
        },
        'Easy': {
            'VE_lag': 29,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 25110,
                'end': 29090,
            },
            'NIRS_repetition': 'Both',
        },
        'Medium': {
            'VE_lag': 30,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 26680,
                'end': 30540,
            },
            'NIRS_repetition': 'Both',
        },
        'Hard': {
            'VE_lag': 30,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 28910,
                'end': 32700,
            },
            'notes': 'Weird Artifact at start of 2nd rep',
            'NIRS_repetition': 'Both',
        },
    },
    'P11': {
        'Ramp': {
            'VE_lag': np.nan, # Missing
            'HR_source': None,
            'Occlusion': {
                'start': 18340,
                'end': 19350,
            },
            'notes': 'Too short, TSI not reflective, high peaks in data from standing up too much',
            'NIRS_repetition': None,
        },
        'Easy': {
            'VE_lag': 85,
            'HR_source': 'VO2',
            'Occlusion': {
                'start': 24680,
                'end': 27540,
            },
            'notes': 'Occlusion is wrong for TSI, channels look fine',
            'NIRS_repetition': None,
        },
        'Medium': {
            'VE_lag': 425,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 26600,
                'end': 30500,
            },
            'NIRS_repetition': 'Both',
        },
        'Hard': {
            'VE_lag': 520,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 27990,
                'end': 32110,
            },
            'NIRS_repetition': 'One',
        },
    },
    'P12': {
        'Ramp': {
            'VE_lag': 405,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 14400,
                'end': 19670,
            },
            'NIRS_repetition': 'Both',
        },
        'Easy': {
            'VE_lag': 355,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 26140,
                'end': 29820,
            },
            'NIRS_repetition': 'Both',
        },
        'Medium': {
            'VE_lag': 456,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 25670,
                'end': 29530,
            },
            'NIRS_repetition': 'Both',
        },
        'Hard': {
            'VE_lag': 30,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 28230,
                'end': 32410,
            },
            'NIRS_repetition': 'Both',
        },
    },
    'P13': {
        'FailedRamp': {
            'VE_lag': np.nan, # Missing
            'HR_source': None,
            'Occlusion': None,
        },
        'Ramp': {
            'VE_lag': 160,
            'HR_source': None,
            'Occlusion': {
                'start': 16670,
                'end': 20600,
            },
            'NIRS_repetition': 'Both',
        },
        'Easy': {
            'VE_lag': 44,
            'HR_source': None,
            'Occlusion': {
                'start': 24390,
                'end': 28770,
            },
            'NIRS_repetition': 'Both',
        },
        'Medium': {
            'VE_lag': 287,
            'HR_source': 'VO2',
            'Occlusion': {
                'start': 27370,
                'end': 31900,
            },
            'NIRS_repetition': 'Both',
        },
        'Hard': {
            'VE_lag': 361,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 25100,
                'end': 29040,
                'notes': 'Max not quite maximal.'
            },
            'NIRS_repetition': 'Both',
        },
    },
    'P14': {
        'Ramp': {
            'VE_lag': 30,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 18120,
                'end': 23280,
            },
            'notes': 'Weird artifact at end of ramp',
            'NIRS_repetition': 'Both',
        },
        'Easy': {
            'VE_lag': 30,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 24700,
                'end': 28780,
            },
            'NIRS_repetition': 'Both',
        },
        'Hard': {
            'VE_lag': 29,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 29900,
                'end': 33900,
            },
            'NIRS_repetition': 'Both',
        },
    },
    'P15': {
        'Ramp': {
            'VE_lag': 3844,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 43160,
                'end': 47470,
            },
            'notes': 'Fit factor drops towards end of test, which yields a weird signal, but is fine during occlusion',
            'NIRS_repetition': None,
        },
        'Easy': {
            'VE_lag': 565,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 27390,
                'end': 31670,
            },
            'NIRS_repetition': 'Both',
        },
        'Medium': {
            'VE_lag': 498,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 28340,
                'end': 32400,
            },
            'NIRS_repetition': 'Both',
        },
        'Hard': {
            'VE_lag': 30,
            'HR_source': 'TW',
            'Occlusion': {
                'start': 37410,
                'end': 41510,
            },
            'NIRS_repetition': 'Both',
        },
    },
}
