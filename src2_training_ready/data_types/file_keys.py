from enum import StrEnum

class VO2Keys(StrEnum):
    TIME = 't'
    WR = 'Power'
    VO2 = 'VO2'
    VO2kg = 'VO2/kg'
    VCO2 = 'VCO2'
    VE = 'VE'
    VEVO2 = 'VEVO2'
    VEVCO2 = 'VEVCO2'
    PetO2 = 'PetO2'
    PetCO2 = 'PetCO2'
    Rf = 'Rf'
    HR = 'HR'
    Phase = 'Phase'

class FitFileKeys(StrEnum):
    weight = 'weight'
    height = 'height'
    age = 'age'
    muscle_mass = 'muscle mass'
    body_fat = 'body fat'
    gender = 'gender'

class NIRSFileKeys(StrEnum):
    O2Hb1 = 'O2Hb1'
    HHb1 = 'HHb1'
    tHb1 = 'tHb1'
    O2Hb2 = 'O2Hb2'
    HHb2 = 'HHb2'
    tHb2 = 'tHb2'
    O2Hb3 = 'O2Hb3'
    HHb3 = 'HHb3'
    tHb3 = 'tHb3'
    TSI = 'TSI%'
    fit_factor = 'TSI Fit Factor'
    Event = 'Event'
    Event_time = 'Event Time'

class VEBxBFileKeys(StrEnum):
    VEBxB = 'VE breath by breath'
    VTBxB = 'VT breath by breath'
    BRBxB = 'BR breath by breath'
    VEBxB_smoothed = 'VE breath by breath smoothed'
    VTBxB_smoothed = 'VT breath by breath smoothed'
    BRBxB_smoothed = 'BR breath by breath smoothed'


class VEFileKeys(StrEnum):
    VE = 'VE'
    VT = 'VT'
    BR = 'BR'
    VE_calibrated = 'VE calibrated'
    VT_calibrated = 'VT calibrated'
    BR_calibrated = 'BR calibrated'
    VE_calibrated_inst = 'VE calibrated inst'
    VT_calibrated_inst = 'VT calibrated inst'
    BR_calibrated_inst = 'BR calibrated inst'

