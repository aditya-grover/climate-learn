NAME_TO_VAR = {
    "geopotential": "zg",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "ta",
    "specific_humidity": "hus",
    "air_temperature": "tas",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

SINGLE_LEVEL_VARS = [
    "air_temperature",
]

PRESSURE_LEVEL_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
]

VAR_TO_UNIT = {
    "air_temperature": "C",
    "geopotential": "m^2/s^2",
    "u_component_of_wind": "m/s",
    "v_component_of_wind": "m/s",
    "temperature": "C",
    "specific_humidity": "kg/kg",
}

DEFAULT_PRESSURE_LEVELS = [50, 250, 500, 600, 700, 850, 925]

CONSTANTS = []

NAME_LEVEL_TO_VAR_LEVEL = {}

for var in SINGLE_LEVEL_VARS:
    NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

for var in PRESSURE_LEVEL_VARS:
    for l in DEFAULT_PRESSURE_LEVELS:
        NAME_LEVEL_TO_VAR_LEVEL[var + "_" + str(l)] = NAME_TO_VAR[var] + "_" + str(l)

VAR_LEVEL_TO_NAME_LEVEL = {v: k for k, v in NAME_LEVEL_TO_VAR_LEVEL.items()}
