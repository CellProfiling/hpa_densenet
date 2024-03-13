"""
Constants for the HPA Densenet module.
This module is not intended for use by other modules.
"""
LOGGER_NAME = "HPADenseNet"
NUM_PREDICTOR_CLASSES = 28
NUM_IN_CHANNELS = 4
DEFAULT_MODEL = "models/bestfitting_default_model.pth"
CROP_SIZE = 1024
SEEDS = [0]
CLASS2NAME = {
    0: "Nucleoplasm",
    1: "Nuclear membrane",
    2: "Nucleoli",
    3: "Nucleoli fibrillar center",
    4: "Nuclear speckles",
    5: "Nuclear bodies",
    6: "Endoplasmic reticulum",
    7: "Golgi apparatus",
    8: "Peroxisomes",
    9: "Endosomes",
    10: "Lysosomes",
    11: "Intermediate filaments",
    12: "Actin filaments",
    13: "Focal adhesion sites",
    14: "Microtubules",
    15: "Microtubule ends",
    16: "Cytokinetic bridge",
    17: "Mitotic spindle",
    18: "Microtubule organizing center",
    19: "Centrosome",
    20: "Lipid droplets",
    21: "Plasma membrane",
    22: "Cell junctions",
    23: "Mitochondria",
    24: "Aggresome",
    25: "Cytosol",
    26: "Cytoplasmic bodies",
    27: "Rods & rings",
}
CLASS2COLOR = {
    0: '#f44336',
    1: '#e91e63',
    2: '#9c27b0',
    3: '#673ab7',
    4: '#3f51b5',
    5: '#2196f3',
    6: '#03a9f4',
    7: '#00bcd4',
    8: '#009688',
    9: '#4caf50',
    10: '#8bc34a',
    11: '#cddc39',
    12: '#ffeb3b',
    13: '#ffc107',
    14: '#ff9800',
    15: '#ff5722',
    16: '#795548',
    17: '#9e9e9e',
    18: '#607d8b',
    19: '#dddddd',
    20: '#212121',
    21: '#ff9e80',
    22: '#ff6d00',
    23: '#ffff00',
    24: '#76ff03',
    25: '#00e676',
    26: "#64ffda",
    27: "#18ffff",
}
