# Reproducibility
SEED = 2022

# ARTIFACT RELATED CONSTANTS
PIPELINE_DIRECTORY = "pipelines"
MODEL_DIRECTORY = "models"
PIPELINE_NAME = "pipeline"
MODEL_NAME = "model"
PREDICTION_DIRECTORY = "predictions"
PREDICTION_NAME = "prediction"

# DATA RELATED CONSTANTS
TEXT_COLUMNS = [
    "description",
    "administrative_status",
    "marketing_status",
    "dosage_form",
    "marketing_authorization_status",
    "marketing_authorization_process",
    "pharmaceutical_companies",
]
DATES_COLUMNS = ["marketing_declaration_date", "marketing_authorization_date"]
DESCRIPTION_COLUMN = "description"

LABEL_PLAQUETTE = "plaquette"
LABEL_STYLO = ("stylo",)
LABEL_TUBE = "tube"
LABEL_SERINGUE = "seringue"
LABEL_CACHET = "cachet"
LABEL_GELULE = "gelule"
LABEL_FALCON = "flacon"
LABEL_AMPOULE = "ampoule"
LABEL_ML = "ml"
LABEL_G = "g"
LABEL_PILULIER = "pilulier"
LABEL_COMPRIME = "comprime"
LABEL_FILM = "film"
LABEL_POCHE = "poche"
LABEL_CAPSULE = "capsule"

ALL_LABELS = [
    LABEL_PLAQUETTE,
    LABEL_STYLO,
    LABEL_TUBE,
    LABEL_SERINGUE,
    LABEL_CACHET,
    LABEL_GELULE,
    LABEL_FALCON,
    LABEL_AMPOULE,
    LABEL_ML,
    LABEL_G,
    LABEL_PILULIER,
    LABEL_FILM,
    LABEL_POCHE,
    LABEL_CAPSULE,
]

HIGH_CARD_COLUMNS = [
    "dosage_form",
    "route_of_administration",
    "pharmaceutical_companies",
]
ONE_HOT_COLUMNS = [
    "marketing_status",
    "marketing_authorization_status",
    "marketing_authorization_process",
]
STRS_TO_CHECK = ["active", "oui"]
PRICE = "price"
DRUG_ID = "drug_id"

ACTIVE_INGREDIENT = "active_ingredient"
REIMBURSEMENT_RATE = "reimbursement_rate"

MARKETING_STATUS_MAP = {
    "declaration de commercialisation": "dec_com",
    "declaration darret de commercialisation": "dec_arret_comm",
    "arret de commercialisation le medicament na plus dautorisation": "arret_com_med",
    "declaration de suspension de commercialisation": "dec_sus_com",
}

DROP_COLUMNS = [
    "drug_id",
    "description",
    "administrative_status",
    "marketing_status",
    "approved_for_hospital_use",
    "reimbursement_rate",
    "dosage_form",
    "route_of_administration",
    "marketing_authorization_status",
    "marketing_declaration_date",
    "marketing_authorization_date",
    "marketing_authorization_process",
    "pharmaceutical_companies",
    "price",
]

PHARMACY = ["pharmaceutical_companies"]
