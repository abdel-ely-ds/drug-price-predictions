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
]
DATES_COLUMNS = ["marketing_declaration_date", "marketing_authorization_date"]
DESCRIPTION_COLUMN = "description"

ALL_LABELS = [
    "plaquette",
    "stylo",
    "tube",
    "seringue",
    "cachet",
    "gelule",
    "flacon",
    "ampoule",
    "ml",
    "g",
    "pilulier",
    "comprime",
    "film",
    "poche",
    "capsule",
]

HIGH_CARD_COLUMNS = [
    "dosage_form",
    "route_of_administration",
    "pharmaceutical_companies",
    "year",
    "active_ingredient",
]
ONE_HOT_COLUMNS = [
    "marketing_status",
    "marketing_authorization_status",
    "marketing_authorization_process",
]

BINARY_COLUMNS = ["administrative_status", "approved_for_hospital_use"]

STRS_TO_CHECK = ["active", "oui"]
PRICE = "price"
DRUG_ID = "drug_id"

REIMBURSEMENT_RATE = "reimbursement_rate"

PHARMACY_COLUMN = "pharmaceutical_companies"
YEAR = "year"

SELECTED_FEATURES = [
    "reimbursement_rate_feature",
    "marketing_authorization_process4_feature",
    "dosage_form_feature",
    "route_of_administration_feature",
    "pharmaceutical_companies_feature",
    "year_feature",
    "active_ingredient_feature",
]
