# ARTIFACT RELATED CONSTANTS
PIPELINE_DIRECTORY = "pipelines"
MODEL_DIRECTORY = "models"
PIPELINE_NAME = "pipeline"
MODEL_NAME = "model"
PREDICTION_DIRECTORY = "predictions"
PREDICTION_NAME = "prediction"

# DATA RELATED CONSTANTS
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
