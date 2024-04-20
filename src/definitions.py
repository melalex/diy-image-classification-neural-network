from pathlib import Path


PROJECT_ROOT_DIR = Path(__file__).parent.parent.absolute()

LOGGING_CONFIG_PATH = PROJECT_ROOT_DIR / "logging.ini"

RAW_DATA_FOLDER = PROJECT_ROOT_DIR / "data" / "raw"

PROCESSED_DATA_FOLDER = PROJECT_ROOT_DIR / "data" / "processed"
TRAIN_DATA_FOLDER = PROCESSED_DATA_FOLDER / "train"
TEST_DATA_FOLDER = PROCESSED_DATA_FOLDER / "test"
GRAD_CHECK_FOLDER = PROCESSED_DATA_FOLDER / "grad-check"

VALID_DATASET_FLAG_FILE = PROCESSED_DATA_FOLDER / ".valid"

CATS_VS_DOGS_DATASET_OWNER = "shaunthesheep"
CATS_VS_DOGS_DATASET_NAME = "microsoft-catsvsdogs-dataset"
SAMPLE_DATASET_OWNER = "pankajkumar2002"
SAMPLE_DATASET_NAME = "random-image-sample-dataset"
CATS_DATASET_CONTENT_PATH = "PetImages/Cat"
DOGS_DATASET_CONTENT_PATH = "PetImages/Dog"
SAMPLE_DATASET_CONTENT_PATH = "data"

LOG_PERIOD = 100
IMAGE_WIDTH = 64
IMAGE_HEIGHT = IMAGE_WIDTH
SAMPLE_IMAGES_RATIO = 1
TEST_TRAIN_RATIO = 0.8
LEARNING_RATE = 0.005
ITERATION_COUNT = 1000
GRAD_CHECK_SAMPLES_COUNT = 128

MODELS_FOLDER = PROJECT_ROOT_DIR / "models"
MODEL_PATH = (
    MODELS_FOLDER
    / f"animal-classification-{IMAGE_WIDTH}x{IMAGE_HEIGHT}-{LEARNING_RATE}-{ITERATION_COUNT}.pickle"
)
