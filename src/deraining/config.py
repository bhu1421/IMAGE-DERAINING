from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "Rain100L"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TRAIN_RAIN_DIR = DATASET_ROOT / "rain_data_train_Light" / "rain"
TRAIN_CLEAN_DIR = DATASET_ROOT / "rain_data_train_Light" / "norain"
TEST_RAIN_DIR = DATASET_ROOT / "rain_data_test_Light" / "rain" / "X2"
TEST_CLEAN_DIR = DATASET_ROOT / "rain_data_test_Light" / "norain"

GENERATOR_CHECKPOINT = OUTPUTS_DIR / "pix2pix_generator.pth"
DISCRIMINATOR_CHECKPOINT = OUTPUTS_DIR / "pix2pix_discriminator.pth"

APP_RESULT_IMAGE = OUTPUTS_DIR / "app_result.png"
APP_COMPARISON_IMAGE = OUTPUTS_DIR / "app_comparison.png"
TEST_RESULTS_DIR = OUTPUTS_DIR / "test_results"

IMAGE_SIZE = (256, 256)
