import pathlib

MODULE_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR = MODULE_DIR.parent.parent.resolve()
DATASET_DIR = PROJECT_DIR / 'dataset'
