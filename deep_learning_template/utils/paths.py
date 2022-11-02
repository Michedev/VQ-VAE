from path import Path

CODE_ROOT: Path = Path(__file__).parent.parent
CODE_MODEL: Path = CODE_ROOT / 'model'
ROOT: Path = CODE_ROOT.parent
CONFIG: Path = CODE_ROOT / 'config'
DATA: Path = ROOT / 'data'
CONFIG_DATA: Path = CONFIG / 'dataset'
CONFIG_MODEL: Path = CONFIG / 'model'
CONFIG_MODEL_DATASET: Path = CONFIG / 'model_dataset'

SAVED_MODELS: Path = ROOT / 'saved_models'
