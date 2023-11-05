from src.config.downsample import * # noqa: F401, F403

OVERFIT_BATCHES = 5 # 10 #0.1
MAX_EPOCHS: int = 400
ACCUMULATIONS: int = 5
SPLITS = [0.5, 0.5, 0.0]
DEVICES: list | int = [2, 3]
