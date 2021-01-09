from starlette.config import Config

config = Config(".env")

# LOGGING #
LOGGING_LEVEL: int = config(
    "LOGGING_LEVEL",
    default=20
)

# MT5_TRANSLATOR CONFIGURATION #
## Model path
DEFAULT_MODEL_PATH: str = config(
    "DEFAULT_MODEL_PATH",
    default='trainer/outputs/best_model/')
## Model architecture
MODEL_ARCHITECTURE: str = config(
    "MODEL_ARCHITECTURE",
    default="mt5"
)
## GPU usage
GPU: bool = config("GPU", default=True)
## Batch size of the model
BATCH_SIZE: int = config("BATCH_size", default=8)