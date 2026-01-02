from dataclasses import dataclass
from typing import List


@dataclass
class RNAConfig:
    """Global configuration for the RNAcompete Data Pipeline."""
    
    # Data Path
    DATA_PATH: str = "norm_data.txt" #NOTE: Only change this if you want to use a different path
    
    # Metadata is an Excel file
    METADATA_PATH: str = "metadata.xlsx" #NOTE: Only change this if you want to use a different path
    METADATA_SHEET: str = "Master List--Plasmid Info"
    
    # Save Path
    SAVE_DIR: str = "data" #NOTE: Only change this if you want to use a different path
    
    # Sequence Parameters
    SEQ_MAX_LEN: int = 41
    ALPHABET: str = "ACGUN"
    
    # Preprocessing
    CLIP_PERCENTILE: float = 99.95
    EPSILON: float = 1e-6  # For numerical stability
    
    # Split Identifiers
    TRAIN_SPLIT_ID: str = "SetA"
    TEST_SPLIT_ID: str = "SetB"
    
    VAL_SPLIT_PCT: float = 0.2
    SEED: int = 42 # NOTE: Change this only if you want to test reproducibility

    # Fraction of the full dataset used in preliminary experiments to reduce training time
    DATA_FRACTION = 0.2  # 20%

@dataclass
class RNNHyperparamSpace:
    hidden_size: List[int] = (64, 128, 256, 512)
    batch_size: List[int] = (32, 64, 128)
    lr_min: float = 1e-4
    lr_max: float = 1e-2
    dropout_min: float = 0.0
    dropout_max: float = 0.5
    bidirectional_options: List[bool] = (True, False)
    num_epochs: int = 30


@dataclass
class CNNHyperparamSpace:
    kernel_size: List[int] = (3, 5, 7)
    batch_size: List[int] = (32, 64, 128)
    lr_min: float = 1e-4
    lr_max: float = 1e-2
    dropout_min: float = 0.0
    dropout_max: float = 0.5
    conv_params: List[list] = (
        [8, 16],
        [16, 32],
        [16, 32, 64],
        [32, 64, 128]
    )
    fc_params: List[list] = (
        [32],
        [64],
        [128],
        [64, 32],
        [128, 64]
    )
    no_maxpool: List[bool] = (True, False)
    num_epochs: int = 30
