"""
Configuration for MLP fault diagnosis pipeline.
"""
from pathlib import Path

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "partitioned_dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = BASE_DIR / "mlp_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sensor directories
SENSOR_DIRS = {
    "vibration": "vibration",
    "acoustic": "acoustic",
    "current_temp": "current,temp",
}

# ============================================================
# Signal parameters (derived from data inspection)
# ============================================================
SAMPLE_RATES = {
    "vibration": 25600,    # Hz
    "acoustic": 51200,     # Hz
    "current_temp": 25600, # Hz (approximately)
}

CHANNEL_NAMES = {
    "vibration": ["x_housing_A", "y_housing_A", "x_housing_B", "y_housing_B"],
    "acoustic": ["pressure"],
    "current_temp": ["temp_A", "temp_B", "U_phase"],
}

# Units (for reference)
CHANNEL_UNITS = {
    "vibration": "g",
    "acoustic": "Pa",
    "current_temp": ["°C", "°C", "A", "A", "A"],
}

# ============================================================
# Segmentation parameters
# ============================================================
WINDOW_SECONDS = 1.0          # Window length in seconds
OVERLAP_RATIO = 0.5           # 50% overlap
MIN_WINDOWS_PER_FILE = 1      # Minimum windows to extract from a file

# Derived window sizes (samples)
WINDOW_SIZES = {
    sensor: int(WINDOW_SECONDS * SAMPLE_RATES[sensor])
    for sensor in SAMPLE_RATES
}
STEP_SIZES = {
    sensor: int(WINDOW_SIZES[sensor] * (1 - OVERLAP_RATIO))
    for sensor in SAMPLE_RATES
}

# ============================================================
# Feature extraction parameters
# ============================================================
# Wavelet packet decomposition
WAVELET = "db4"
WP_MAX_LEVEL = 3  # Gives 2^3 = 8 sub-bands

# Frequency bands for band-energy ratios (Hz)
FREQ_BANDS = {
    "vibration": [(0, 1000), (1000, 3000), (3000, 6000), (6000, 12800)],
    "acoustic": [(0, 2000), (2000, 6000), (6000, 12000), (12000, 25600)],
    "current_temp": [(0, 100), (100, 500), (500, 2000), (2000, 12800)],
}

# Bearing fault characteristic frequency bands (Hz)
# Approximate ranges for common motor bearings at ~1200-1800 RPM
# These are wide bands to capture harmonics
BEARING_FAULT_BANDS = {
    "BPFI": [(100, 150), (200, 300), (400, 600), (800, 1200)],  # Inner race
    "BPFO": [(60, 100), (120, 200), (240, 400), (480, 800)],    # Outer race
    "BSF":  [(40, 70), (80, 140), (160, 280), (320, 560)],       # Ball spin
    "FTF":  [(10, 20), (20, 40), (40, 80)],                       # Cage frequency
}

# ============================================================
# Preprocessing
# ============================================================
SCALER_TYPE = "standard"  # "standard" or "robust"

# ============================================================
# PSO hyperparameter optimization
# ============================================================
PSO_CONFIG = {
    "n_particles": 30,
    "n_iterations": 50,
    "c1": 1.5,      # Cognitive parameter
    "c2": 1.5,      # Social parameter
    "w": 0.7,       # Inertia weight
    "bounds": {
        # (hidden_layer_sizes_1, hidden_layer_sizes_2, alpha, learning_rate_init)
        # We encode hidden layers as: 0=single, 1=two layers
        # Layer sizes: 32, 64, 128, 256
        "hidden_config": (0, 3),      # Index into HIDDEN_CONFIGS
        "alpha": (1e-5, 1e-1),        # L2 regularization
        "learning_rate_init": (1e-4, 1e-1),
        "activation_idx": (0, 2),     # 0=relu, 1=tanh, 2=logistic
    },
}

# Predefined hidden layer configurations
HIDDEN_CONFIGS = [
    (64,),           # Single layer
    (128,),          # Single layer
    (256,),          # Single layer
    (64, 32),        # Two layers
    (128, 64),       # Two layers
    (256, 128),      # Two layers
]

ACTIVATIONS = ["relu", "tanh", "logistic"]

# Cross-validation
CV_FOLDS = 5
CV_SCORING = "f1_macro"  # or "accuracy"

# PSO fitness metric
PSO_METRIC = "cv_f1_macro"

# ============================================================
# MLP training (final model)
# ============================================================
MLP_MAX_ITER = 500
MLP_EARLY_STOPPING = True
MLP_VALIDATION_FRACTION = 0.1
MLP_N_ITER_NO_CHANGE = 20
MLP_RANDOM_STATE = 42

# ============================================================
# Evaluation
# ============================================================
N_PERMUTATION_REPEATS = 10
TOP_K_FEATURES = 20
RANDOM_STATE = 42

# ============================================================
# Fault class mapping (handles typo in filenames)
# ============================================================
FAULT_CLASSES = ["Normal", "BPFI", "BPFO", "Misalign", "Unbalance"]

def parse_filename(fname: str):
    """Parse filename into (load, condition, severity).
    Handles 'Unbalalnce' typo in 2Nm vibration files.
    """
    fname = fname.replace("Unbalalnce", "Unbalance")
    stem = fname.replace(".mat", "").replace(".tdms", "")
    parts = stem.split("_")
    load = parts[0]           # e.g., "0Nm", "2Nm", "4Nm"
    condition = parts[1]      # e.g., "Normal", "BPFI", "BPFO", "Misalign", "Unbalance"
    severity = "_".join(parts[2:]) if len(parts) > 2 else "none"
    return load, condition, severity

def get_class_index(condition: str) -> int:
    """Map condition string to class index."""
    return FAULT_CLASSES.index(condition)