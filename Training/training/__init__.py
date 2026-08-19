import os
import sys

# Make the interpretability module's `single` package importable from Training,
# so both modules share the same label-map logic (single.label_maps).
_single_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Single"))
if _single_dir not in sys.path:
    sys.path.insert(0, _single_dir)

from .config import TrainingConfig
from .data import TokenClassificationDataset, load_data_from_csv, split_dataset, build_label_map, build_id2label, label_map_n_classes, compute_class_weights, compute_class_weights_from_dataset
from .models import PLMModel
from .trainers import Trainer
from .utils import create_task_folder, save_config, load_config
