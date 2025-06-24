import numpy as np
import logging
from typing import Tuple, List, Dict

def load_data(file_path: str) -> Tuple[List[Dict[int, float]], np.ndarray]:
    """Load LIBSVM-style data from a file."""
    data = []
    labels = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split()
                if not items:
                    continue
                try:
                    labels.append(float(items[0]))
                    feat_dict = {}
                    for item in items[1:]:
                        index, value = item.split(':')
                        feat_dict[int(index)] = float(value)
                    data.append(feat_dict)
                except Exception as e:
                    logging.warning(f"Skipping malformed line: {line.strip()} ({e})")
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise
    return data, np.array(labels)

def create_feature_matrix(data: List[Dict[int, float]], n_features: int) -> np.ndarray:
    X = np.zeros((len(data), n_features))
    for i, feat_dict in enumerate(data):
        for j, v in feat_dict.items():
            if 1 <= j <= n_features:
                X[i, j-1] = v
    return X 