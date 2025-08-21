# clearit/io/extract_test_ids.py
from typing import List, Dict, Any
import yaml
import re

def extract_test_ids(
    head_yaml_path: str,
    test_yaml_path: str,
    dataset_name: str,
    annotation_name: str,
    modes: List[str] = ["fold"]
) -> List[Dict[str, Any]]:
    """
    Extracts test IDs based on specified data_index_list modes.

    Parameters:
    - head_yaml_path: Path to the head YAML file.
    - test_yaml_path: Path to the test YAML file.
    - dataset_name: Dataset name to filter evaluations.
    - annotation_name: Annotation name to filter evaluations.
    - modes: List of modes to match in data_index_list. Supported:
        * "fold"         -> matches Nxx_foldYY (e.g., N01_fold00)
        * "best"         -> matches Nxx_best (excluding 'best_images')
        * "best_images"  -> matches Nxx_best_images

    Returns:
    - A sorted list of dicts with keys:
        * 'test_id': the matching test identifier
        * 'config': the N number as an integer
        * 'mode': the matched mode (fold, best, or best_images)
    """
    # Define regex patterns for each mode
    mode_patterns = {
        "fold": re.compile(r"N(\d+)_fold\d+"),
        "best": re.compile(r"N(\d+)_best(?!_images)"),
        "best_images": re.compile(r"N(\d+)_best_images"),
    }
    
    # Validate modes
    for mode in modes:
        if mode not in mode_patterns:
            raise ValueError(f"Unsupported mode '{mode}'. Choose from {list(mode_patterns)}.")
    
    # Load YAML
    with open(head_yaml_path) as f:
        head_data = yaml.safe_load(f)
    with open(test_yaml_path) as f:
        test_data = yaml.safe_load(f)
    
    # Map head_id -> (config, mode)
    config_map = {}
    for entry in head_data.get('heads', []):
        data_index = entry.get('data_index_list', "")
        for mode in modes:
            match = mode_patterns[mode].search(data_index)
            if match:
                head_id = entry.get('id')
                config_map[head_id] = {
                    "config": int(match.group(1)),
                    "mode": mode
                }
                break  # stop after first matching mode

    # Collect matching test IDs
    results = []
    for eval_entry in test_data.get('evaluations', []):
        head_id = eval_entry.get('head_id')
        info = config_map.get(head_id)
        if info and \
           eval_entry.get('dataset_name') == dataset_name and \
           eval_entry.get('annotation_name') == annotation_name:
            results.append({
                "test_id": eval_entry['test_id'],
                "config": info["config"],
                "mode": info["mode"]
            })
    
    # Sort by mode then config then test_id
    results.sort(key=lambda x: (x["mode"], x["config"], x["test_id"]))
    return results