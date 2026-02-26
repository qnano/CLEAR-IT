# clearit/models/utils.py
from pathlib import Path
from clearit.config import OUTPUTS_DIR

def get_group_color(group_label, shade=0):
    """
    Retrieve a color corresponding to a given class label (or other group label) and shade index.

    Parameters:
    group_label (str): The group label for which the color is to be retrieved (e.g. class labels for datasets, or input types or encoder names for other comparison plots).
    shade (int, optional): The shade index of the color to be retrieved. Defaults to 0.

    Returns:
    list: A list of three float values representing the RGB color corresponding to the given group label and shade index.

    Usage:
    >>> get_group_color("CK+", 1)
    [0.0, 1.0, 1.0]
    """
    
    colors = {
        "CK+":    [[0,   194, 213], [  0, 255, 255], [0,   116, 128]], 
        "CK":     [[0,   194, 213], [  0, 255, 255], [0,   116, 128]], 
        "CXCR3+": [[0,    94, 213], [  0, 155, 255], [0,    16, 128]], 
        "CD21+":  [[0,    94, 213], [  0, 155, 255], [0,    16, 128]], 
        "CD3+":   [[0,   194,   0], [  0, 255,   0], [0,   116,   0]], 
        "CD3":    [[0,   194,   0], [  0, 255,   0], [0,   116,   0]], 
        "CD4+":   [[0,   194,   0], [  0, 255,   0], [0,   116,   0]], 
        "CD8+":   [[255,   0,   0], [255,  50,  50], [204,   0,   0]], 
        "CD3 CD8":[[255,   0,   0], [255,  50,  50], [204,   0,   0]], 
        "FOXP3+": [[255,   0,   0], [255,  50,  50], [204,   0,   0]], 
        "CD20+":  [[255, 194,   0], [255, 255,   0], [204, 116,   0]], 
        "CD20":   [[255, 194,   0], [255, 255,   0], [204, 116,   0]], 
        "CD11b+": [[255, 194,   0], [255, 255,   0], [204, 116,   0]], 
        "CD56+":  [[255, 114, 213], [255, 150, 255], [204,  68, 128]], 
        "CD56":   [[255, 114, 213], [255, 150, 255], [204,  68, 128]], 
        "PD1+":   [[255, 114, 213], [255, 150, 255], [204,  68, 128]], 
        "GZM-B+": [[255, 114, 213], [255, 150, 255], [204,  68, 128]], 
        "CD68+":  [[235, 120,   0], [235, 158,   0], [204,  72,   0]],
        "CD68":   [[235, 120,   0], [235, 158,   0], [204,  72,   0]],
        "T-bet+": [[235, 120,   0], [235, 158,   0], [204,  72,   0]],
        "features":             [[0,   154, 213], [0,   194, 255], [0,   104, 128]], 
        "expressions":          [[213,  81,  67], [255, 111,  97],  [128,  51,  37]], 
        "features_expressions": [[213,   213,   0], [255, 255,   0], [128,   128,   0]],
        "TNBC1-MxIF8":  [[  0,   154, 213], [0,   194, 255], [0,     104, 128]], 
        "TNBC2-MIBI8":  [[ 66,   213,  66], [ 87, 255,  87], [ 33,   128,  33]], 
        "TNBC2-MIBI44": [[ 66,   213,  66], [ 87, 255,  87], [ 33,   128,  33]], 
        "CRC-CODEX26":  [[213,    81,  67], [255, 111,  97], [128,    51,  37]], 
        "TONSIL-IMC41": [[213,   213,   0], [255, 255,   0], [128,   128,   0]],
    }
    shades = colors.get(group_label, [[0, 0, 0], [179, 179, 179]])  # default to gray shades if the group is not in the dictionary
    return [shade_val/255 for shade_val in shades[shade % len(shades)]]  # return the requested shade as a list of values between 0 and 1 (wrapping around if necessary)

def build_single_patient_entries(
    patient_start: int = 1,
    patient_end: int = 47,
    test_start_id: int = 412,
    dataset_label: str = "TNBC1-MxIF8",
) -> list:
    """
    Build entries for networks trained on individual patients.

    Mapping:
        P01 -> T0412
        P02 -> T0413
        ...
        P47 -> T0458

    Parameters
    ----------
    patient_start, patient_end : int
        Inclusive range of patient indices (1-based).
    test_start_id : int
        Numeric part of the test_dir for patient_start (e.g. 412 for T0412).
    dataset_label : str
        Value to put in the 'group' field (e.g. dataset name).

    Returns
    -------
    entries : list of dict
        Each dict has keys: "path", "group", "config".
    """
    entries = []
    for i in range(patient_start, patient_end + 1):
        patient_id = f"P{i:02d}"                         # e.g. "P01"
        test_id = test_start_id + (i - patient_start)    # 412, 413, ...
        test_dir_name = f"T{test_id:04d}"                # e.g. "T0412"

        entry = {
            "path": Path(OUTPUTS_DIR, "tests", test_dir_name),
            "group": dataset_label,
            "config": patient_id,
        }
        entries.append(entry)

    return entries