"""File utility functions"""

import json
from pathlib import Path
from typing import List, Dict, Union


def save_json(data: Union[List, Dict], file_path: Path) -> None:
    """Save data to JSON file

    Args:
        data: Data to save (list or dict)
        file_path: Path to save the file to
    """
    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert data to JSON-serializable format
    if isinstance(data, list):
        json_data = [item.dict() if hasattr(item, "dict") else item for item in data]
    else:
        json_data = data.dict() if hasattr(data, "dict") else data

    # Save to file
    with open(file_path, "w") as f:
        json.dump(json_data, f, indent=2)
