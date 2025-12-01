"""
Mission config parser for extracting text prompts from mission briefing.
"""

import json
from pathlib import Path
from typing import List, Optional


def parse_mission_config(config_path: str) -> Optional[List[str]]:
    """
    Extract entity types from mission config JSON.

    Args:
        config_path: Path to mission_briefing/config.json

    Returns:
        List of class names (e.g., ["car", "pedestrian", "drone"])
        Returns None if parsing fails

    Example config.json:
        {
          "mission": {
            "entities": [
              {"type": "car", "priority": "high"},
              {"type": "pedestrian", "priority": "high"},
              {"type": "drone", "priority": "medium"}
            ]
          }
        }
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"[mission_parser] Config file not found: {config_path}")
            return None

        with open(config_file, 'r') as f:
            config = json.load(f)

        # Extract entity types from mission specification
        entities = config.get('mission', {}).get('entities', [])
        if not entities:
            print(f"[mission_parser] No entities found in config")
            return None

        class_names = [entity['type'] for entity in entities if 'type' in entity]

        if not class_names:
            print(f"[mission_parser] No valid entity types found")
            return None

        print(f"[mission_parser] Parsed {len(class_names)} entity types: {class_names}")
        return class_names

    except json.JSONDecodeError as e:
        print(f"[mission_parser] Failed to parse JSON: {e}")
        return None
    except Exception as e:
        print(f"[mission_parser] Error parsing mission config: {e}")
        return None


def construct_text_prompt(class_names: List[str]) -> str:
    """
    Construct GroundingDINO text prompt from class names.

    Args:
        class_names: List of class names (e.g., ["car", "pedestrian"])

    Returns:
        Formatted text prompt (e.g., "car. pedestrian.")

    Note:
        GroundingDINO expects format: "class1. class2. class3."
        Each class name separated by ". " and ending with "."
    """
    if not class_names:
        return ""

    # Remove any trailing/leading whitespace
    class_names = [name.strip() for name in class_names if name.strip()]

    # Join with ". " and ensure trailing "."
    prompt = ". ".join(class_names)
    if not prompt.endswith("."):
        prompt += "."

    return prompt


def get_text_prompt_from_mission(
    config_path: str,
    default_classes: Optional[List[str]] = None
) -> str:
    """
    Get text prompt from mission config with fallback to defaults.

    Args:
        config_path: Path to mission config JSON
        default_classes: Fallback class names if config parsing fails

    Returns:
        Text prompt string (e.g., "car. pedestrian.")
    """
    if default_classes is None:
        default_classes = ["car", "pedestrian"]

    # Try to parse mission config
    class_names = parse_mission_config(config_path)

    # Fallback to defaults if parsing failed
    if class_names is None or len(class_names) == 0:
        print(f"[mission_parser] Using default classes: {default_classes}")
        class_names = default_classes

    return construct_text_prompt(class_names)


if __name__ == "__main__":
    # Test the parser
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "/mission_briefing/config.json"

    print(f"Testing mission parser with: {config_path}")
    print(f"{'='*60}")

    # Parse config
    class_names = parse_mission_config(config_path)
    print(f"Extracted classes: {class_names}")

    # Construct prompt
    if class_names:
        prompt = construct_text_prompt(class_names)
        print(f"Text prompt: '{prompt}'")

    # Test with fallback
    prompt = get_text_prompt_from_mission(config_path, default_classes=["object", "person"])
    print(f"Final prompt (with fallback): '{prompt}'")
