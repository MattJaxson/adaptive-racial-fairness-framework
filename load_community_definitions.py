import json
import logging


def load_community_definitions(file_path='data/community_definitions.json'):
    """
    Load community fairness definitions from JSON.
    Falls back to defaults if the file is not found.
    """
    try:
        with open(file_path, 'r') as f:
            definitions = json.load(f)
            logging.info("Loaded community fairness definitions from %s", file_path)
            return definitions
    except FileNotFoundError:
        logging.info("No community definitions file found at %s — using defaults.", file_path)
        return {
            "fairness_definition": "Community-driven definition of equity",
            "priority_groups": ["Black", "Latinx"],
            "fairness_target": "White",
            "custom_metrics": [],
        }
