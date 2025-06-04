import json

def load_community_definitions(file_path='data/community_definitions.json'):
    """
    Load community fairness definitions from JSON.
    """
    try:
        with open(file_path, 'r') as f:
            definitions = json.load(f)
            print(f"Loaded community fairness definitions from {file_path}")
            return definitions
    except FileNotFoundError:
        print(f"No community fairness definitions found. Using flexible defaults.")
        return {
            "fairness_definition": "Community-driven definition of equity",
            "priority_groups": ["Black", "Latinx"],
            "fairness_target": "White",
            "custom_metrics": []
        }
