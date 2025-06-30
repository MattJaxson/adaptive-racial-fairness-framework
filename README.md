# Adaptive Racial Fairness Framework

A project to measure and mitigate racial disparities in AI systems, with community-driven fairness definitions, adaptive reweighting, and interactive real-time dashboards.

This is **Step 1: Creating the project structure and core modules.**

---

## Project Overview

The Adaptive Racial Fairness Framework provides tools to evaluate, visualize, and rebalance datasets using fairness definitions defined by impacted communities. It includes:

- A Dash-based Fairness Dashboard for interactive analysis.
- Modules to calculate racial bias metrics.
- Adaptive sample reweighting functions to align datasets with community fairness goals.
- Community definition loaders for customizing what “fair” means in your context.
- Logging for transparency and debugging.

---

## Repository Structure

adaptive-racial-fairness-framework/
│
├── adaptive_app.py
    The main Dash application. This script launches the interactive dashboard and sets up the callbacks that update the visuals in real time.

├── fairness_reweight.py
    Contains the functions that adjust (reweight) your dataset so it better matches the fairness goals defined by your community.

├── calculate_racial_bias_score.py
    Provides functions to calculate metrics that show how racially biased your dataset or model outcomes might be.

├── load_community_definitions.py
    Loads community-defined fairness goals from a JSON file so you can apply them to your dataset.

├── utils.py
    Helper functions for tasks like setting up consistent logging throughout your code.

├── assets/style.css
    Stylesheet used by the Dash dashboard to customize the layout and appearance.

├── data/community_definitions.json
    Example JSON file showing how to define your community’s fairness goals and bias tolerance thresholds.

├── requirements.txt
    List of all Python packages your project needs. Used to install dependencies with pip.

└── README.md
    Documentation file explaining what the project does, how to set it up, and how to use it.


---

## Setup

Clone the repository:

```bash
git clone https://github.com/MattJaxson/adaptive-racial-fairness-framework.git
cd adaptive-racial-fairness-framework
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```
Run the dashboard locally:
```bash
python adaptive_app.py
```

