#!/usr/bin/env python3
"""
Simple example: How to use PathologyMetric with survival training data
"""

from xtuner.evaluation.metrics.pathology_metric import PathologyMetric

# Tokenizer config (replace with your actual config)
tokenizer_config = {
    'type': 'AutoTokenizer', 
    'pretrained_model_name_or_path': 'your_model_path'
}

# Initialize metric with training data path
metric = PathologyMetric(
    tokenizer=tokenizer_config,
    output_dir="./eval_output",
    survival_time_intervals=[0, 1, 2, 3, 5, 7, 10],
    survival_training_data_path="/path/to/training_data.json"  # Replace with actual path
)

# Training data JSON format example:
example_training_data = [
    {
        "category": "Survival_BRCA",
        "survival_targets": {
            "target_y": [0, 0, 1, 0, 0, 0, 0],      # Event occurs in interval 2
            "at_risk_mask": [1, 1, 1, 1, 0, 0, 0]   # At risk until interval 3
        }
    },
    {
        "category": "Survival_LUAD", 
        "survival_targets": {
            "target_y": [0, 0, 0, 0, 0, 0, 0],      # Censored (no event)
            "at_risk_mask": [1, 1, 1, 1, 1, 1, 1]   # At risk throughout
        }
    }
]

print("Usage: Initialize PathologyMetric with survival_training_data_path parameter")
print("Training data will be automatically loaded when survival metrics are first used")