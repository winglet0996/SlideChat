#!/usr/bin/env python3
"""
Generate evaluation samples from test dataset for SlideChat training configuration.
This script samples evaluation data and saves the results to be imported by the config file.
"""

import json
import random
import os
from pathlib import Path


def sample_eval_data(eval_data, sample_num=5, seed=42):
    """Sample evaluation data with reproducible random seed"""
    random.seed(seed)
    return random.sample(eval_data, min(sample_num, len(eval_data)))


def generate_eval_samples(eval_data_path, output_path, sample_num=5, seed=42):
    """
    Generate evaluation samples from the test dataset and save to a Python file.
    
    Args:
        eval_data_path: Path to the evaluation JSON file
        output_path: Path to save the generated Python file
        sample_num: Number of samples to select
        seed: Random seed for reproducibility
    """
    
    # Load evaluation data
    if not os.path.exists(eval_data_path):
        raise FileNotFoundError(f"Evaluation data file not found: {eval_data_path}")
    
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    print(f"Loaded {len(eval_data)} items from {eval_data_path}")
    
    # Sample evaluation data
    sampled_eval = sample_eval_data(eval_data, sample_num, seed)
    print(f"Sampled {len(sampled_eval)} items for evaluation")
    
    # Extract evaluation components
    evaluation_images = [item['image'][0] for item in sampled_eval]
    evaluation_inputs = [
        item['conversations'][0]['value'].replace('\n<image>', '').replace('<image>', '').strip()
        # item['conversations'][0]['value'].strip()
        for item in sampled_eval
    ]
    evaluation_targets = [item['conversations'][1]['value'] for item in sampled_eval]
    
    # Generate Python code for the configuration file
    python_code = f'''"""
Evaluation samples generated from {eval_data_path}
Generated with seed={seed}, sample_num={sample_num}
"""

# Evaluation images (WSI feature paths)
evaluation_images = [
'''
    
    for i, img in enumerate(evaluation_images):
        python_code += f'    "{img}",  # Sample {i+1}\n'
    
    python_code += ']\n\n# Evaluation inputs (questions/prompts)\nevaluation_inputs = [\n'
    
    for i, inp in enumerate(evaluation_inputs):
        # Escape quotes and newlines in the input text
        escaped_inp = inp.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        python_code += f'    "{escaped_inp}",  # Question {i+1}\n'
    
    python_code += ']\n\n# Evaluation targets (ground truth answers)\nevaluation_targets = [\n'
    
    for i, target in enumerate(evaluation_targets):
        # Escape quotes and newlines in the target text
        escaped_target = target.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        python_code += f'    "{escaped_target}",  # Target {i+1}\n'
    
    python_code += ']\n'
    
    # Save to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(python_code)
    
    print(f"Evaluation samples saved to {output_path}")
    
    # Also save raw data as JSON for reference
    json_output_path = output_path.replace('.py', '_raw.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'source_file': eval_data_path,
                'seed': seed,
                'sample_num': sample_num,
                'total_samples': len(sampled_eval)
            },
            'sampled_data': sampled_eval
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Raw sampled data saved to {json_output_path}")
    
    return evaluation_images, evaluation_inputs, evaluation_targets


def main():
    """Main function"""
    # Configuration - use available data file
    eval_data_path = '/home/winglet/pathology/vqa/dataset_pp/PathoVerse_train_stage1_caption.json'
    output_path = '/home/winglet/pathology/vqa/SlideChat/xtuner/configs/slidechat/eval_samples.py'
    sample_num = 2
    seed = 42
    
    print("SlideChat Evaluation Sample Generator")
    print("=" * 50)
    print(f"Input file: {eval_data_path}")
    print(f"Output file: {output_path}")
    print(f"Sample count: {sample_num}")
    print(f"Random seed: {seed}")
    print("=" * 50)
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate evaluation samples
        eval_images, eval_inputs, eval_targets = generate_eval_samples(
            eval_data_path, output_path, sample_num, seed
        )
        
        print("\nGeneration completed successfully!")
        print(f"\nSample preview:")
        print(f"First image: {eval_images[0]}")
        print(f"First question: {eval_inputs[0][:100]}...")
        print(f"First target: {eval_targets[0][:100]}...")
        
        print(f"\nTo use in your config file, add this import:")
        print(f"from .eval_samples import evaluation_images, evaluation_inputs, evaluation_targets")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
