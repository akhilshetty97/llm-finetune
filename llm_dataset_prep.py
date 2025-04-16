#!/usr/bin/env python3
"""
Simplified dataset preparation script for MIMIC-CXR fine-tuning.
This script creates a JSONL file with prompt-completion pairs for LLM fine-tuning.

Simply run this script from /src/ directory to generate the dataset.
"""

import os
import json
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random

# UPDATED PATHS FOR YOUR ENVIRONMENT
# The path to the CheXpert labels CSV file (absolute path)
CHEXPERT_LABELS_PATH = "/Users/akhilshetty/Desktop/CapstoneProjects/mimic-cxr_flower_try/mimic-cxr-2.0.0-chexpert.csv"

# Base directory where report files are stored
BASE_PATH = "/Users/akhilshetty/Desktop/CapstoneProjects/mimic-cxr_flower_try"

# Where to save the output files
OUTPUT_DIR = "/Users/akhilshetty/Desktop/CapstoneProjects/mimic-cxr_flower_try/llmdata"
OUTPUT_FILE = "mimic_cxr_finetune.jsonl"

# Configuration options
ONLY_POSITIVE = True  # Only include studies with at least one positive finding
CREATE_SPLITS = True  # Create train/val/test splits

def extract_findings_impression(report_text):
    """
    Extract the findings and impression sections from a radiology report.
    """
    # Define regex patterns for findings and impression sections
    findings_pattern = r"FINDINGS:(.*?)(?:IMPRESSION:|$)"
    impression_pattern = r"IMPRESSION:(.*?)(?:$)"
    
    # Extract findings
    findings_match = re.search(findings_pattern, report_text, re.DOTALL | re.IGNORECASE)
    findings = findings_match.group(1).strip() if findings_match else ""
    
    # Extract impression
    impression_match = re.search(impression_pattern, report_text, re.DOTALL | re.IGNORECASE)
    impression = impression_match.group(1).strip() if impression_match else ""
    
    return findings, impression

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Print current directory to help with debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for CheXpert labels at: {CHEXPERT_LABELS_PATH}")
    
    # Check if the file exists
    if not os.path.exists(CHEXPERT_LABELS_PATH):
        print("ERROR: CheXpert labels file not found!")
        print("Please check the path provided.")
        return
    
    print(f"Loading CheXpert labels from {CHEXPERT_LABELS_PATH}...")
    chexpert_df = pd.read_csv(CHEXPERT_LABELS_PATH)
    
    # Define condition columns
    condition_columns = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]
    
    # Check if all columns exist in the DataFrame
    missing_columns = [col for col in condition_columns if col not in chexpert_df.columns]
    if missing_columns:
        print(f"WARNING: The following columns are missing from the CSV: {missing_columns}")
        print("Available columns:")
        for col in chexpert_df.columns:
            print(f"  - {col}")
        
        # Continue with available columns
        condition_columns = [col for col in condition_columns if col in chexpert_df.columns]
        print(f"Continuing with available columns: {condition_columns}")
    
    # Filter studies with at least one positive finding if requested
    if ONLY_POSITIVE and condition_columns:
        print("Filtering for studies with positive findings...")
        chexpert_df = chexpert_df[chexpert_df[condition_columns].eq(1.0).any(axis=1)]
    
    # Print a sample row to check format
    print("\nSample row from CSV:")
    sample_row = chexpert_df.iloc[0]
    print(sample_row)
    print(f"subject_id type: {type(sample_row['subject_id'])}, value: {sample_row['subject_id']}")
    print(f"study_id type: {type(sample_row['study_id'])}, value: {sample_row['study_id']}")
    
    # Process each study
    print(f"Processing {len(chexpert_df)} studies...")
    records = []
    skipped_count = 0
    found_count = 0
    
    for _, row in tqdm(chexpert_df.iterrows(), total=len(chexpert_df)):
        # Convert study_id and subject_id to integers if they're floats
        study_id = int(row['study_id']) if isinstance(row['study_id'], float) else row['study_id']
        subject_id = int(row['subject_id']) if isinstance(row['subject_id'], float) else row['subject_id']
        
        # Construct path to the report using the BASE_PATH - remove any decimal part
        # Try multiple possible formats
        possible_paths = [
            # Format without decimal points
            os.path.join(BASE_PATH, f"files/p{str(subject_id)[:2]}/p{subject_id}/s{study_id}.txt"),
            
            # Original path format from CSV
            os.path.join(BASE_PATH, f"files/p{str(subject_id)[:2]}/p{row['subject_id']}/s{row['study_id']}.txt"),
            
            # Try additional common formats
            os.path.join(BASE_PATH, f"files/p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/image.txt")
        ]
        
        # Try to find the report
        report_path = None
        for path in possible_paths:
            if os.path.exists(path):
                report_path = path
                break
        
        # Skip if report doesn't exist
        if not report_path:
            skipped_count += 1
            if skipped_count <= 5:  # Only print the first few to avoid spam
                print(f"Warning: Report not found for subject_id={subject_id}, study_id={study_id}")
                print(f"Tried paths: {possible_paths}")
            continue
        
        # Read report
        try:
            with open(report_path, 'r') as f:
                report_text = f.read()
            
            # Extract findings and impression
            findings, impression = extract_findings_impression(report_text)
            
            # Skip if findings or impression is missing
            if not findings or not impression:
                skipped_count += 1
                continue
            
            # Create the completion text
            completion = f" {findings}\nImpression: {impression}"
            
            # Create the prompt
            prompt_conditions = []
            for condition in condition_columns:
                if condition in row:
                    value = row[condition]
                    # Convert float values to integers if they're whole numbers
                    if pd.notna(value):
                        if value == 1.0:
                            display_value = "1"
                        elif value == 0.0:
                            display_value = "0"
                        elif value == -1.0:
                            display_value = "uncertain"
                        else:
                            display_value = str(value)
                        
                        prompt_conditions.append(f"{condition}: {display_value}")
            
            prompt = "; ".join(prompt_conditions) + ". \nFindings:"
            
            # Create the record
            record = {
                "prompt": prompt,
                "completion": completion,
                "study_id": study_id,
                "subject_id": subject_id
            }
            
            records.append(record)
            found_count += 1
            
            # Print a few successful examples
            if found_count <= 3:
                print(f"\nSuccessfully processed report at: {report_path}")
                print(f"Prompt: {prompt[:100]}...")
                print(f"Completion: {completion[:100]}...")
        
        except Exception as e:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"Error processing {report_path}: {e}")
    
    print(f"Skipped {skipped_count} studies due to missing reports or incomplete content.")
    print(f"Successfully processed {found_count} studies.")
    
    # Write to JSONL file
    output_file_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    print(f"Writing {len(records)} records to {output_file_path}...")
    
    with open(output_file_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    # Split into train/val/test sets if requested
    if CREATE_SPLITS and records:
        # Shuffle the records
        random.seed(42)
        random.shuffle(records)
        
        # Calculate split indices
        train_idx = int(len(records) * 0.8)
        val_idx = int(len(records) * 0.9)
        
        # Split the records
        train_records = records[:train_idx]
        val_records = records[train_idx:val_idx]
        test_records = records[val_idx:]
        
        # Write splits
        print(f"Splitting into {len(train_records)} train, {len(val_records)} val, {len(test_records)} test...")
        
        with open(os.path.join(OUTPUT_DIR, "train.jsonl"), 'w') as f:
            for record in train_records:
                f.write(json.dumps(record) + '\n')
                
        with open(os.path.join(OUTPUT_DIR, "val.jsonl"), 'w') as f:
            for record in val_records:
                f.write(json.dumps(record) + '\n')
                
        with open(os.path.join(OUTPUT_DIR, "test.jsonl"), 'w') as f:
            for record in test_records:
                f.write(json.dumps(record) + '\n')
    
    print(f"Done! Created dataset with {len(records)} examples.")
    print(f"Output files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()