import os
import pandas as pd
import re

# Directory containing result files
RESULTS_DIR = 'results/'

# Initialize an empty list to store data
data = []

# Regex pattern to extract dataset and model from filename
filename_pattern = re.compile(r"^result_(?P<dataset>[A-Za-z0-9_\.]+)_(?P<model>[A-Za-z0-9\-\.]+)\.txt$")

# Iterate over all files in the results directory
for filename in os.listdir(RESULTS_DIR):
    match = filename_pattern.match(filename)
    if match:
        dataset = match.group("dataset")
        model = match.group("model")
        filepath = os.path.join(RESULTS_DIR, filename)
        
        try:
            with open(filepath, 'r') as file:
                for line in file:
                    # Parse the line
                    parts = line.strip().split(',')
                    length = int(parts[0].split(':')[1].strip())
                    
                    # Check if this is a line with fine-grained statistics
                    has_subset_info = any('template:' in part for part in parts) or any('mode:' in part for part in parts)
                    
                    # Extract operation number (N)
                    op_part = parts[1].strip()
                    if op_part.startswith('op:'):
                        N = int(op_part.split(':')[1].strip())
                    elif op_part.startswith('N:'):
                        N = int(op_part.split(':')[1].strip())
                    else:
                        raise ValueError(f"Unexpected format for operation: {op_part}")
                    
                    # Extract accuracy
                    acc_part = next(part for part in parts if 'acc:' in part)
                    acc = float(acc_part.split(':')[1].strip())
                    
                    # Initialize entry with common fields
                    entry = {
                        'dataset': dataset,
                        'model': model,
                        'length': length,
                        'N': N,
                        'accuracy': acc,
                        'has_subset_info': has_subset_info
                    }
                    
                    # Add subset information if available
                    if has_subset_info:
                        for part in parts[2:]:  # Skip length and op parts
                            if 'num_examples:' in part:
                                entry['num_examples'] = int(part.split(':')[1].strip())
                            elif 'template:' in part:
                                entry['template'] = part.split(':')[1].strip()
                            elif 'mode:' in part:
                                entry['mode'] = part.split(':')[1].strip()
                    
                    data.append(entry)
                    
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            raise

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Fill NaN values for subset columns
if 'template' in df.columns:
    df['template'].fillna('default', inplace=True)
if 'mode' in df.columns:
    df['mode'].fillna('default', inplace=True)
if 'num_examples' in df.columns:
    df['num_examples'].fillna(-1, inplace=True)

# Optional: Save the processed data for future use
df.to_csv('results/processed_results.csv', index=False)

print(f"Processed {len(df)} data points from {len(df['dataset'].unique())} datasets and {len(df['model'].unique())} models.")