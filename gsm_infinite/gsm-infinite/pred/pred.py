from model_handler import ModelHandler
from no_rag_pipeline import NoRAGPipeline

def dump_dict_to_json(data, filename):
    import os
    import json
    """Dumps a Python dictionary to a JSON file, creating the directory if needed.

    Args:
        data: The Python dictionary to be dumped.
        filename: The name of the JSON file to be created (e.g., "data/output.json").
    """
    try:
        # Extract the directory path from the filename
        directory = os.path.dirname(filename)

        # Create the directory if it doesn't exist
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
            print(f"Successfully dumped dictionary to {filename}")
    except (TypeError, OSError) as e:
          print(f"Error dumping dictionary to JSON: {e}")


# print(get_payload(100, 2))
if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor
    import concurrent.futures
    import tqdm
    from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
    import json

    # parser = argparse.ArgumentParser(description="Run benchmark tests and organize results")
    # parser.add_argument('--model-name', type=str, help="The name of the model for organizing the folders")
    import argparse
    parser = argparse.ArgumentParser(
        description="Sample with command line arguments."
    )
    parser.add_argument('--save-name', type=str, help="Save model name", default="base")
    parser.add_argument('--save-dataset', type=str, help="Save dataset name", default="base")
    parser.add_argument('--dataset-name', type=str, help="The name of the dataset for organizing the folders")
    # Required arguments
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Name of the model to use in api call.'
    )
    parser.add_argument(
        '--backend-type',
        type=str,
        default="openai",
        help='backend type in [\'openai\', \'anthropic\', \'gemini\']'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of samples to generate per example.'
    )

    # Optional arguments with default values
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Sampling temperature (default: None).'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=3072,
        help='Maximum number of tokens (default: 3072).'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=200,
        help='Batch size (default: 200).'
    )

    parser.add_argument(
        '--length',
        type=str,
        default="0",
        help='noise context length'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help="max number of examples per op"
    )

    parser.add_argument(
        '--filter-config',
        type=json.loads,
        help='Filter configuration as a JSON string.'
    )

    parser.add_argument(
        '--op-range',
        type=str,
        help='Operating range, can be an integer, or a list of integers separated by commas.'
    )
    args = parser.parse_args()

    if args.op_range:
        try:
            # Attempt to parse as a single integer
            args.op_range = [int(args.op_range)]
        except ValueError:
            # If not a single integer, split by comma and convert to integers
            try:
                args.op_range = [int(x.strip()) for x in args.op_range.split(',')]
            except ValueError:
                raise ValueError("Invalid input for --op-range. Please provide an integer or a comma-separated list of integers.")


    subsets = [f"ops_{x}" for x in args.op_range]
    use_full_query = True

    model_handler = ModelHandler(
        model_name=args.model_name,
        backend_type=args.backend_type
    ) 
    pipeline = NoRAGPipeline(
        model_handler = model_handler, 
        max_tokens=args.max_tokens,
        temperature=args.temperature
    ) 
    use_full_query = True

    
    # for length in [0, 8000, 16000, 32000, 64000, 128000]:
    length = args.length
    try:

        # opset = set(args.op_range)
        # unprocessed_dataset = unprocessed_dataset.filter(lambda example: example["op"] in opset)
        full_dataset = load_dataset(f"{args.dataset_name}_{length}")
        filter_config = args.filter_config
        if filter_config:
            filtered_datasets = []
            for split in subsets:
                dataset_split = full_dataset[split]
                total_samples = min(args.limit, len(dataset_split))
                filtered_data = []
                for config in filter_config:
                    num_to_add = int(total_samples * config["percentage"])
                    current_filter = {key: value for key, value in config.items() if key not in ["percentage"]}
                    filtered_subset = dataset_split.filter(lambda example: all(example[key] == value for key, value in current_filter.items()))
                    filtered_data.extend(filtered_subset.select(range(min(num_to_add, len(filtered_subset)))))
                filtered_datasets.append(Dataset.from_list(filtered_data))
            unprocessed_dataset = concatenate_datasets(filtered_datasets)
        else:
            unprocessed_dataset = concatenate_datasets([full_dataset[split].select(range(min(args.limit, len(full_dataset[split])))) for split in subsets])
        # unprocessed_dataset = concatenate_datasets([full_dataset[split].select(range(min(args.limit, len(full_dataset[split])))) for split in subsets])
        # unprocessed_dataset = load_from_disk(
        #     f"{args.dataset_name}_{length}",
        #     # data_dir=f"o
        #     # split=str(length),
        # )
        # with open(args.dataset_name, 'r') as f:
        #     unprocessed_dataset = json.load(f)[str(length)]
        # print(unprocessed_dataset)

        len_dataset = len(unprocessed_dataset)
        contexts = []
        queries = []
        for i in range(0, len_dataset):
            for _ in range(args.num_samples):
                queries.append(unprocessed_dataset[i]['messages'])

        replies = pipeline.process_batch(queries=queries, max_workers=args.batch_size)
        processed_examples = []
        
        for i in range(0, len_dataset):
            newline = unprocessed_dataset[i]
            newline["replies"] = replies[i*args.num_samples:(i+1)*args.num_samples]
            newline.pop("problem", "")
            newline.pop("question", "")
            newline.pop("messages", "")
            processed_examples.append(newline)

        # print(replies[0])
        import os
        dir_name = "datasets"
        os.makedirs(dir_name, exist_ok=True)  # Create directory if it doesn't exist

        dump_dict_to_json(processed_examples, f"{dir_name}/{args.save_dataset}-{args.save_name}_{length}")
    except Exception as e:
        print(e)
        raise
