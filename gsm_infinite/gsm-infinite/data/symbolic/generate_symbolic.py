# import generate_answer
import random
import generate_payload
# import generator_calc
# from datasets import Dataset
import json
import os


def dump_dict_to_json(data, filename):
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

def prepare_payload(payload_items):

    payloads = [f"{str(item)}." for item in payload_items]

    new_context = " ".join(payloads)
    
    return new_context

def get_payload(length, insert_points=None, op=1, N=1):
    output_list, query, value, solution = generator.generate_task(op, N, with_solution=True)
    
    context = prepare_payload(output_list)
    import utils
    return f"{utils.get_symbolic_prompt(query[0], context)}", context, f"{utils.get_symbolic_prompt_query(query[0])}", solution, query
    

def get_benchmark_info(length, insert_points, N, op, id, close_rate):
    N+= int(length * 828 / 9668 * close_rate)
    input, problem, question, solution, query = get_payload(int(length * (1-close_rate)), insert_points, op, N)
    answer_q, answer_list = query
    # conversation = [{"role": "user", "content": input},
    #                 {"role": "assistant", "content": solution}]
    messages = [{"role": "user", "content": input}]

    return {"problem": problem, "question": question, "solution": solution, "op": op, "n": N, "length": length, "id": id, "d": 1, "answer_q": answer_q, "answer_list": answer_list, "messages": messages}

# print(get_payload(100, 2))
if __name__ == '__main__':
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import concurrent.futures
    import tqdm
    import random
    random.seed(42)
    from datasets import Dataset, DatasetDict

    import argparse
    parser = argparse.ArgumentParser(
        description="generate dataset with command line arguments."
    )

    parser.add_argument('--dataset-name', type=str, help="The name of the dataset for organizing the folders")
    # Required arguments
    parser.add_argument(
        '--max-op',
        type=int,
        required=False,
        default=40,
        help='max op to generate, default 40'
    )

    parser.add_argument(
        '--min-op',
        type=int,
        required=False,
        default=1,
        help='min op to generate, default 1'
    )

    parser.add_argument(
        '--stride',
        type=int,
        required=False,
        default=1,
        help='stride size to skip op, default 1',
    )

    parser.add_argument(
        '--examples-per-op',
        type=int,
        required=False,
        default=50,
        help='examples per op to generate, default 50',
    )

    parser.add_argument(
        '--length',
        type=str,
        default="0",
        help='noise context length'
    )


    args = parser.parse_args()
    ddict = {}
    length = args.length
    if isinstance(length, str):
        if length.lower().endswith('k'):
            length = int(length[:-1]) * 1000
        else:
            length = int(length)
    seed = 42
    generator = generate_payload.FindGraphGenerator(seed)
    dataset_list = []
    ops = []
    ids = []
    for op in range(1, args.max_op, args.stride):
        ops = [op]*args.examples_per_op
        ids = list(range(args.examples_per_op))

        def generate_examples(ops, ids, length):
            for id, op in tqdm.tqdm(zip(ids, ops), total=len(ops), desc=f"Generating examples for length {length}"): # Added tqdm here
                insert_points = None
                yield get_benchmark_info(length, insert_points, op, op, id, 1.0)

        dataset = Dataset.from_generator(generate_examples, gen_kwargs={"ops": ops, "ids": ids, "length": length})
        if (op < args.min_op):
            continue
        dataset.push_to_hub(f"{args.dataset_name}_{args.length}", split=f"ops_{op}", private=True)
 

    

    



