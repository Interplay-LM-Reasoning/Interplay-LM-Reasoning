import re
from no_rag_pipeline import NoRAGPipeline
from model_handler import ModelHandler

import re

def get_prompt(reply, value):
    if "</think>" in reply:
        reply = reply.split("</think>")[-1].strip()
    return f"""
    Carefully read the input text below and extract the variable names that are mentioned as being equal to {value}. If multiple variables are listed, separate them with commas.
    
    If value {value} is not mentioned, you can safely assume that they are all equal to {value}. If there are no such variables, just output None.

    Only output the variable names, not the values or any other text.
    
    Examples:
    1. input: "V0, V3, V4", output: "V0, V3, V4"
    2. input: "Variable V2 is equal to {value} in the assignment statement 'assign V2 = V1 - 1.'.", output: "V2"
    3. input: "The answer is: V1, V2.", output: "V1, V2"
    4. input: "There are no variables equal to {value}.", output: "None"

    Input:
    {reply}

    Output:
    """

def preprocess_line(line):
    replies = line["replies"]
    value = line["answer_q"]   
    return [get_prompt(reply, value) for reply in replies]
    



def postprocess_line(line, extractions):
    correct_counter = 0
    line.pop("replies")
    variable_list = line["answer_list"]
    reply_variables = []
    for extraction in extractions:
        variables = re.findall(r'\bV\d+\b', extraction)
        reply_variables.append(list(variables))
        if set(variable_list) == set(variables):
            correct_counter += 1
    line["correct_num"] = correct_counter
    line["reply_answers"] = reply_variables
    
    
    return line
            
if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor
    import concurrent.futures
    import tqdm
    import argparse
    import json
    parser = argparse.ArgumentParser(
        description="Eval with command line arguments."
    )
    parser.add_argument('--save-name', type=str, help="The name of the file saved for organizing the folders", default="base")
    # parser.add_argument('--dataset-name', type=str, help="The name of the dataset for organizing the folders")
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of samples to generate.'
    )

    parser.add_argument(
        '--length',
        type=str,
        default="0",
        help='noise context length'
    )

    args = parser.parse_args()
    model_handler = ModelHandler(
        model_name = "Qwen/Qwen2.5-7B-Instruct", 
    ) 
    pipeline = NoRAGPipeline(
        model_handler = model_handler, 
        temperature=0.0,
        max_tokens=2048
    )
    
    if (args.num_samples == 1):
        sample_nums = [None]
    else: 
        sample_nums = [1<<i for i in range(args.num_samples.bit_length()) if (1<<i) < args.num_samples]

    length=args.length
    try:
        dir_name = "datasets"
        with open(f"{dir_name}/symbolic-{args.save_name}_{str(length)}", 'r') as f:
            unprocessed_dataset = json.load(f)
        
        results = []
        processed_examples = []
        values = []
        count_dict = {}
        correct_dict = {}
        
        submission_list = []
        num_samples = len(unprocessed_dataset[0]["replies"])
    
        len_dataset = len(unprocessed_dataset)
        # len_dataset = 1

        for i in range(len_dataset):
            submission_list.extend(preprocess_line(unprocessed_dataset[i]))

        
        # extractions = [retrieve_vars_formatted(submission) for submission in submission_list_regex]
        extractions = pipeline.process_batch(submission_list, ["" for _ in submission_list], max_workers=50)
        # print(extractions)
        # print(unprocessed_dataset[0])
                    
        for i in range(0, len_dataset):
            results.append(postprocess_line(unprocessed_dataset[i], 
                                            [extractions[j] for j in range(i*num_samples, (i+1)*num_samples)])
                        )
            
        # print(results)

        for processed_example in results:
            op = processed_example["op"]
            count_dict.setdefault(op, 0)
            correct_dict.setdefault(op, [])
            
            count_dict[op] += 1
            correct_dict[op].append(processed_example["correct_num"] / len(processed_example["reply_answers"]))                
                
        sorted_keys = sorted(count_dict.keys())
        
        for sample_num in sample_nums:
            file_sample_suffix = ""
            if not sample_num is None:
                file_sample_suffix = f"-sample-{sample_num}"
            else:
                sample_num = 1
                
            
            import os
            dir_name = "results"
            os.makedirs(dir_name, exist_ok=True)  # Create directory if it doesn't exist
            with open(f"results/result_symbolic_{args.save_name}{file_sample_suffix}.txt", "a+") as file:
                for op in sorted_keys:
                    correctnum = 0.0
                    for elem in correct_dict[op]:
                        correctnum += 1.0 - (1.0 - elem) ** sample_num
                    file.write(f"length: {length}, op: {op}, acc: {format(correctnum/count_dict[op], '.4f').rstrip('0').rstrip('.')}" + "\n")

    except Exception as e:
        print(e)
        raise

    

    




