import re

def preprocess_line(line):
    replies = line["replies"]
    return replies
    # value = line["answer_q"]   
    # return [get_prompt(reply, value) for reply in replies]
    
def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False 
    
def criteriaoutput(generatedtext, inputexample): 
    correctedanswers = 0 
    totalanswers = 0 
    # parsing the answer key 
    for i in range(len(generatedtext)): 
        totalanswers += 1 
        idx_answer_start = inputexample["solution"].find("Answer: ") 
        idx_answer_end = inputexample["solution"].find(".", idx_answer_start) 
        answer_text = inputexample["solution"][idx_answer_start + len("Answer: ") : idx_answer_end] 
        answer_text = int(answer_text.lower()) 
        
        generatedtext[i] = re.sub('.\x08', 'b', generatedtext[i])
        generatedtext[i] = generatedtext[i].lower() 
        # if args.verbose and args.local_rank == 0: 
        #     print(colored(inputexample["solution"], "yellow"), flush = True) 
        #     print(colored(generatedtext[i], "cyan"), flush = True) 
        
        idx_generated_begin = -1 
        idx_generated_conclude = -1 
        keywords = ["answer: ", "solution: ", "oxed{", "**answer:** ", "**answer: ", "final answer: answer: ", "\nanswer: ", r"\text{answer: } ",  "is ", "answer: "] # updated 
        keywordsend = [".", ".", "}", ".", "**", ".", ".", None, ".", "\n"] 
        cnt = 0 
        
        while not (idx_generated_begin != -1 and idx_generated_conclude != -1) and cnt < len(keywords): 
            if keywords[cnt] in ["oxed{", "is "]: 
                idx_generated_begin = generatedtext[i].rfind(keywords[cnt]) # this relies on the generated is stopped before generated next question plus onwoards by stop 
            else: 
                idx_generated_begin = generatedtext[i].find(keywords[cnt]) 
            if idx_generated_begin != -1: 
                if keywordsend[cnt] is None: 
                    idx_generated_conclude = idx_generated_begin + len(keywords[cnt]) 
                    while generatedtext[0][idx_generated_conclude].isdigit() == True: 
                        idx_generated_conclude += 1 
                else: 
                    idx_generated_conclude = generatedtext[i].find(keywordsend[cnt], idx_generated_begin + len(keywords[cnt])) 
                if idx_generated_conclude == -1: 
                    idx_generated_conclude = len(generatedtext[i]) 
            cnt += 1 
            if not is_integer(generatedtext[i][idx_generated_begin + len(keywords[cnt - 1]) : idx_generated_conclude]): 
                idx_generated_begin = -1 
                idx_generated_conclude = -1 
                continue # if not this line, it will exit the loop 
        
        if idx_generated_begin == -1: 
            # if args.local_rank == 0: 
            #     print(colored("Answer not found", "red"), flush = True) 
            correctedanswers += 0 
            continue 
        else: 
            try: 
                answergenerated_text = int(generatedtext[i][idx_generated_begin + len(keywords[cnt - 1]) : idx_generated_conclude]) 
            except: 
                # if args.local_rank == 0: 
                #     print(colored("Answer not found", "red"), flush = True) 
                correctedanswers += 0 
                continue 
            # if args.local_rank == 0: 
            #     if answergenerated_text == answer_text: 
            #         print(colored("Answer {} expected {}".format(answergenerated_text, answer_text), "green"), flush = True) 
            #     else: 
            #         print(colored("Answer {} expected {}".format(answergenerated_text, answer_text), "red"), flush = True) 
            correctedanswers += int(answergenerated_text == answer_text) 
    return correctedanswers, totalanswers 



def postprocess_line(line, extractions):
    corrected, total = criteriaoutput(extractions, line) 
    line["correct_num"] = corrected
    line["reply_answers"] = [""] * total
    
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
    parser.add_argument('--save-dataset', type=str, help="Save dataset name", default="base")
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
    
    parser.add_argument(
        '--filter-config',
        type=json.loads,
        help='Filter configuration as a JSON string.'
    )

    args = parser.parse_args()

    if (args.num_samples == 1):
        sample_nums = [None]
    else: 
        sample_nums = [1<<i for i in range(args.num_samples.bit_length()) if (1<<i) < args.num_samples]

    length=args.length
    try:
        dir_name = "datasets"
        with open(f"{dir_name}/{args.save_dataset}-{args.save_name}_{str(length)}", 'r') as f:
            unprocessed_dataset = json.load(f)
        filter_config = args.filter_config
        if filter_config:
            filtered_datasets = []
            for config in filter_config:
                current_filter = {key: value for key, value in config.items() if key not in ["percentage"]}
                filtered_subset = [example for example in unprocessed_dataset if all(example[key] == value for key, value in current_filter.items())]
                filtered_datasets.append(filtered_subset)

            unprocessed_dataset = [example for sublist in filtered_datasets for example in sublist]
            
        def process_dataset(unprocessed_dataset, filter=None):
            results = []
            count_dict = {}
            correct_dict = {}
            
            submission_list = []
            num_samples = len(unprocessed_dataset[0]["replies"])
        
            len_dataset = len(unprocessed_dataset)
            # len_dataset = 1

            for i in range(len_dataset):
                submission_list.extend(preprocess_line(unprocessed_dataset[i]))

            extractions = submission_list
                        
            for i in range(0, len_dataset):
                results.append(postprocess_line(unprocessed_dataset[i], 
                                                [extractions[j] for j in range(i*num_samples, (i+1)*num_samples)])
                            )

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
                with open(f"results/result_{args.save_dataset}_{args.save_name}{file_sample_suffix}.txt", "a+") as file:
                    for op in sorted_keys:
                        correctnum = 0.0
                        for elem in correct_dict[op]:
                            correctnum += 1.0 - (1.0 - elem) ** sample_num
                        # file.write(f"length: {length}, op: {op}, acc: {format(correctnum/count_dict[op], '.4f').rstrip('0').rstrip('.')}" + "\n")
                        output_line = f"length: {length}, op: {op}, acc: {format(correctnum/count_dict[op], '.4f').rstrip('0').rstrip('.')}"

                        if filter:
                            output_line += f", num_examples: {len(correct_dict[op])}"
                            for key, value in filter.items():
                                output_line += f", {key}: {value}"
                        
                        output_line += "\n"
                        file.write(output_line)

        process_dataset(unprocessed_dataset)

        if filter_config:
            for config in filter_config:
                current_filter = {key: value for key, value in config.items() if key not in ["percentage"]}
                filtered_subset = [example for example in unprocessed_dataset if all(example[key] == value for key, value in current_filter.items())]
                process_dataset(filtered_subset, filter=current_filter)
    except Exception as e:
        print(e)
        raise

    

    




