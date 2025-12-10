from forward_generator import drawAll 
from reverse_generator import drawAllEquan 

import json 
import hashlib 
from tqdm import tqdm 
import multiprocessing as mp 
import random 
import numpy as np 
import argparse 
import os 

from transformers import AutoTokenizer 
from termcolor import colored 

from simple_names_three import message, messagetwo, messagethree 

generator = {
    "normalforward": drawAll, 
    "forwardreverse": drawAllEquan, 
} 

tokenizer = AutoTokenizer.from_pretrained("/netcache/huggingface/Meta-Llama-3-8B-Instruct") 

def work_function(
    op_set,
    ip_set,
    force,
    mod,
    number_range,
    target_length,
    listoperations,
    contextname,
    d,
    tokenizer,
    nums,
    identifier,
    output_dir,
):
    for mode in ["normalforward", "forwardreverse"]: 
        for template in ["crazy_zootopia", "teachers_in_school", "movie_festival_awards"]: 
            files = [] 
            num = nums[template] 
            
            oplist = listoperations 
            for op in oplist:
                # Build output path allowing custom base directory
                base_dir = output_dir if output_dir else "Igsm"
                filename = f"{base_dir}/{target_length}/{'medium' if d == 2 else 'hard'}/{op}/"
                filename += f"igsm_op{op}_ip{ip_set}_force_{force}_{identifier}.jsonl"
                files.append(filename)
                print(filename) 
            # print(num) 
            items = [[] for _ in range(len(oplist))] 
            lines = 0 
            
            np.random.seed(identifier) 
            random.seed(identifier) 
            
            while True: 
                try: 
                    problem_text, question_text, solution_text, op, id = generator[mode](
                        op_max = op_set, 
                        ip_max = ip_set, 
                        force = force, 
                        number_range = number_range, 
                        strictline = op_set, 
                        mod = mod, 
                        target_length = target_length, 
                        template = template, 
                        d = d, 
                        tokenizer = tokenizer, 
                        oplist = oplist, 
                    ) 
                except: 
                    continue 
                found = False 
                for idx, ask_op in enumerate(oplist): 
                    if op == ask_op: 
                        found == True 
                        item = { 
                            "problem": problem_text, 
                            "question": question_text, 
                            "solution": solution_text, 
                            "op": op, 
                            "id": id, 
                            "template": template, 
                            "mode": mode, 
                            "length": target_length, 
                            "d": d, 
                        } 
                        items[idx].append(item) 
                        break 
                        
                lines = np.min([len(items[idx]) for idx in range(len(oplist))]) 
                maxlines = np.max([len(items[idx]) for idx in range(len(oplist))]) 
                print(colored("{}({}){} out of {}".format(lines, [len(items[idx]) for idx in range(len(oplist))], maxlines, num), "green"), flush = True) 
                if lines > num: 
                    break 
                if found == False: 
                    continue 

            for idx, op in enumerate(oplist): 
                filename = files[idx] 
                os.makedirs(os.path.dirname(filename), exist_ok=True) 
                file = open(filename, "a") 
                file.write("\n".join([json.dumps(item) for item in items[idx]]) + "\n") 
                file.close() 

def calculate_offset(num, numprocs): 
    return (num + numprocs - 1) // numprocs 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--numprocs", type = int, default = 1) 
    parser.add_argument("--opmax", type = int, default = 15) 
    parser.add_argument("--ipmax", type = int, default = 20) 
    parser.add_argument("--force", action = "store_true") 
    parser.add_argument("--total", type = int, default = 1608) 
    parser.add_argument("--number_range", type = int, default = 23) 
    parser.add_argument("--mod", type = int, default = 23) 
    parser.add_argument("--target_length", type = str, default = None) 
    parser.add_argument("--listoperations", nargs = "+", type = int, default = [4]) 
    parser.add_argument("--d", type = int, default = 2) 
    parser.add_argument("--output_dir", type=str, default="Igsm", help="Base directory for generated files (can be absolute path)")
    args = parser.parse_args() 
    print(args) 
    
    numprocs = args.numprocs 
    processes = [] 
    op_max = args.opmax 
    ip_max = args.ipmax 
    force = args.force 
    total = args.total 
    number_range = args.number_range 
    mod = args.mod 
    target_length = args.target_length 
    
    num = (total + numprocs - 1) // numprocs 
    
    nums = {
        "crazy_zootopia": calculate_offset(args.total, args.numprocs), 
        "teachers_in_school": calculate_offset((args.total // 2), args.numprocs), 
        "movie_festival_awards": calculate_offset((args.total // 2), args.numprocs), 
    } 
    
    for i in range(numprocs):
        p = mp.Process(target=work_function, args=(op_max, ip_max, force, mod, number_range, target_length, args.listoperations, args.target_length, args.d, tokenizer, nums, i, args.output_dir))
        processes.append(p) 
        p.start() 
    
    for p in processes: 
        p.join() 
    
    print("processes joined") 

def merge_files(file1_path, file2_path, file3_path, output_path): 
    # Open files
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2, open(file3_path, 'r') as f3, open(output_path, 'w') as output:
        # Read lines from each file
        file1_lines = f1.readlines()
        file2_lines = f2.readlines()
        file3_lines = f3.readlines()

        # Indices for each file
        index1, index2, index3 = 0, 0, 0

        # Loop until we reach the end of the shortest file
        while index1 < len(file1_lines) and index2 < len(file2_lines) and index3 < len(file3_lines):
            # Add two lines from file1
            if index1 < len(file1_lines) - 1:
                output.write(file1_lines[index1])
                output.write(file1_lines[index1 + 1])
                index1 += 2
            elif index1 < len(file1_lines):  # Case when only one line is left in file1
                output.write(file1_lines[index1])
                index1 += 1

            # Add one line from file2 and file3
            if index2 < len(file2_lines):
                output.write(file2_lines[index2])
                index2 += 1
            if index3 < len(file3_lines):
                output.write(file3_lines[index3])
                index3 += 1

        # Optionally, add remaining lines from file1 if needed
        for i in range(index1, len(file1_lines)):
            output.write(file1_lines[i])

    print("Files merged successfully into", output_path) 
