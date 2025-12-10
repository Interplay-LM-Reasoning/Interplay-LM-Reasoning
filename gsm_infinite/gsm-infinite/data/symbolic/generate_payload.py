import random
import string

def create_forest(rng, nodes, k):
    rng.shuffle(nodes)

    N = len(nodes)
    
    edges = []
    parent = [-1] * N
    
    for i in range(k, N):
        parent[i] = rng.randint(0, i-1)
        if (parent[i] != i):
            edges.append((nodes[parent[i]], nodes[i]))
        
    return nodes, edges

def generate_output(rng, nodes, edges, operations):
    node_values = {node: None for node in nodes}
    output_list = []

    roots = set(nodes) - {e[1] for e in edges}

    for root in roots:
        val = rng.randint(0, 10)
        node_values[root] = val
        output_list.append(f'assign {root} = {val}')
    
    processed = set(roots)
    while len(processed) < len(nodes):
        for (parent, child), operation in zip(edges, operations):
            if node_values[child] is None and parent in processed:
                val = node_values[parent]
                node_values[child] = val + (1 if operation == ' + 1' else -1 if operation == ' - 1' else 0)
                output_list.append(f'assign {child} = {parent}{operation}')
                processed.add(child)
    
    # Sort the output_list based on the numeric part of the variable names
    query_value = rng.choice(list(node_values.values()))
    query_list = [node for node, value in node_values.items() if value == query_value]
    
    return output_list, (query_value, query_list), node_values


def generate_filler(nodes, edges, operations):
    output_list = []

    roots = set(nodes) - {e[1] for e in edges}

    
    processed = set(roots)
    for (parent, child), operation in zip(edges, operations):
        output_list.append(f'assign {child} = {parent}{operation}')
        processed.add(child)
    
    return output_list




class FindGraphGenerator:
    NUM_VARIABLES = 1000000
    MAX_CONSTANT = 9

    def __init__(self, seed):
        self.seed = seed
        self.rng = random.Random(seed)
        self.variables = [f'V{i:06d}' for i in range(self.NUM_VARIABLES)]

    def generate_task(self, op, N, with_solution=False):
        variables = self.rng.sample(self.variables, k=N)
        k1 = 1
        nodes, edges = create_forest(self.rng, variables[:op], k1)
        operations = self.rng.choices(['', ' + 1', ' - 1'], k=len(edges))
        output_list, query, node_values = generate_output(self.rng, nodes, edges, operations)
        if (N > op):
            nodes_filler, edges_filler = create_forest(self.rng, variables[op:], 1)
            operations_filler = self.rng.choices(['', ' + 1', ' - 1'], k=len(edges_filler))
            output_list.extend(generate_filler(nodes_filler, edges_filler, operations_filler))
        self.rng.shuffle(output_list)

        if with_solution:
            assignment_dict = {}
            stack = []
            evaluation_order = []
            solution = "First, there are such direct assignment(s):\n"
            for line in output_list:
                parts = line.split(" = ")
                variable = parts[0].split(" ")[1]
                if (variable in node_values):
                    expression = parts[1]
                    parts_expression = expression.split(" ")
                    if len(parts_expression) == 1 and not parts_expression[0].startswith("V"):
                        stack.append(variable)
                        evaluation_order.append(variable)
                        solution += f"{variable} = {parts_expression[0]};\n"
                        continue
                    elif len(parts_expression) == 1 and parts_expression[0].startswith("V"):
                        assignment_dict.setdefault(parts_expression[0], [])
                        assignment_dict[parts_expression[0]].append((variable, expression, ""))
                    elif len(parts_expression) == 3:  # Check for 3 parts
                        assignment_dict.setdefault(parts_expression[0], [])
                        assignment_dict[parts_expression[0]].append((variable, expression, f"{node_values[parts_expression[0]]} {parts_expression[1]} {parts_expression[2]} ="))
            solution+="\nFrom these direct assignment(s), we have: \n"

            while stack:
                if (stack[-1] in assignment_dict) and (assignment_dict[stack[-1]]):
                    variable, expression, evaluated_expression = assignment_dict[stack[-1]].pop()
                    stack.append(variable)
                    evaluation_order.append(variable)
                    solution+=f"{variable} = {expression} = {evaluated_expression}{node_values[variable]};\n"
                else:
                    solution+=f"No other variables can be derived from {stack[-1]};\n"
                    stack.pop()

            solution+=f"\nNow we have done all the calculations. Derived variables are:\n"
            solution_dict = []
            for variable in evaluation_order:
                solution += f"{variable} = {node_values[variable]};\n"
                if node_values[variable] == query[0]:
                    solution_dict.append(variable)
            solution += f"\nNow we have finished this problem. ANSWER:\n{', '.join(solution_dict)}\n"

            
            return output_list, query, node_values, solution
        else:
            return output_list, query, node_values




if __name__ == '__main__':
    # Example Usage:   
    N = 12  # Total number of nodes
    op = 6    # Number of operations in the solution
    generator = FindGraphGenerator(43)
    
    output_list, query, node_values, solution = generator.generate_task(op, N, with_solution=True)
    print("Output List:")
    for line in output_list:
        print(line)
    print("Query:", query)
    print("Node Values:", node_values)
    print(solution)