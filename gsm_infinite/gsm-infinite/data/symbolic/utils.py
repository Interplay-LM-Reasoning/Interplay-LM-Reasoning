
def get_symbolic_prompt(query_value, context):
   prompt = f"<context>\n{context}\n</context>\n\nThe context contains relationships between variables. These relationships are independent mathematical equations that are all satisfied simultaneously.\nUsing only these relationships, determine which variables (if any) from which values can be derived are equal to {query_value}.\nShow your step-by-step reasoning and calculations, and then conclude your final answer in a sentence."
   # prompt = f"{context}\n**Question:**\nCan you tell me which variables are equal to {query_value} in those relationships in '<<<>>>'? These relationships are in no particular order.\nShow your step-by-step reasoning and calculations, and then conclude your final answer in a sentence."
   return  prompt

def get_symbolic_prompt_query(query_value):
   prompt = f"The context contains relationships between variables. These relationships are independent mathematical equations that are all satisfied simultaneously.\nUsing only these relationships, determine which variables (if any) from which values can be derived are equal to {query_value}.\nShow your step-by-step reasoning and calculations, and then conclude your final answer in a sentence."
   # prompt = f"{context}\n**Question:**\nCan you tell me which variables are equal to {query_value} in those relationships in '<<<>>>'? These relationships are in no particular order.\nShow your step-by-step reasoning and calculations, and then conclude your final answer in a sentence."
   return  prompt