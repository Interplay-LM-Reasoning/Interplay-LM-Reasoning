from sympy import symbols, Eq, solve, sympify
from sympy.core.sympify import SympifyError 
from sympy import simplify 
from math import sqrt 

def solve_linear_equation(equation, variable):
    steps = []
    
    # Ensure the equation is in the form lhs = rhs
    lhs, rhs = equation.lhs, equation.rhs
    steps.append(f"Original equation: {lhs} = {rhs}")
    
    # Move all terms to the lhs
    # For simplicity, assume the equation is linear
    # Find the coefficient and constant
    coeff = lhs.coeff(variable)
    constant = lhs - coeff * variable
    
    # Step 1: Subtract constant from both sides
    step1_lhs = coeff * variable
    step1_rhs = rhs - constant
    steps.append(f"Step 1: Subtract {constant} from both sides: {coeff}*x = {rhs} - ({constant})")
    steps.append(f"Result: {coeff}*x = {step1_rhs}")
    
    # Step 2: Divide both sides by the coefficient
    solution = step1_rhs / coeff
    steps.append(f"Step 2: Divide both sides by {coeff}: x = {step1_rhs} / {coeff}")
    steps.append(f"Solution: x = {solution}")
    
    return steps, solution 

def solve_linear_equation_from_string(equation_str, variable_str='x'):
    """
    Solves a linear equation for a given variable and provides step-by-step deduction.
    
    Parameters:
    - equation_str (str): The equation as a string, e.g., "2*x + 3 = 7"
    - variable_str (str): The variable to solve for, default is 'x'
    
    Returns:
    - steps (list of str): A list of deduction steps
    - solution (sympy expression): The solution for the variable
    """
    steps = []
    
    # Define the variable
    variable = symbols(variable_str) 
    
    try:
        # Parse the equation string into a SymPy equation
        # sympify converts the string into a SymPy expression
        # Then, split into lhs and rhs based on the '=' sign
        if '=' not in equation_str:
            raise ValueError("Equation must contain an '=' sign.")
        
        lhs_str, rhs_str = equation_str.split('=', 1)
        lhs = sympify(lhs_str)
        rhs = sympify(rhs_str)
        
        equation = Eq(lhs, rhs)
        steps.append(f"Simplifying the equation: {lhs} = {rhs}") 
        
        # Ensure the equation is linear in the given variable
        if equation.lhs.as_poly(variable) is None or equation.rhs.as_poly(variable) is None:
            raise ValueError(f"The equation is not linear in {variable_str}.")
        
        # Move all terms to the lhs
        # new_eq = lhs - rhs = 0
        new_eq = Eq(lhs - rhs, 0)
        steps.append(f"Move all terms to one side: {new_eq.lhs} = {new_eq.rhs}") 
        
        # Find the coefficient and constant term
        poly = new_eq.lhs.as_poly(variable)
        coeff = poly.coeff_monomial(variable)
        constant = poly.as_expr() - coeff * variable
        
        # Step 2: Isolate the term with the variable
        step2_lhs = coeff * variable
        step2_rhs = -constant  # Because we moved the constant to the other side
        steps.append(f"Isolate the term with {variable_str}: {coeff}*{variable_str} = {-constant}") 
        
        # Step 3: Solve for the variable by dividing both sides by the coefficient
        solution = step2_rhs / coeff
        steps.append(f"Divide both sides by {coeff}: {variable_str} = {step2_rhs} / {coeff}") 
        steps.append(f"Solution: {variable_str} = {solution}") 
        
        return steps, solution
    
    except SympifyError:
        steps.append("Error: The equation string could not be parsed. Please check the syntax.")
        return steps, None
    except ValueError as ve:
        steps.append(f"Error: {ve}")
        return steps, None
    except Exception as e:
        steps.append(f"An unexpected error occurred: {e}")
        return steps, None 

def solve_quadratic_step_by_step(equation_str, variable_str='x'):
    """
    Solves a quadratic equation step-by-step and returns positive integer solutions.
    
    Parameters:
    - equation_str (str): The quadratic equation as a string, e.g., "x**2 - 5*x + 6 = 0"
    - variable_str (str): The variable in the equation, default is 'x'
    
    Returns:
    - steps (list of str): A list containing each step of the solution process.
    - solutions (list): A list of positive integer solutions if they exist.
    """
    steps = []
    solutions = []
    
    try:
        # Define the variable
        variable = symbols(variable_str)
        
        # Ensure the equation contains an '=' sign
        if '=' not in equation_str:
            steps.append("Error: Equation must contain an '=' sign.")
            return steps, solutions
        
        # Split the equation into LHS and RHS
        lhs_str, rhs_str = equation_str.split('=', 1)
        lhs = sympify(lhs_str)
        rhs = sympify(rhs_str)
        
        # Form the equation
        equation = Eq(lhs, rhs)
        steps.append(f"Original Equation: {equation}")
        
        # Move all terms to LHS to get standard form: ax^2 + bx + c = 0
        standard_eq = Eq(lhs - rhs, 0)
        steps.append(f"Step 1: Move all terms to one side to form the standard quadratic equation:")
        steps.append(f"        {standard_eq.lhs} = {standard_eq.rhs}")
        
        # Extract coefficients a, b, c
        poly = standard_eq.lhs.as_poly(variable)
        if poly is None:
            steps.append(f"Error: The equation is not a polynomial in {variable_str}.")
            return steps, solutions
        
        a = poly.coeff_monomial(variable**2)
        b = poly.coeff_monomial(variable)
        c = poly.coeff_monomial(1)
        
        steps.append(f"Step 2: Identify the coefficients:")
        steps.append(f"        a = {a}")
        steps.append(f"        b = {b}")
        steps.append(f"        c = {c}")
        
        # Calculate the discriminant D = b^2 - 4ac
        D = b**2 - 4*a*c
        steps.append(f"Step 3: Calculate the discriminant (D):")
        steps.append(f"        D = b^2 - 4ac = ({b})^2 - 4*({a})*({c}) = {D}")
        
        # Determine the nature of the roots based on the discriminant
        if D > 0:
            nature = "two distinct real roots"
        elif D == 0:
            nature = "one real root (a double root)"
        else:
            nature = "no real roots"
        steps.append(f"Step 4: Determine the nature of the roots based on D:")
        steps.append(f"        Since D = {D}, the equation has {nature}.")
        
        # If D < 0, no real solutions
        if D < 0:
            steps.append(f"Conclusion: There are no real roots to this equation.")
            return steps, solutions
        
        # Apply the quadratic formula: x = (-b ± sqrt(D)) / (2a)
        steps.append(f"Step 5: Apply the quadratic formula:")
        steps.append(f"        x = (-b ± sqrt(D)) / (2a)")
        steps.append(f"        x = (-({b}) ± sqrt({D})) / (2*{a})")
        
        # Calculate the roots
        sqrt_D = sqrt(D)
        root1 = simplify((-b + sqrt_D) / (2*a))
        root2 = simplify((-b - sqrt_D) / (2*a))
        steps.append(f"        x₁ = ({-b} + sqrt({D})) / (2*{a}) = {root1:.2f}")
        steps.append(f"        x₂ = ({-b} - sqrt({D})) / (2*{a}) = {root2:.2f}")
        
        # Simplify roots
        steps.append(f"Step 6: Simplify the roots:")
        steps.append(f"        x₁ = {root1:.2f}")
        steps.append(f"        x₂ = {root2:.2f}")
        
        # Collect the roots
        roots = [root1, root2]
        
        # Filter for positive integer solutions
        for sol in roots:
            if abs(int(sol) - sol) < 1e-5:  # Check if the solution is an integer 
                sol = int(sol) 
            if sol > 0:
                solutions.append(int(sol)) 
        
        # Conclusion
        if solutions:
            steps.append(f"Conclusion: The positive integer solution(s) is/are {solutions}.")
        else:
            steps.append(f"Conclusion: There are no positive integer solutions.")
        
        return steps, solutions
    
    except SympifyError:
        steps.append("Error: The equation string could not be parsed. Please check the syntax.")
        return steps, solutions
    except Exception as e:
        steps.append(f"An unexpected error occurred: {e}")
        return steps, solutions 

def check_equation_order(equation_str, variable_str='x'):
    """
    Checks whether the given single-variable equation is first-order or second-order.
    
    Parameters:
    - equation_str (str): The equation as a string, e.g., "2*x + 3 = 7"
    - variable_str (str): The variable to check the order for, default is 'x'
    
    Returns:
    - str: "first-order" if the equation is linear,
           "second-order" if the equation is quadratic,
           or an error message if the input is invalid.
    """
    try:
        # Define the variable
        variable = symbols(variable_str)
        
        # Ensure the equation contains an '=' sign
        if '=' not in equation_str:
            return "Error: Equation must contain an '=' sign."
        
        # Split the equation into LHS and RHS
        lhs_str, rhs_str = equation_str.split('=', 1)
        
        # Parse the LHS and RHS into SymPy expressions
        lhs = sympify(lhs_str)
        rhs = sympify(rhs_str)
        
        # Form the standard equation: lhs - rhs = 0
        standard_eq = lhs - rhs
        
        # Convert the standard equation to a polynomial in the specified variable
        poly = standard_eq.as_poly(variable)
        
        if poly is None:
            return f"Error: The equation is not a polynomial in '{variable_str}'."
        
        # Get the degree of the polynomial
        degree = poly.degree()
        
        if degree == 1:
            return "first-order"
        elif degree == 2:
            return "second-order"
        else:
            return f"Error: The equation is of degree {degree}, which is unsupported."
    
    except SympifyError:
        return "Error: The equation string could not be parsed. Please check the syntax."
    except Exception as e:
        return f"An unexpected error occurred: {e}" 

if __name__ == "__main__": 
    # Example usage
    x = symbols('x') 
    # equation = "2*x + 3 = 7" 
    equation = "2 * (3 + x) - 4 = 5" 
    # steps, solution = solve_linear_equation_from_string(equation, 'x') 
    # equation = Eq(2*x + 3, 7) 
    # steps, solution = solve_linear_equation(equation, x) 

    print(check_equation_order(equation)) 
    equation = "x**2 - 5*x + 6 = 0" 
    print(check_equation_order(equation)) 
    steps, equation = solve_quadratic_step_by_step(equation) 
    for step in steps: 
        print(step) 
