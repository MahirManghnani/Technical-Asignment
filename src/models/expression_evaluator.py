import re

class ExpressionEvaluator:
    def __init__(self):
        # Define the available mathematical operations
        self.operations = {
            'add': lambda x, y: x + y,
            'subtract': lambda x, y: x - y,
            'multiply': lambda x, y: x * y,
            'divide': lambda x, y: x / y if y != 0 else float('inf'),
            'exp': lambda x, y: x ** y,
            'greater': lambda x, y: 1 if x > y else 0
        }
    
    def evaluate(self, expression: str) -> float:
        """
        Evaluates a nested function expression and returns the result.
        Example: "divide(subtract(9362.2, 9244.9), 9244.9)" -> 0.0127
        """
        # Remove any whitespace
        expression = expression.replace(" ", "")
        
        # Base case: if it's a number, return it
        try:
            return float(expression)
        except ValueError:
            pass
        
        # Find the first complete function call (no nested functions inside)
        pattern = r'(\w+)\(([^()]+)\)'
        while True:
            match = re.search(pattern, expression)
            if not match:
                raise ValueError(f"Invalid expression format: {expression}")
            
            func_name = match.group(1)
            args = match.group(2).split(',')
            
            if func_name not in self.operations:
                raise ValueError(f"Unknown operation: {func_name}")
            
            try:
                # Convert arguments to float
                args = [float(arg.strip()) for arg in args]
                # Calculate result
                result = self.operations[func_name](*args)
                
                # Replace the function call with its result in the original expression
                expression = expression[:match.start()] + str(result) + expression[match.end():]
                
                # If no more functions, we're done
                if '(' not in expression:
                    return float(expression)
            except Exception as e:
                raise ValueError(f"Error evaluating {func_name}{tuple(args)}: {str(e)}")
