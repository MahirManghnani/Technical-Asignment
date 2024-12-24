SYSTEM_PROMPT = """You are a financial analyst, and an expert in reading and performing numerical analysis on financial reports."""

INITIAL_PROMPT = """
I will provide financial data and a question about that data. Please respond with the following:

1. A mathematical function that calculates the answer using only these basic operations:
- add(x, y): Returns x + y
- subtract(x, y): Returns x - y
- multiply(x, y): Returns x * y
- divide(x, y): Returns x / y
- exp(x, y): Returns x raised to power y
- greater(x, y): Returns 1 if x > y, otherwise 0

Rules:
 - Each operation must take exactly 2 arguments
 - You can nest operations (e.g., divide(subtract(206588, 181001), 181001))
 - Use only the operations listed above - no other mathematical operations are allowed

2. Formatting Instructions for the result. Determine how best to format the number based on the question and the supporting data. The instructions should include:
 - prefix: A string to be added before the number (e.g., "$" for currency). If no prefix is needed, use an empty string "".
 - suffix: A string to be added after the number (e.g., "%" for percentages). If no suffix is needed, use an empty string "".
 - rounding: The number of decimal places to round the result to (e.g., 2 for two decimal places). Use 0 if no rounding is required.
 - multiplier: A number to multiply the result by (e.g., 100 for percentages). Use 1 if no multiplication is required.

Give the output in a JSON format.
"""

def format_question_prompt(context: dict) -> str:
    """Format the question prompt with the given context."""
    return f"""
    pre_text: 
    ```
    {context['pre_text']}
    ```
    post_text:
    ```
    {context['post_text']}
    ```
    table:
    ```
    {context['table']}
    ```
    question:
    {context['question']}
"""