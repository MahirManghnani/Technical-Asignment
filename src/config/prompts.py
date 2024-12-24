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

Example:
pre_text:
```
['26 | 2009 annual report in fiscal 2008 , revenues in the credit union systems and services business segment increased 14% ( 14 % ) from fiscal 2007 .', 'all revenue components within the segment experienced growth during fiscal 2008 .', 'license revenue generated the largest dollar growth in revenue as episys ae , our flagship core processing system aimed at larger credit unions , experienced strong sales throughout the year .', 'support and service revenue , which is the largest component of total revenues for the credit union segment , experienced 34 percent growth in eft support and 10 percent growth in in-house support .', 'gross profit in this business segment increased $ 9344 in fiscal 2008 compared to fiscal 2007 , due primarily to the increase in license revenue , which carries the highest margins .', 'liquidity and capital resources we have historically generated positive cash flow from operations and have generally used funds generated from operations and short-term borrowings on our revolving credit facility to meet capital requirements .', 'we expect this trend to continue in the future .', 'the company 2019s cash and cash equivalents increased to $ 118251 at june 30 , 2009 from $ 65565 at june 30 , 2008 .', 'the following table summarizes net cash from operating activities in the statement of cash flows : 2009 2008 2007 .']
```
post_text:
```['year ended june 30 , cash provided by operations increased $ 25587 to $ 206588 for the fiscal year ended june 30 , 2009 as compared to $ 181001 for the fiscal year ended june 30 , 2008 .', 'this increase is primarily attributable to a decrease in receivables compared to the same period a year ago of $ 21214 .', 'this decrease is largely the result of fiscal 2010 annual software maintenance billings being provided to customers earlier than in the prior year , which allowed more cash to be collected before the end of the fiscal year than in previous years .', 'further , we collected more cash overall related to revenues that will be recognized in subsequent periods in the current year than in fiscal 2008 .', 'cash used in investing activities for the fiscal year ended june 2009 was $ 59227 and includes $ 3027 in contingent consideration paid on prior years 2019 acquisitions .', 'cash used in investing activities for the fiscal year ended june 2008 was $ 102148 and includes payments for acquisitions of $ 48109 , plus $ 1215 in contingent consideration paid on prior years 2019 acquisitions .', 'capital expenditures for fiscal 2009 were $ 31562 compared to $ 31105 for fiscal 2008 .', 'cash used for software development in fiscal 2009 was $ 24684 compared to $ 23736 during the prior year .', 'net cash used in financing activities for the current fiscal year was $ 94675 and includes the repurchase of 3106 shares of our common stock for $ 58405 , the payment of dividends of $ 26903 and $ 13489 net repayment on our revolving credit facilities .', 'cash used in financing activities was partially offset by proceeds of $ 3773 from the exercise of stock options and the sale of common stock ( through the employee stock purchase plan ) and $ 348 excess tax benefits from stock option exercises .', 'during fiscal 2008 , net cash used in financing activities for the fiscal year was $ 101905 and includes the repurchase of 4200 shares of our common stock for $ 100996 , the payment of dividends of $ 24683 and $ 429 net repayment on our revolving credit facilities .', 'cash used in financing activities was partially offset by proceeds of $ 20394 from the exercise of stock options and the sale of common stock and $ 3809 excess tax benefits from stock option exercises .', 'beginning during fiscal 2008 , us financial markets and many of the largest us financial institutions have been shaken by negative developments in the home mortgage industry and the mortgage markets , and particularly the markets for subprime mortgage-backed securities .', 'since that time , these and other such developments have resulted in a broad , global economic downturn .', 'while we , as is the case with most companies , have experienced the effects of this downturn , we have not experienced any significant issues with our current collection efforts , and we believe that any future impact to our liquidity will be minimized by cash generated by recurring sources of revenue and due to our access to available lines of credit. .']
```
table:
```
[['2008', 'year ended june 30 2009 2008', 'year ended june 30 2009 2008', 'year ended june 30 2009'], ['net income', '$ 103102', '$ 104222', '$ 104681'], ['non-cash expenses', '74397', '70420', '56348'], ['change in receivables', '21214', '-2913 ( 2913 )', '-28853 ( 28853 )'], ['change in deferred revenue', '21943', '5100', '24576'], ['change in other assets and liabilities', '-14068 ( 14068 )', '4172', '17495'], ['net cash from operating activities', '$ 206588', '$ 181001', '$ 174247']]
```
question:
what was the percentage change in the net cash from operating activities from 2008 to 2009

{
    "formula": "divide(subtract(206588, 181001), 181001)",
    "formatting_instructions": {
        "prefix": "",
        "suffix": "%",
        "rounding": 2,
        "multiplier": 100
        }
}
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