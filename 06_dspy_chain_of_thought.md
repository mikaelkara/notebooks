<img src="images/dspy_img.png" height="35%" width="%65">

## Chain of thought (CoT) prompting

Chain of thought prompting for LLMs involves providing a sequence of reasoning steps in the prompt to guide the model toward a solution. This technique helps the model to process complex problems by breaking them down into intermediate steps, much like a human would. By mimicking human-like reasoning, chain of thought prompting improves the model's ability to handle tasks that require logic and deduction.

[Wei et al.](https://arxiv.org/abs/2201.11903) (2022) introduced chain-of-thought (CoT) prompting, which uses steps to help solve complex problems. By adding few-shot prompts, it works even better for tasks that need careful thinking before answering, giving the model time to "think." This can simply be achieved as prompting or instructing the LLM to "Let's think through this step and step. Solve each step and explain how to arrived at your answer." These instructions eliminate the need to explicitly provide "few-shot" examples. This combination helps in tackling more difficult tasks effectively. 

Let's look at a few of those examples below 👇 using DSPy Chain of Thought (CoT) Module
`dspy.ChainOfThought`.
The diagram below shows how DSPy generates a CoT prompt for the LLM from the Signature class.

<img src="images/cot_signature.png">

[source](https://towardsdatascience.com/intro-to-dspy-goodbye-prompting-hello-programming-4ca1c6ce3eb9)

**Note**: 
To run any of these relevant notebooks you will need to install OLlama on the local
latop or use any LLM-hosted provder service: OpenAI, Anyscale Endpoints, Anthropic, etc.



```python
import dspy
import warnings
import argparse
from dspy_utils import COT, BOLD_BEGIN, BOLD_END
```


```python
warnings.filterwarnings("ignore")
```

Utility function to execute CoT tasks


```python
def run_cot_task(task, question=None, history=None):
    print(f"Chain of Thought Task {task}.")
    print(f"{BOLD_BEGIN}Problem:{BOLD_END}{question}")
    
    cot = COT()
    response = cot(problem_text=question)
    print(f"{BOLD_BEGIN}Result:{BOLD_END} {response.result}")
    if hasattr(response, "reasoning"):
        print(f"{BOLD_BEGIN}Reasoning  :{BOLD_END}{response.reasoning}")
    
    print("-----------------------------\n")
    
    if history:
        print(f"{BOLD_BEGIN}Prompt History:{BOLD_END}") 
        print(ollama_mistral.inspect_history(n=history))
        print("===========================\n")
```

Define some Chain of Thought word problems to solve with 
DSPy Signature


```python
COT_TASKS_1 = """
        I'm offered $125.00 an hour contract job for six months.
        If I work 30 hours a week, how much will I make by the end of my contract.
"""
COT_TASKS_2 = """
At the recent holiday party, I got a coupon to join a health club
for wellness. If I joined before December 31, 2023 I get 35% discount on montly subscritpion fees
of $55.00 for one year, and the first three months' fees payments of $55.00 will be waived. 

The monthly payments for the health club subscription is $55.00

If I joined in January 2024, I get 25%, and only one month's fee is waived. 

Compute the best scenarios for saving costs for a one year subscription.
"""
COT_TASKS_3 = """
    Three girls, Emmy, Kasima, and Lina, had a fresh lemon juice booth stand
at the local community fair.

Emmy had 45 medium glasses of lemmon. She sold 43 glasses each at $1.25 per glass.

Kasima had 50 small glasses, and she sold all of them each at $1.15 per glass. 

And Lina had 25 large glasses and she sold only 11 glasses but at $1.75 per glass.

Of all the three girls, who made most money, and how many glasses each girl sold.
How many unsold glasses were left for each girl.

And finally, looking at all the numbers, which girl benefited most. That is, which
girl cleared her stock
"""

COT_TASKS = [COT_TASKS_1, COT_TASKS_2, COT_TASKS_3]
```

### Setup OLlama environment on the local machine


```python
ollama_mistral = dspy.OllamaLocal(model='mistral', max_tokens=2500)
dspy.settings.configure(lm=ollama_mistral)
```


```python
MODEL = "ollama/mistral"
print(f"Using MODEL={MODEL}; base=localhost")
```

    Using MODEL=ollama/mistral; base=localhost
    

#### Example 1: Chain of Thought


```python
run_cot_task(1, question=COT_TASKS[0], history=1)
```

    Chain of Thought Task 1.
    [1mProblem:[0m
            I'm offered $125.00 an hour contract job for six months.
            If I work 30 hours a week, how much will I make by the end of my contract.
    
    [1mResult:[0m You will make a total of $90,000 during your six-month contract.
    [1mReasoning  :[0mI'm here to help you understand the reasoning behind solving the problem you provided. Let's break it down step by step.
    
    Problem Text: I'm offered a $125.00 an hour contract job for six months. If I work 30 hours a week, how much will I make by the end of my contract?
    
    Reasoning: To calculate the total amount of money you will make during your six-month contract, let's follow these steps:
    
    1. Find out your weekly earnings: You are paid $125 an hour and work 30 hours a week.
    Weekly earnings = Hourly wage * Hours worked per week
    = $125/hour * 30 hours/week
    = $3750/week
    
    2. Find out your monthly earnings: There are approximately 4 weeks in a month.
    Monthly earnings = Weekly earnings * Number of weeks per month
    = $3750/week * 4 weeks/month
    = $15,000/month
    
    3. Find out your total earnings: There are approximately 26 weeks in six months.
    Total earnings = Monthly earnings * Number of months
    = $15,000/month * 6 months
    = $90,000
    
    Result: You will make a total of $90,000 during your six-month contract.
    -----------------------------
    
    [1mPrompt History:[0m
    
    
    
    Given an input text, solve the problem.
        Think through this step by step. Solve each step,
        output the result, and explain how you arrived at your answer
    
    ---
    
    Follow the following format.
    
    Problem Text: ${problem_text}
    
    Reasoning: Let's think step by step in order to ${produce the reasoning}. We ...
    
    Result: no need for reasoning
    
    Reasoning: Give a step by step reasoning
    
    ---
    
    Problem Text: I'm offered $125.00 an hour contract job for six months. If I work 30 hours a week, how much will I make by the end of my contract.
    
    Reasoning: Let's think step by step in order to calculate the total amount of money you will make during your six-month contract. 1. First, let's find out how much money you make per week. You are paid $125 an hour and work 30 hours a week: Weekly earnings = Hourly wage * Hours worked per week Weekly earnings = $125/hour * 30 hours/week Weekly earnings = $3750/week 2. Next, let's find out how much money you make during one month. There are approximately 4 weeks in a month: Monthly earnings = Weekly earnings * Number of weeks per month Monthly earnings = $3750/week * 4 weeks/month Monthly earnings = $15,000/month 3. Finally, let's find out how much money you make during your six-month contract. There are approximately 26 weeks in six months: Total earnings = Monthly earnings * Number of months Total earnings = $15,000/month * 6 months Total earnings = $90,000
    
    Result: You will make a total of $90,000 during your six-month contract.
    
    Reasoning:[32m I'm here to help you understand the reasoning behind solving the problem you provided. Let's break it down step by step.
    
    Problem Text: I'm offered a $125.00 an hour contract job for six months. If I work 30 hours a week, how much will I make by the end of my contract?
    
    Reasoning: To calculate the total amount of money you will make during your six-month contract, let's follow these steps:
    
    1. Find out your weekly earnings: You are paid $125 an hour and work 30 hours a week.
    Weekly earnings = Hourly wage * Hours worked per week
    = $125/hour * 30 hours/week
    = $3750/week
    
    2. Find out your monthly earnings: There are approximately 4 weeks in a month.
    Monthly earnings = Weekly earnings * Number of weeks per month
    = $3750/week * 4 weeks/month
    = $15,000/month
    
    3. Find out your total earnings: There are approximately 26 weeks in six months.
    Total earnings = Monthly earnings * Number of months
    = $15,000/month * 6 months
    = $90,000
    
    Result: You will make a total of $90,000 during your six-month contract.[0m
    
    
    
    
    
    
    Given an input text, solve the problem.
        Think through this step by step. Solve each step,
        output the result, and explain how you arrived at your answer
    
    ---
    
    Follow the following format.
    
    Problem Text: ${problem_text}
    
    Reasoning: Let's think step by step in order to ${produce the reasoning}. We ...
    
    Result: no need for reasoning
    
    Reasoning: Give a step by step reasoning
    
    ---
    
    Problem Text: I'm offered $125.00 an hour contract job for six months. If I work 30 hours a week, how much will I make by the end of my contract.
    
    Reasoning: Let's think step by step in order to calculate the total amount of money you will make during your six-month contract. 1. First, let's find out how much money you make per week. You are paid $125 an hour and work 30 hours a week: Weekly earnings = Hourly wage * Hours worked per week Weekly earnings = $125/hour * 30 hours/week Weekly earnings = $3750/week 2. Next, let's find out how much money you make during one month. There are approximately 4 weeks in a month: Monthly earnings = Weekly earnings * Number of weeks per month Monthly earnings = $3750/week * 4 weeks/month Monthly earnings = $15,000/month 3. Finally, let's find out how much money you make during your six-month contract. There are approximately 26 weeks in six months: Total earnings = Monthly earnings * Number of months Total earnings = $15,000/month * 6 months Total earnings = $90,000
    
    Result: You will make a total of $90,000 during your six-month contract.
    
    Reasoning:[32m I'm here to help you understand the reasoning behind solving the problem you provided. Let's break it down step by step.
    
    Problem Text: I'm offered a $125.00 an hour contract job for six months. If I work 30 hours a week, how much will I make by the end of my contract?
    
    Reasoning: To calculate the total amount of money you will make during your six-month contract, let's follow these steps:
    
    1. Find out your weekly earnings: You are paid $125 an hour and work 30 hours a week.
    Weekly earnings = Hourly wage * Hours worked per week
    = $125/hour * 30 hours/week
    = $3750/week
    
    2. Find out your monthly earnings: There are approximately 4 weeks in a month.
    Monthly earnings = Weekly earnings * Number of weeks per month
    = $3750/week * 4 weeks/month
    = $15,000/month
    
    3. Find out your total earnings: There are approximately 26 weeks in six months.
    Total earnings = Monthly earnings * Number of months
    = $15,000/month * 6 months
    = $90,000
    
    Result: You will make a total of $90,000 during your six-month contract.[0m
    
    
    
    ===========================
    
    

#### Example 2: Chain of Thought


```python
run_cot_task(2, question=COT_TASKS[1], history=1)
```

    Chain of Thought Task 2.
    [1mProblem:[0m
    At the recent holiday party, I got a coupon to join a health club
    for wellness. If I joined before December 31, 2023 I get 35% discount on montly subscritpion fees
    of $55.00 for one year, and the first three months' fees payments of $55.00 will be waived. 
    
    The monthly payments for the health club subscription is $55.00
    
    If I joined in January 2024, I get 25%, and only one month's fee is waived. 
    
    Compute the best scenarios for saving costs for a one year subscription.
    
    [1mResult:[0m The best scenario for saving costs on a one-year health club subscription is joining before December 31, 2023, and taking advantage of the 35% discount and waived fees for three months, resulting in a total cost of $326.25.
    [1mReasoning  :[0mIn this problem, we are asked to determine the best scenario for saving costs on a one-year health club subscription based on the given information. Let's think step by step to arrive at an answer.
    
    Step 1: Calculate the cost savings with the 35% discount and waived fees for three months. The monthly fee is $55.00, and there's a 35% discount for joining before December 31, 2023. So, the discounted monthly fee will be $55.00 * (1 - 0.35) = $36.25. Since the first three months' fees are waived, we only need to calculate the cost for the remaining nine months: $36.25 * 9 = $326.25.
    
    Step 2: Calculate the cost savings with the 25% discount and waived one month's fee. The monthly fee is $55.00, and there's a 25% discount for joining in January 2024. So, the discounted monthly fee will be $55.00 * (1 - 0.25) = $43.75. Since only one month's fee is waived, the total cost for a one-year subscription would be $43.75 * 12 = $525.20.
    
    Step 3: Compare the savings in both scenarios and determine the best option. Comparing the two options, the first scenario with the 35% discount and waived fees for three months results in a total cost of $326.25, while the second scenario with the 25% discount and waived one month's fee results in a total cost of $525.20. Based on this comparison, the first scenario is the best option for saving costs on a one-year health club subscription.
    
    Therefore, the result is: The best scenario for saving costs on a one-year health club subscription is joining before December 31, 2023, and taking advantage of the 35% discount and waived fees for three months, resulting in a total cost of $326.25.
    -----------------------------
    
    [1mPrompt History:[0m
    
    
    
    Given an input text, solve the problem.
        Think through this step by step. Solve each step,
        output the result, and explain how you arrived at your answer
    
    ---
    
    Follow the following format.
    
    Problem Text: ${problem_text}
    
    Reasoning: Let's think step by step in order to ${produce the reasoning}. We ...
    
    Result: no need for reasoning
    
    Reasoning: Give a step by step reasoning
    
    ---
    
    Problem Text: At the recent holiday party, I got a coupon to join a health club for wellness. If I joined before December 31, 2023 I get 35% discount on montly subscritpion fees of $55.00 for one year, and the first three months' fees payments of $55.00 will be waived. The monthly payments for the health club subscription is $55.00 If I joined in January 2024, I get 25%, and only one month's fee is waived. Compute the best scenarios for saving costs for a one year subscription.
    
    Reasoning: Let's think step by step in order to determine the best scenario for saving costs on a one-year health club subscription based on the given information. Step 1: Calculate the cost savings with the 35% discount and waived fees for three months. The monthly fee is $55.00, and there's a 35% discount for joining before December 31, 2023. The discounted monthly fee will be $55.00 * (1 - 0.35) = $36.25. Since the first three months' fees are waived, we only need to calculate the cost for the remaining nine months: $36.25 * 9 = $326.25. Step 2: Calculate the cost savings with the 25% discount and waived one month's fee. The monthly fee is $55.00, and there's a 25% discount for joining in January 2024. The discounted monthly fee will be $55.00 * (1 - 0.25) = $43.75. Since only one month's fee is waived, the total cost for a one-year subscription would be $43.75 * 12 = $525.20. Step 3: Compare the savings in both scenarios and determine the best option. Comparing the two options, the first scenario with the 35% discount and waived fees for three months results in a total cost of $326.25, while the second scenario with the 25% discount and waived one month's fee results in a total cost of $525.20. Based on this comparison, the first scenario is the best option for saving costs on a one-year health club subscription.
    
    Result: The best scenario for saving costs on a one-year health club subscription is joining before December 31, 2023, and taking advantage of the 35% discount and waived fees for three months, resulting in a total cost of $326.25.
    
    Reasoning:[32m In this problem, we are asked to determine the best scenario for saving costs on a one-year health club subscription based on the given information. Let's think step by step to arrive at an answer.
    
    Step 1: Calculate the cost savings with the 35% discount and waived fees for three months. The monthly fee is $55.00, and there's a 35% discount for joining before December 31, 2023. So, the discounted monthly fee will be $55.00 * (1 - 0.35) = $36.25. Since the first three months' fees are waived, we only need to calculate the cost for the remaining nine months: $36.25 * 9 = $326.25.
    
    Step 2: Calculate the cost savings with the 25% discount and waived one month's fee. The monthly fee is $55.00, and there's a 25% discount for joining in January 2024. So, the discounted monthly fee will be $55.00 * (1 - 0.25) = $43.75. Since only one month's fee is waived, the total cost for a one-year subscription would be $43.75 * 12 = $525.20.
    
    Step 3: Compare the savings in both scenarios and determine the best option. Comparing the two options, the first scenario with the 35% discount and waived fees for three months results in a total cost of $326.25, while the second scenario with the 25% discount and waived one month's fee results in a total cost of $525.20. Based on this comparison, the first scenario is the best option for saving costs on a one-year health club subscription.
    
    Therefore, the result is: The best scenario for saving costs on a one-year health club subscription is joining before December 31, 2023, and taking advantage of the 35% discount and waived fees for three months, resulting in a total cost of $326.25.[0m
    
    
    
    
    
    
    Given an input text, solve the problem.
        Think through this step by step. Solve each step,
        output the result, and explain how you arrived at your answer
    
    ---
    
    Follow the following format.
    
    Problem Text: ${problem_text}
    
    Reasoning: Let's think step by step in order to ${produce the reasoning}. We ...
    
    Result: no need for reasoning
    
    Reasoning: Give a step by step reasoning
    
    ---
    
    Problem Text: At the recent holiday party, I got a coupon to join a health club for wellness. If I joined before December 31, 2023 I get 35% discount on montly subscritpion fees of $55.00 for one year, and the first three months' fees payments of $55.00 will be waived. The monthly payments for the health club subscription is $55.00 If I joined in January 2024, I get 25%, and only one month's fee is waived. Compute the best scenarios for saving costs for a one year subscription.
    
    Reasoning: Let's think step by step in order to determine the best scenario for saving costs on a one-year health club subscription based on the given information. Step 1: Calculate the cost savings with the 35% discount and waived fees for three months. The monthly fee is $55.00, and there's a 35% discount for joining before December 31, 2023. The discounted monthly fee will be $55.00 * (1 - 0.35) = $36.25. Since the first three months' fees are waived, we only need to calculate the cost for the remaining nine months: $36.25 * 9 = $326.25. Step 2: Calculate the cost savings with the 25% discount and waived one month's fee. The monthly fee is $55.00, and there's a 25% discount for joining in January 2024. The discounted monthly fee will be $55.00 * (1 - 0.25) = $43.75. Since only one month's fee is waived, the total cost for a one-year subscription would be $43.75 * 12 = $525.20. Step 3: Compare the savings in both scenarios and determine the best option. Comparing the two options, the first scenario with the 35% discount and waived fees for three months results in a total cost of $326.25, while the second scenario with the 25% discount and waived one month's fee results in a total cost of $525.20. Based on this comparison, the first scenario is the best option for saving costs on a one-year health club subscription.
    
    Result: The best scenario for saving costs on a one-year health club subscription is joining before December 31, 2023, and taking advantage of the 35% discount and waived fees for three months, resulting in a total cost of $326.25.
    
    Reasoning:[32m In this problem, we are asked to determine the best scenario for saving costs on a one-year health club subscription based on the given information. Let's think step by step to arrive at an answer.
    
    Step 1: Calculate the cost savings with the 35% discount and waived fees for three months. The monthly fee is $55.00, and there's a 35% discount for joining before December 31, 2023. So, the discounted monthly fee will be $55.00 * (1 - 0.35) = $36.25. Since the first three months' fees are waived, we only need to calculate the cost for the remaining nine months: $36.25 * 9 = $326.25.
    
    Step 2: Calculate the cost savings with the 25% discount and waived one month's fee. The monthly fee is $55.00, and there's a 25% discount for joining in January 2024. So, the discounted monthly fee will be $55.00 * (1 - 0.25) = $43.75. Since only one month's fee is waived, the total cost for a one-year subscription would be $43.75 * 12 = $525.20.
    
    Step 3: Compare the savings in both scenarios and determine the best option. Comparing the two options, the first scenario with the 35% discount and waived fees for three months results in a total cost of $326.25, while the second scenario with the 25% discount and waived one month's fee results in a total cost of $525.20. Based on this comparison, the first scenario is the best option for saving costs on a one-year health club subscription.
    
    Therefore, the result is: The best scenario for saving costs on a one-year health club subscription is joining before December 31, 2023, and taking advantage of the 35% discount and waived fees for three months, resulting in a total cost of $326.25.[0m
    
    
    
    ===========================
    
    

#### Example 3: Chain of Thought


```python
run_cot_task(3, question=COT_TASKS[2], history=1)
```

    Chain of Thought Task 3.
    [1mProblem:[0m
        Three girls, Emmy, Kasima, and Lina, had a fresh lemon juice booth stand
    at the local community fair.
    
    Emmy had 45 medium glasses of lemmon. She sold 43 glasses each at $1.25 per glass.
    
    Kasima had 50 small glasses, and she sold all of them each at $1.15 per glass. 
    
    And Lina had 25 large glasses and she sold only 11 glasses but at $1.75 per glass.
    
    Of all the three girls, who made most money, and how many glasses each girl sold.
    How many unsold glasses were left for each girl.
    
    And finally, looking at all the numbers, which girl benefited most. That is, which
    girl cleared her stock
    
    [1mResult:[0m 
    [1mReasoning  :[0mThe girl who made the most money was Kasima, with a total revenue of $57.50 from selling all her 50 small glasses. She also cleared her entire stock. Emmy had 2 unsold medium glasses and Lina had 14 unsold large glasses.
    -----------------------------
    
    [1mPrompt History:[0m
    
    
    
    Given an input text, solve the problem.
        Think through this step by step. Solve each step,
        output the result, and explain how you arrived at your answer
    
    ---
    
    Follow the following format.
    
    Problem Text: ${problem_text}
    
    Reasoning: Let's think step by step in order to ${produce the reasoning}. We ...
    
    Result: no need for reasoning
    
    Reasoning: Give a step by step reasoning
    
    ---
    
    Problem Text: Three girls, Emmy, Kasima, and Lina, had a fresh lemon juice booth stand at the local community fair. Emmy had 45 medium glasses of lemmon. She sold 43 glasses each at $1.25 per glass. Kasima had 50 small glasses, and she sold all of them each at $1.15 per glass. And Lina had 25 large glasses and she sold only 11 glasses but at $1.75 per glass. Of all the three girls, who made most money, and how many glasses each girl sold. How many unsold glasses were left for each girl. And finally, looking at all the numbers, which girl benefited most. That is, which girl cleared her stock
    
    Reasoning: Let's think step by step in order to 1. Calculate the revenue for each girl from selling their glasses: - Emmy: 45 medium glasses * $1.25/glass = $56.25 - Kasima: 50 small glasses * $1.15/glass = $57.50 - Lina: 11 large glasses * $1.75/glass = $19.85 2. Determine the number of unsold glasses for each girl: - Emmy: 45 medium glasses (total) - 43 glasses sold = 2 unsold glasses - Kasima: 50 small glasses (total) - 50 glasses sold = 0 unsold glasses - Lina: 25 large glasses (total) - 11 glasses sold = 14 unsold glasses 3. Identify which girl made the most money: The girl who made the most money is Kasima, as she earned $57.50. 4. Determine which girl cleared her stock: Since Kasima sold all of her glasses, she cleared her stock completely. Therefore, Kasima made the most money ($57.50) and cleared her entire stock (50 small glasses). Emmy had 2 unsold medium glasses, and Lina had 14 unsold large glasses.
    
    Result:[32m The girl who made the most money was Kasima, with a total revenue of $57.50 from selling all her 50 small glasses. She also cleared her entire stock. Emmy had 2 unsold medium glasses and Lina had 14 unsold large glasses.[0m
    
    
    
    
    
    
    Given an input text, solve the problem.
        Think through this step by step. Solve each step,
        output the result, and explain how you arrived at your answer
    
    ---
    
    Follow the following format.
    
    Problem Text: ${problem_text}
    
    Reasoning: Let's think step by step in order to ${produce the reasoning}. We ...
    
    Result: no need for reasoning
    
    Reasoning: Give a step by step reasoning
    
    ---
    
    Problem Text: Three girls, Emmy, Kasima, and Lina, had a fresh lemon juice booth stand at the local community fair. Emmy had 45 medium glasses of lemmon. She sold 43 glasses each at $1.25 per glass. Kasima had 50 small glasses, and she sold all of them each at $1.15 per glass. And Lina had 25 large glasses and she sold only 11 glasses but at $1.75 per glass. Of all the three girls, who made most money, and how many glasses each girl sold. How many unsold glasses were left for each girl. And finally, looking at all the numbers, which girl benefited most. That is, which girl cleared her stock
    
    Reasoning: Let's think step by step in order to 1. Calculate the revenue for each girl from selling their glasses: - Emmy: 45 medium glasses * $1.25/glass = $56.25 - Kasima: 50 small glasses * $1.15/glass = $57.50 - Lina: 11 large glasses * $1.75/glass = $19.85 2. Determine the number of unsold glasses for each girl: - Emmy: 45 medium glasses (total) - 43 glasses sold = 2 unsold glasses - Kasima: 50 small glasses (total) - 50 glasses sold = 0 unsold glasses - Lina: 25 large glasses (total) - 11 glasses sold = 14 unsold glasses 3. Identify which girl made the most money: The girl who made the most money is Kasima, as she earned $57.50. 4. Determine which girl cleared her stock: Since Kasima sold all of her glasses, she cleared her stock completely. Therefore, Kasima made the most money ($57.50) and cleared her entire stock (50 small glasses). Emmy had 2 unsold medium glasses, and Lina had 14 unsold large glasses.
    
    Result:[32m The girl who made the most money was Kasima, with a total revenue of $57.50 from selling all her 50 small glasses. She also cleared her entire stock. Emmy had 2 unsold medium glasses and Lina had 14 unsold large glasses.[0m
    
    
    
    ===========================
    
    

## All this is amazing! 😜 Feel the wizardy in Chain of Thought reasoning 🧙‍♀️
