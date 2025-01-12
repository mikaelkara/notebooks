<img src="./images/llm_prompt_req_resp.png" height="35%" width="%65">

## Chain of thought (CoT) prompting

Chain of thought prompting for LLMs involves providing a sequence of reasoning steps in the prompt to guide the model toward a solution. This technique helps the model to process complex problems by breaking them down into intermediate steps, much like a human would. By mimicking human-like reasoning, chain of thought prompting improves the model's ability to handle tasks that require logic and deduction.

[Wei et al.](https://arxiv.org/abs/2201.11903) (2022) introduced chain-of-thought (CoT) prompting, which uses steps to help solve complex problems. By adding few-shot prompts, it works even better for tasks that need careful thinking before answering, giving the model time to "think." This can simply be achieved as prompting or instructing the LLM to "Let's think through this step and step. Solve each step and explain how to arrived at your answer." These instructions eliminate the need to explicitly provide "few-shot" examples. This combination helps in tackling more difficult tasks effectively. 

Let's look at a few of those examples below 👇

**Note**: 
To run any of these relevant notebooks you will need an account on Anyscale Endpoints, Anthropic, or OpenAI, depending on what model you elect, along with the respective environment file. Use the template environment files to create respective `.env` file for either Anyscale Endpoints, Anthropic, or OpenAI.


```python
import warnings
import os

import openai
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
```

Load the environment


```python
_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings('ignore')
openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("MODEL")
print(f"Using MODEL={MODEL}; base={openai.api_base}")
```

    Using MODEL=gpt-4-1106-preview; base=https://api.openai.com/v1
    


```python
# create the OpenAI client, which can be used transparently with Anyscale Endpoints too

from openai import OpenAI

client = OpenAI(
    api_key = openai.api_key,
    base_url = openai.api_base
)
```


```python
# Utility function to send and fetch response

def get_commpletion(clnt: object, model: str, system_content: str, user_content:str) -> str:
    chat_completion = clnt.chat.completions.create(
        model=model,
    messages=[{"role": "system", "content": system_content},
              {"role": "user", "content": user_content}],
    temperature = 0.8)

    response = chat_completion.choices[0].message.content
    return response
```


```python
system_content = """You are supreme repository of knowledge and an engine
of reason. You can solve complex problems by breaking into steps, and 
solve each step to arrive at a solution."""
```

#### Example 1: Chain of Thought


```python
user_content = """I'm offered $125.00 an hour contract job for six months.
If I work 30 hours a week, how much will I make by the end of my contract.

Let's think through this step by step. Solve each step and explain how you arrived 
at your answer."""

```


```python
response = get_commpletion(client, MODEL, system_content, user_content)
print(f"{response}\n")
```

    Sure, let's break down the problem into manageable steps to arrive at the solution.
    
    Step 1: Calculate weekly earnings
    First, we need to calculate how much you would earn in a week.
    If you are offered $125.00 per hour and you work 30 hours a week, you would calculate your weekly earnings as follows:
    Hourly rate x Hours per week = Weekly earnings
    $125.00/hour x 30 hours/week = $3,750.00 per week
    
    Step 2: Calculate total earnings for the duration of the contract
    Since the contract is for six months, we need to find out how many weeks are in six months. Typically, a month is considered to have about 4.33 weeks on average (since a year has 52 weeks and is divided by 12 months).
    6 months x 4.33 weeks/month = 25.98 weeks
    
    However, to be more precise, we would usually count the actual number of weeks or use a consistent average of 4 weeks per month for simpler calculations.
    
    For precision, let's use the 4.33 weeks per month average:
    6 months x 4.33 weeks/month ≈ 25.98 weeks
    Since you can't work a fraction of a week, we can either round this number to the nearest whole number or simply use it as is for an approximation. Let's use it as it is for now.
    
    Now, we calculate the total earnings for 25.98 weeks:
    Weekly earnings x Number of weeks = Total earnings
    $3,750.00/week x 25.98 weeks ≈ $97,425.00
    
    Step 3: Conclusion
    By the end of your six-month contract, working 30 hours a week at $125.00 an hour, you would approximately earn $97,425.00. Keep in mind that this is an approximation, as the exact number could vary slightly based on the actual number of workdays in those six months.
    
    

#### Example 2: Chain of Thought


```python
user_content = """At the recent holiday party, I got a coupon to join a health club
for wellness. If I joined before December 31, 2023 I get 35% discount on montly subscritpion fees
of $55.00 for one year, and the first three months' fees payments of $55.00 will be waived. 

The monthly payments for the health club subscription is $55.00

If I joined in January 2024, I get 25%, and only one month's fee is waived. 

Compute the best scenarios for saving costs for a one year subscription.

Let's think through this step by step. Solve each step and explain how you arrived 
at your answer.
"""
```


```python
response = get_commpletion(client, MODEL, system_content, user_content)
print(f"{response}\n")
```

    Let's calculate the cost of the health club subscription for one year under both scenarios:
    
    Scenario 1: Join before December 31, 2023
    1. Calculate the monthly fee with a 35% discount.
    2. Multiply the discounted monthly fee by the number of months you would pay for (12 months minus the 3 waived months).
    3. Sum up the total payment for the year.
    
    Scenario 2: Join in January 2024
    1. Calculate the monthly fee with a 25% discount.
    2. Multiply the discounted monthly fee by the number of months you would pay for (12 months minus the 1 waived month).
    3. Sum up the total payment for the year.
    
    Now, let's do the math.
    
    Scenario 1:
    1. Calculate the monthly fee with a 35% discount:
       $55.00 - (35/100 * $55.00) = $55.00 - $19.25 = $35.75 per month after discount.
    
    2. Since the first three months are waived, you would only pay for 9 months:
       $35.75 * 9 months = $321.75 for the year.
    
    Scenario 2:
    1. Calculate the monthly fee with a 25% discount:
       $55.00 - (25/100 * $55.00) = $55.00 - $13.75 = $41.25 per month after discount.
    
    2. Since one month is waived, you would only pay for 11 months:
       $41.25 * 11 months = $453.75 for the year.
    
    Comparing the costs:
    - Scenario 1: $321.75 for the year.
    - Scenario 2: $453.75 for the year.
    
    By joining before December 31, 2023, you would save:
    $453.75 - $321.75 = $132.00
    
    So, Scenario 1 offers the best savings for a one-year subscription to the health club.
    
    

#### Example 3: Chain of Thought


```python
user_content = """Three girls, Emmy, Kasima, and Lina, had a fresh lemon juice booth stand
at the local community fair.

Emmy had 45 medium glasses of lemmon. She sold 43 glasses each at $1.25 per glass.

Kasima had 50 small glasses, and she sold all of them each at $1.15 per glass. 

And Lina had 25 large glasses and she sold only 11 glasses but at $1.75 per glass.

Of all the three girls, who made most money, and how many glasses each girl sold.
How many unsold glasses were left for each girl.

And finally, looking at all the numbers, which girl benefited most. That is, which
girl cleared her stock.

Let's think through this step and step. Solve each step and explain how you arrived
at your answer"""
```


```python
response = get_commpletion(client, MODEL, system_content, user_content)
print(f"{response}\n")
```

    Sure, we can calculate the earnings for each girl, determine who made the most money, the number of glasses each sold, the number of unsold glasses, and finally, discuss who benefited most in terms of clearing their stock. Let's tackle this step by step.
    
    1. **Emmy's Earnings:**
    Emmy sold 43 out of her 45 medium glasses of lemon juice.
    - Each glass was sold at $1.25.
    - Her earnings would be: 43 glasses * $1.25/glass = $53.75.
    
    Emmy has 45 - 43 = 2 unsold glasses left.
    
    2. **Kasima's Earnings:**
    Kasima sold all of her 50 small glasses.
    - Each glass was sold at $1.15.
    - Her earnings would be: 50 glasses * $1.15/glass = $57.50.
    
    Kasima has 50 - 50 = 0 unsold glasses left.
    
    3. **Lina's Earnings:**
    Lina sold 11 out of her 25 large glasses.
    - Each glass was sold at $1.75.
    - Her earnings would be: 11 glasses * $1.75/glass = $19.25.
    
    Lina has 25 - 11 = 14 unsold glasses left.
    
    Now, let's compare their earnings to see who made the most money:
    - Emmy earned $53.75.
    - Kasima earned $57.50.
    - Lina earned $19.25.
    
    **Kasima** made the most money at the fair with $57.50.
    
    Next, to determine which girl benefited most in terms of clearing their stock, we look at the number of unsold glasses:
    - Emmy had 2 unsold glasses.
    - Kasima had 0 unsold glasses.
    - Lina had 14 unsold glasses.
    
    **Kasima** clearly benefited the most in terms of clearing her stock, as she sold all of her glasses and had none left unsold.
    
    

## All this is amazing! 😜 Feel the wizardy in Chain of Thought reasoning 🧙‍♀️
