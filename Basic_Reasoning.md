##### Copyright 2024 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Gemini API: Basic reasoning

This notebook demonstrates how to use prompting to perform reasoning tasks using the Gemini API's Python SDK. In this example, you will work through a mathematical word problem using prompting.

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/prompting/Basic_Reasoning.ipynb"><img src = "../../images/colab_logo_32px.png"/>Run in Google Colab</a>
  </td>
</table>

The Gemini API can handle many tasks that involve indirect reasoning, such as solving mathematical or logical proofs.

In this example, you will see how the LLM explains given problems step by step.


```
!pip install -U -q "google-generativeai>=0.7.2"
```


```
import google.generativeai as genai

from IPython.display import Markdown
```

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
```

## Examples

Begin by defining some system instructions that will be include when you define and choose the model.


```
system_prompt = """
You are a teacher solving mathematical and logical problems. Your task:
1. Summarize given conditions.
2. Identify the problem.
3. Provide a clear, step-by-step solution.
4. Provide an explanation for each step.

Ensure simplicity, clarity, and correctness in all steps of your explanation.
Each of your task should be done in order and seperately.
"""
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", system_instruction=system_prompt)
```

Next, you can define a logical problem such as the one below.


```
logical_problem = """
Assume a world where 1 in 5 dice are weighted and have 100% to roll a 6.
A person rolled a dice and rolled a 6.
Is it more likely that the die was weighted or not?
"""
Markdown(model.generate_content(logical_problem).text)
```




## Problem Summary:

* **Scenario:** In a world where 1 in 5 dice are weighted to always roll a 6, a person rolls a 6.
* **Question:** Is it more likely that the person rolled a weighted die or a fair die?

## Identifying the Problem:

We need to determine the probability of rolling a 6 with a weighted die versus a fair die, given that a 6 was rolled. This is a conditional probability problem.

## Solution:

Let's use Bayes' Theorem to solve this:

**P(Weighted | 6) = [P(6 | Weighted) * P(Weighted)] / P(6)**

Where:

* **P(Weighted | 6):** The probability that the die is weighted, given that a 6 was rolled.
* **P(6 | Weighted):** The probability of rolling a 6 given that the die is weighted (which is 1, or 100%).
* **P(Weighted):** The prior probability of the die being weighted (which is 1/5).
* **P(6):** The overall probability of rolling a 6 (which we need to calculate).

**Calculating P(6):**

* **P(6) = P(6 | Weighted) * P(Weighted) + P(6 | Fair) * P(Fair)**
* **P(6) = (1 * 1/5) + (1/6 * 4/5)**
* **P(6) = 1/5 + 2/15 = 1/3**

**Applying Bayes' Theorem:**

* **P(Weighted | 6) = (1 * 1/5) / (1/3)**
* **P(Weighted | 6) = 3/5 = 0.6**

## Explanation:

1. **Bayes' Theorem:**  This theorem helps us calculate the probability of an event (weighted die) given that another event has occurred (rolling a 6). 
2. **Prior Probabilities:**  We know that 1 in 5 dice are weighted (P(Weighted) = 1/5), and therefore 4 in 5 are fair (P(Fair) = 4/5).
3. **Calculating P(6):** We calculate the probability of rolling a 6 by considering the probability of rolling a 6 with a weighted die and a fair die, and weighting them by their respective probabilities.
4. **Conditional Probability:** We use the calculated P(6) and the probabilities from step 2 to apply Bayes' Theorem and calculate the probability of the die being weighted given that a 6 was rolled.

## Conclusion:

The probability of the die being weighted, given that a 6 was rolled, is 0.6 or 60%. This means it is more likely that the person rolled a weighted die than a fair die. 





```
math_problem = "Given a triangle with base b=6 and height h=8, calculate its area"
Markdown(model.generate_content(math_problem).text)
```




## 1. Summarize given conditions:

* The triangle has a base (b) of 6 units.
* The triangle has a height (h) of 8 units.

## 2. Identify the problem:

The problem asks us to calculate the area of the triangle.

## 3. Provide a clear, step-by-step solution:

**Step 1:** Recall the formula for the area of a triangle: 
Area = (1/2) * base * height 

**Step 2:** Substitute the given values into the formula:
Area = (1/2) * 6 * 8

**Step 3:** Simplify the expression:
Area = 3 * 8

**Step 4:** Calculate the final result:
Area = 24

## 4. Provide an explanation for each step:

**Step 1:** The formula for the area of a triangle is a fundamental concept in geometry. It states that the area of a triangle is equal to half the product of its base and height.

**Step 2:** We substitute the given values of base (b=6) and height (h=8) into the formula to obtain a specific equation for the area of this particular triangle.

**Step 3:** We perform the multiplication operation according to the order of operations, simplifying the expression to a single multiplication.

**Step 4:** Finally, we perform the last multiplication to arrive at the final answer, which is 24 square units. 




## Next steps

Be sure to explore other examples of prompting in the repository. Try creating your own prompts that include instructions on how to solve basic reasoning problems, or use the prompt given in this notebook as a template.
