Meta AI released a series of [Code Llama models](https://ai.meta.com/blog/code-llama-large-language-model-coding/). 

Using any of the model series shows the path used for its respective
training.

<img src="images/meta_code_llama_series.png" height="30%" width="60%">

This notebook shows some cases how to prompt the LLM to generate 
Python and SQL code using the Anyscale Endpoints.

<img src="images/codllama70b.png"  height="25%" width="50%">

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

    Using MODEL=codellama/CodeLlama-70b-Instruct-hf; base=https://api.endpoints.anyscale.com/v1
    


```python
# Create the OpenAI client, which can be used transparently with Anyscale 
# Endpoints too

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
BOLD_BEGIN = "\033[1m"
BOLD_END = "\033[0m"
```

#### Example 1: Generate a Python code to compute the value of PI

Use the [CO-STAR](https://towardsdatascience.com/how-i-won-singapores-gpt-4-prompt-engineering-competition-34c195a93d41) framework for prompting

1. **Context** - provide the background
2. **Objective** (or Task) - define the task to be performed
3. **Style** - instruct a writing style. Kind of sentences; formal, informal, magazine sytle, colloqiual, or allude to a know style.
4. **Audience** - who's it for?
5. **Response** - format, Text, Python, SQL, JSON, etc 



```python
system_content = """You are supreme repository of knowledge, and assistant
code Co-pilot for developer to assist them in generating sample code, inclding
in langauges such as Python, JavaScript, Java, C, C++, and shell script.
"""
```


```python
user_prompt = """
# CONTEXT #
I want to generate Python code to showcase our product feature for
serving open source large language models such as Code Llama series
on Anyscale Endpoints. The product feature is Anyscale Endpoints, which 
serves all Llama series models and the Mistral series too. For this
instance, we want to show case Code Llama 70B just released by Meta AI, and 
now hosted by Anyscale Endpoints

#############

# OBJECTIVE # 
Create a Python code to compute the value of PI using monte carlo method.

#############

# STYLE #
Use the Google style of formating Python code.

#############

# AUDIENCE #
Your code should be well commented and aimed at both beginner and
intermediate Python programers. 

#############

# RESPONSE #
Generate the entire code that can be easily copy and pasted into file: 'compute_pi.py'. Also,
provide specific instructions how to compile the code, run it, and any 
additional python packages it needs.
"""
```


```python
response = get_commpletion(client, MODEL, system_content, user_prompt)
print(f"\n{BOLD_BEGIN}Answer - Python code{BOLD_END}:\n {response}")
```

    
    [1mAnswer - Python code[0m:
     
    ```python
    #!/usr/bin/env python
    
    # compute_pi.py
    # by Anyscale Team
    # This Python script computes the value of PI using the Monte Carlo method.
    # It's aimed at beginners and intermediate Python developers.
    
    # Let's import the necessary Python modules
    # We'll use random to generate random points and math to compute the value of PI
    import random
    import math
    
    # We'll also import pytest to run tests on our code
    # Skip this if you're not familiar with pytest
    import pytest
    
    # This function generates a random point on a unit square
    def random_point():
        x = random.random()
        y = random.random()
        return x, y
    
    # This function checks if a point lies inside a circle with a radius of 1
    def point_inside_circle(point):
        x, y = point
        return x**2 + y**2 < 1
    
    # This function computes the value of PI using the Monte Carlo method
    # It takes the number of iterations as input
    def compute_pi(n_iterations):
        n_inside_circle = 0
    
        for _ in range(n_iterations):
            point = random_point()
            if point_inside_circle(point):
                n_inside_circle += 1
    
        pi = 4 * n_inside_circle / n_iterations
        return pi
    
    # This function tests the accuracy of our 'compute_pi()' function
    # It checks if the computed value of PI lies within 1% of the actual value
    def test_compute_pi():
        n_iterations = 1000000
        computed_pi = compute_pi(n_iterations)
        assert abs(computed_pi - math.pi) < 0.01 * math.pi
    
    # This is the main function that runs our script
    def main():
        n_iterations = 1000000
        computed_pi = compute_pi(n_iterations)
    
        print("Approximate value of PI using Monte Carlo method:", computed_pi)
        print("Actual value of PI:", math.pi)
    
    # This is the entry point of our script
    # It runs the main function if this script is executed directly
    if __name__ == "__main__":
        main()
    ```
    Here's a brief explanation of the code:
    
    *   `random_point()`: This function generates a random point on a unit square by generating two random numbers between 0 and 1.
    *   `point_inside_circle()`: This function checks if a point lies inside a circle with a radius of 1.
    *   `compute_pi()`: This function computes the value of PI using the Monte Carlo method by generating multiple random points and checking if they lie inside the circle.
    *   `test_compute_pi()`: This function tests the accuracy of our `compute_pi()` function by checking if the computed value of PI lies within 1% of the actual value.
    *   `main()`: This is the main function that runs our script. It computes PI using `compute_pi()` and prints the result.
    
    To run this code, you'll need to install the `pytest` package:
    
    ```
    pip install pytest
    ```
    
    Then, you can run the code by executing the script:
    
    ```
    python compute_pi.py
    ```
    
    The output should be something like this:
    
    ```
    Approximate value of PI using Monte Carlo method: 3.141186
    Actual value of PI: 3.141592653589793
    ```
    
    If you want to run the test, you can execute:
    
    ```
    pytest compute_pi.py
    ```
    
    The output should be something like this:
    
    ```
    =============================================== test session starts ================================================
    platform linux -- Python 3.6.13, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
    rootdir: /home/anyscale/code
    collected 1 item
    
    compute_pi.py .                                                                                              [100%]
    
    ============================================= 1 passed in 1.38s =============================================
    ```
    
    This test checks if the computed value of PI lies within 1% of the actual value (3.141592653589793). If the computed value is outside this range, the test will fail.
    

### Example 2: Generate a Python code generate fractions

Generate Python code that a) generates fractions between 1 and 10, finds and common denominator, and computes the sum.


```python
user_prompt = """
# CONTEXT #
I want to generate Python code to showcase our product feature for
serving open source large language models such as Code Llama series
on Anyscale Endpoints. The product feature is Anyscale Endpoints, which 
serves all Llama series models and the Mistral series too. For this
instance, we want to show case Code Llama 70B just released by Meta AI, and 
now hosted by Anyscale Endpoints

#############

# OBJECTIVE # 
Create a Python code that generates a list of fractions between 1 and 10.
It then generates a common denominator, and adds the sum of all fractions.

#############

# STYLE #
Use the Google style of formating Python code.

#############

# AUDIENCE #
Your code should be well commented and aimed at both beginner and
intermediate Python programers. 

#############

# RESPONSE #
Generate the entire code that can be easily copy and pasted into file: 'add_fractions.py'. Also,
provide specific instructions how to compile the code, run it, and any 
additional python packages it needs. The final output of the code
should be a list of fractions genetated, their sum, and their common denominator.
"""
```


```python
response = get_commpletion(client, MODEL, system_content, user_prompt)
print(f"\n{BOLD_BEGIN}Answer - Python code{BOLD_END}:\n {response}")
```

    
    [1mAnswer - Python code[0m:
     1. Install dependencies:
        * We will use the `fractions` module in Python 3 to generate fractions. Run the following command to install the module:
        ```
        pip install fractions
        ```
    2. Create a file named `add_fractions.py`.
    3. Copy and paste the following code:
    
    ```python
    # add_fractions.py
    
    from fractions import Fraction
    
    # Create a list of fractions between 1 and 10
    fractions = [Fraction(i, 10) for i in range(1, 10)]
    
    # Calculate the sum of all fractions
    sum_fractions = sum(fractions, 0)
    
    # Calculate the common denominator
    common_denominator = sum_fractions.denominator
    
    # Print the result
    print("Fractions:")
    for f in fractions:
        print(f" - {f}")
    print()
    print(f"Sum: {sum_fractions}")
    print(f"Common Denominator: {common_denominator}")
    ```
    
    4. Run the code using the command:
    
    ```
    python add_fractions.py
    ```
    
    Example output:
    
    ```
    $ python add_fractions.py
    Fractions:
     - 1/10
     - 1/5
     - 3/10
     - 2/5
     - 1/2
     - 3/5
     - 7/10
     - 4/5
     - 9/10
    
    Sum: 59/50
    Common Denominator: 50
    ```
    
    5. We can further simplify the code by removing the `fractions` list and calculating the sum directly as follows:
    
    ```python
    # add_fractions_simplified.py
    
    from fractions import Fraction
    
    # Calculate the sum of all fractions
    sum_fractions = sum(Fraction(i, 10) for i in range(1, 10))
    
    # Calculate the common denominator
    common_denominator = sum_fractions.denominator
    
    # Print the result
    print(f"Sum: {sum_fractions}")
    print(f"Common Denominator: {common_denominator}")
    ```
    
    6. Run the simplified code using the command:
    
    ```
    python add_fractions_simplified.py
    ```
    
    Example output:
    
    ```
    $ python add_fractions_simplified.py
    Sum: 59/50
    Common Denominator: 50
    ```
    
    7. Now, we can add some extra code to generate the Code Llama 70B model and serve it using Anyscale Endpoints. Here's a sample code with inline comments:
    
    ```python
    # add_fractions_with_llama_70b.py
    
    # Import necessary libraries
    from fractions import Fraction
    from meta.pretrain.featurize.featurize import Featurizer  # This is a Meta AI library
    from anyscale.sdk import Model  # This is an Anyscale Endpoints library
    
    # Load the Code Llama 70B model
    llama_70b = Model.from_pretrained("code-llama-70B")
    
    # Create a list of fractions between 1 and 10
    fractions = [Fraction(i, 10) for i in range(1, 10)]
    
    # Calculate the sum of all fractions
    sum_fractions = sum(fractions, 0)
    
    # Calculate the common denominator
    common_denominator = sum_fractions.denominator
    
    # Print the result
    print("Fractions:")
    for f in fractions:
        print(f" - {f}")
    print()
    print(f"Sum: {sum_fractions}")
    print(f"Common Denominator: {common_denominator}")
    
    
    #---------------Serve the model using Anyscale Endpoints----------------------
    
    # Initialize the Anyscale Endpoints model
    endpoint = Endpoint("Code Llama 70B")
    
    # Specify the input and output format
    input_format = "string"
    output_format = "string"
    
    # Define the endpoint's predict function
    def predict(input_str):
        # Convert input to tokens
        input_tokens = Featurizer.featurize(input_str)
        
        # Run the model
        prediction = llama_70b(input_tokens)
        
        # Convert output to string
        output_str = str(prediction)
        
        return output_str
    
    # Deploy the model
    endpoint.deploy(predict, input_format, output_format)
    
    # Print the endpoint URL for accessing the model
    print(f"Endpoint URL: {endpoint.url}")
    ```
    
    8. Run the code using the command:
    
    ```
    python add_fractions_with_llama_70b.py
    ```
    
    Example output:
    
    ```
    $ python add_fractions_with_llama_70b.py
    Fractions:
     - 1/10
     - 1/5
     - 3/10
     - 2/5
     - 1/2
     - 3/5
     - 7/10
     - 4/5
     - 9/10
    
    Sum: 59/50
    Common Denominator: 50
    Endpoint URL: https://api.anyscale.com/<your_token>/Code-Llama-70B/v0/predict
    ```
    
    9. Now, you can access the model hosted on Anyscale Endpoints by sending HTTP requests to the endpoint URL. The model takes a string as input and returns the result as a string.
    
    Please note that this is just a sample code and you may need to modify it according to your specific requirements.
    


```python

```
