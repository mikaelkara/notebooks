<img src="./images/llm_prompt_req_resp.png" height="35%" width="%65">


## Few-shot learning prompting

Few-shot learning in the context of prompting involves teaching a language model to perform a task by showing it only a small number of examples. Essentially, you provide a few samples of the input along with the desired output, and the model learns to generalize from these examples to new, similar tasks. This method is useful for adapting the model to specific tasks without extensive training data.

This approach allows LLMs to adapt to a wide variety of tasks without extensive retraining or fine-tuning on large datasets.

Let's have a go at few examples of few-shot learning prompts

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

## Example 1: Recipe Generation
An example use case for few-shot learning is to provide couple of examples of recipies that the model can learn from and generalize. Using 
what it has learned as couple of in-context examples, it can generate a response for its subsequent prompt.


```python
system_content = """You are supreme repository of knowledge, including all 
exotic culnary dishes across the globe."""

few_shot_recipe_prompt = """Recipe for simple cheese omlette. 
Ingridients: 
1. 3 large eggs
2. 1/4 cup shredded cheddar cheese
3. Salt and pepper to taste
4. 1 tablespoon butter
Directions:
1. Crack the eggs into a bowl, add salt, and pepper, then beat them until well mixed.
2. Heat the butter in a non-stick pan over medium heat until it melts and coats the pan.
3. Pour the beaten eggs into the pan, swirling them to cover the bottom evenly.
4. Sprinkle the shredded cheese over one half of the eggs.
5. Once the edges are set, fold the omelet in half with a spatula, covering the cheese. Cook for another minute until the cheese melts. Enjoy your delicious cheese omelet!

Generate a recipe for choclate pancakes:
Ingridients:
Directions:
"""
```


```python
response = get_commpletion(client, MODEL, system_content, few_shot_recipe_prompt)
print(f"{response}\n")
```

    Here's a simple recipe for chocolate pancakes:
    
    Ingredients:
    - 1 cup all-purpose flour
    - 2 tablespoons cocoa powder
    - 2 tablespoons sugar
    - 1/4 teaspoon salt
    - 2 teaspoons baking powder
    - 1 large egg
    - 1 cup milk
    - 2 tablespoons unsalted butter, melted
    - 1/2 teaspoon pure vanilla extract
    - 1/4 cup semi-sweet chocolate chips (optional for added chocolatey goodness)
    - Butter or oil for cooking
    - Maple syrup, whipped cream, or your choice of toppings
    
    Directions:
    1. In a large mixing bowl, whisk together the flour, cocoa powder, sugar, salt, and baking powder to combine and break up any lumps.
    2. In another bowl, beat the egg and then add the milk, melted butter, and vanilla extract. Stir to combine.
    3. Pour the wet ingredients into the dry ingredients, mixing gently until just combined. Be careful not to overmix; a few lumps are okay. If the batter is too thick, you can add a little more milk to reach the desired consistency.
    4. Fold in the chocolate chips if using.
    5. Heat a non-stick skillet or griddle over medium heat. Brush with a small amount of butter or oil.
    6. For each pancake, ladle about 1/4 cup of batter onto the griddle. Cook until bubbles form on the surface and the edges look set, about 2-3 minutes.
    7. Flip the pancakes carefully and cook for another 2 minutes on the other side, or until cooked through.
    8. Repeat with the remaining batter, adding more butter or oil to the skillet as needed.
    9. Serve hot with your favorite toppings like maple syrup, more chocolate chips, fruit, whipped cream, or a dusting of powdered sugar.
    
    Enjoy your homemade chocolate pancakes!
    
    

### Example 2: Code Generation and Scripting

Using few-shot learning, we can provide in-context examples of how to write a script that the model can generalize and generate any script 
in the language its trained on.



```python
system_content = """You are supreme repository of knowledge, and assistant
code Co-pilot for developer to assist them in generating sample code, inclding
in langauges such as Python, JavaScript, Java, C, C++, and shell script.
"""

few_shot_code_gen_prompt = """
Python code to compute Fibonacci series:

Problem: Compute Fibonnacci series
Language: Python
Code: 
def fibonacci(n):
    fib_series = [0, 1]
    while fib_series[-1] + fib_series[-2] <= n:
        fib_series.append(fib_series[-1] + fib_series[-2])
    return fib_series

# Example: Compute Fibonacci series up to 50
result = fibonacci(50)
print(result)

Write a JavaScript to read HTML contents from a Web URL. Follow
the format and labels provide below:
Problem: 
Language:
Code:
"""
```


```python
response = get_commpletion(client, MODEL, system_content, few_shot_code_gen_prompt)
print(f"{response}\n")
```

    Problem: Read HTML contents from a web URL
    Language: JavaScript
    Code:
    ```javascript
    const https = require('https');
    
    // Function to read HTML content from a given URL
    function readHTMLContent(url) {
        return new Promise((resolve, reject) => {
            https.get(url, (response) => {
                if (response.statusCode < 200 || response.statusCode >= 300) {
                    return reject(new Error(`Request failed with status code ${response.statusCode}`));
                }
    
                let data = '';
    
                // A chunk of data has been received.
                response.on('data', (chunk) => {
                    data += chunk;
                });
    
                // The whole response has been received. Print out the result.
                response.on('end', () => {
                    resolve(data);
                });
    
            }).on('error', (e) => {
                reject(e);
            });
        });
    };
    
    // Example usage:
    const url = 'https://example.com'; // Replace with your desired URL
    
    readHTMLContent(url)
        .then((htmlContent) => {
            console.log(htmlContent); // Outputs the HTML content of the page
        })
        .catch((error) => {
            console.error(error);
        });
    ```
    
    Before running this JavaScript code, make sure that you are operating in an environment where the `https` module is available, such as Node.js. This code will not work in a front-end JavaScript environment like a browser without additional configurations such as CORS policies or using a server-side proxy.
    
    

### Example 3: Language transation form and format


```python
system_content = """You are supreme polyglot and speak numerous languages across the globe."""

few_shot_lang_trans_prompt = """English to {target_language} translation. 
English: What is your first and last name?
French: Comment vous appelez-vous (prénom et nom) ?

English: How long have been living in Brazil?
Protugese: Há quanto tempo você vive no Brasil?

English: What's your favorite footaball team in La Liga in Spain?
Spanish: ¿Cuál es tu equipo favorito de fútbol en La Liga en España?

English: When is the best time to attend October Fest in Germany, and where should I stay?"
Germany: Wann ist die beste Zeit, um das Oktoberfest in Deutschland zu besuchen, und wo sollte ich übernachten?

English: 
Mandarin:

Translate the English sentences into the Germany, Spanish, and Mandarin.
English: Who is regarded as the best footballer in England?"
Germany:
Spanish:
Mandarin:
"""
```


```python
response = get_commpletion(client, MODEL, system_content, few_shot_lang_trans_prompt)
print(f"{response}\n")
```

    Germany: Wer gilt als der beste Fußballspieler in England?
    Spanish: ¿Quién es considerado el mejor futbolista en Inglaterra?
    Mandarin: 谁被认为是英格兰最好的足球运动员？(Shéi bèi rènwéi shì Yīnggélán zuì hǎo de zúqiú yùndòngyuán?)
    
    


```python
lang_prompt = """"Translate the English sentences into the Germany.
English: When is the best time to attend October Fest in Germany, and where should I stay?
Germany:
"""
```


```python
response = get_commpletion(client, MODEL, system_content, lang_prompt)
print(f"{response}\n")
```

    German: Wann ist die beste Zeit, um das Oktoberfest in Deutschland zu besuchen, und wo sollte ich übernachten?
    
    

## All this is amazing! 😜 Feel the wizardy in prompt power 🧙‍♀️
