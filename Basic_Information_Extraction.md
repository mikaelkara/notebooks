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

# Gemini API: Basic information extraction

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/prompting/Basic_Information_Extraction.ipynb"><img src = "../../images/colab_logo_32px.png"/>Run in Google Colab</a>
  </td>
</table>

This example notebook shows how Gemini API's Python SDK can be used to extract information from a block of text and return it in defined structure.

In this notebook, the LLM is given a recipe and is asked to extract all the ingredients to create a shopping list. According to best practices, complex tasks will be executed better if divided into separate steps, such as:

1. First, the model will extract all the groceries into a list.

2. Then, you will prompt it to convert this list into a shopping list.

You can find more tips for writing prompts [here](https://ai.google.dev/gemini-api/docs/prompting-intro).



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

## Example

First, start by extracting all the groceries. To dod this, set the system instructions when defining the model


```
groceries_system_prompt = f"""
Your task is to extract to a list all the groceries with its quantities based on the provided recipe.
Make sure that groceries are in the order of appearance.
"""
grocery_extraction_model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest',
                                                 system_instruction=groceries_system_prompt)
```

Next, the recipe is defined. You will pass the recipe into `generate_content`, and see that the list of groceries was successfully extracted from the input.


```
recipe = """
Step 1:
Grind 3 garlic cloves, knob of fresh ginger, roughly chopped, 3 spring onions to a paste in a food processor.
Add 2 tbsp of clear honey, juice from one orange, 1 tbsp of light soy sauce and 2 tbsp of vegetable oil, then blend again.
Pour the mixture over the cubed chicken from 4 small breast fillets and leave to marnate for at least 1hr.
Toss in the 20 button mushrooms for the last half an hour so the take on some of the flavour, too.

Step 2:
Thread the chicken, 20 cherry tomatoes, mushrooms and 2 large red peppers onto 20 wooden skewers,
then cook on a griddle pan for 7-8 mins each side or until the chicken is thoroughly cooked and golden brown.
Turn the kebabs frequently and baste with the marinade from time to time until evenly cooked.
Arrange on a platter, and eat with your fingers.
"""

grocery_list = grocery_extraction_model.generate_content(recipe)
print(grocery_list.text)
```

    - 3 garlic cloves
    - knob of fresh ginger
    - 3 spring onions
    - 2 tbsp clear honey
    - 1 orange
    - 1 tbsp light soy sauce
    - 2 tbsp vegetable oil
    - 4 small chicken breast fillets
    - 20 button mushrooms
    - 20 cherry tomatoes
    - 2 large red peppers
    - 20 wooden skewers 
    
    

The next step is to further format the shopping list based on the ingredients extracted.


```
shopping_list_system_prompt = """
You are given a list of groceries. Complete the following:
- Organize groceries into categories for easier shopping.
- List each item one under another with a checkbox [].
"""

shopping_list_model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest',
                                            system_instruction=shopping_list_system_prompt)
```

Now that you have defined the instructions, you can also decide how you want to format your grocery list. Give the prompt a couple examples, or perform few-shot prompting, so it understands how to format your grocery list.


```
shopping_list_prompt = f"""
LIST: 3 tomatoes, 1 turkey, 4 tomatoes
OUTPUT:
## VEGETABLES
- [ ] 7 tomatoes
## MEAT
- [ ] 1 turkey

LIST: {grocery_list.text}
OUTPUT:
"""
Markdown(shopping_list_model.generate_content(shopping_list_prompt).text)
```




## PRODUCE
- [ ] 3 garlic cloves
- [ ] knob of fresh ginger
- [ ] 3 spring onions
- [ ] 1 orange
- [ ] 20 button mushrooms
- [ ] 20 cherry tomatoes
- [ ] 2 large red peppers

## PANTRY
- [ ] 2 tbsp clear honey
- [ ] 1 tbsp light soy sauce
- [ ] 2 tbsp vegetable oil

## MEAT
- [ ] 4 small chicken breast fillets

## OTHER
- [ ] 20 wooden skewers 




## Next steps

Be sure to explore other examples of prompting in the repository. Try creating your own prompts for information extraction or adapt the ones provided in the notebook.
