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

# Gemini API: JSON Mode Quickstart

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/JSON_mode.ipynb"><img src="https://github.com/google-gemini/cookbook/blob/0f9a09fd0db1b54cc1af0f87f11f88b493e0cb3b/images/colab_logo_32px.png?raw=1" />Run in Google Colab</a>
  </td>
</table>

The Gemini API can be used to generate a JSON output if you set the schema that you would like to use.

Two methods are available. You can either set the desired output in the prompt or supply a schema to the model separately.

### Install dependencies


```
!pip install -U -q "google-generativeai>=0.7.2"
```

    [?25l   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/164.0 kB[0m [31m?[0m eta [36m-:--:--[0m[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m [32m163.8/164.0 kB[0m [31m8.8 MB/s[0m eta [36m0:00:01[0m[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m164.0/164.0 kB[0m [31m3.1 MB/s[0m eta [36m0:00:00[0m
    [?25h[?25l   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/725.4 kB[0m [31m?[0m eta [36m-:--:--[0m[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m [32m716.8/725.4 kB[0m [31m38.0 MB/s[0m eta [36m0:00:01[0m[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m725.4/725.4 kB[0m [31m16.1 MB/s[0m eta [36m0:00:00[0m
    [?25h


```
import google.generativeai as genai

import json
import dataclasses
import typing_extensions as typing
```

### Configure your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
```

## Set your constrained output in the prompt

 Activate JSON mode by specifying `respose_mime_type` in the `generation_config` parameter:


```
model = genai.GenerativeModel("gemini-1.5-flash-latest",
                              generation_config={"response_mime_type": "application/json"})
```

For this first example just describe the schema you want back in the prompt:


```
prompt = """List a few popular cookie recipes using this JSON schema:

Recipe = {'recipe_name': str}
Return: list[Recipe]"""
```


```
raw_response = model.generate_content(prompt)
```

Parse the string to JSON:


```
response = json.loads(raw_response.text)
print(response)
```

    [{'recipe_name': 'Chocolate Chip Cookies'}, {'recipe_name': 'Oatmeal Raisin Cookies'}, {'recipe_name': 'Snickerdoodles'}, {'recipe_name': 'Sugar Cookies'}, {'recipe_name': 'Peanut Butter Cookies'}]
    

For readability serialize and print it:


```
print(json.dumps(response, indent=4))
```

    [
        {
            "recipe_name": "Chocolate Chip Cookies"
        },
        {
            "recipe_name": "Oatmeal Raisin Cookies"
        },
        {
            "recipe_name": "Snickerdoodles"
        },
        {
            "recipe_name": "Sugar Cookies"
        },
        {
            "recipe_name": "Peanut Butter Cookies"
        }
    ]
    

## Supply the schema to the model directly

The newest models (1.5 and beyond) allow you to pass a schema object (or a python type equivalent) directly and the output will strictly follow that schema.

Following the same example as the previous section, here's that recipe type:


```
class Recipe(typing.TypedDict):
    recipe_name: str
    recipe_description: str
    recipe_ingredients: list[str]
```

For this example you want a list of `Recipe` objects, so pass `list[Recipe]` to the `response_schema` field of the `generation_config`.


```
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")

result = model.generate_content(
    "List a few imaginative cookie recipes along with a one-sentence description as if you were a gourmet restaurant and their main ingredients",
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema = list[Recipe]),
    request_options={"timeout": 600},
)
```


```
print(json.dumps(json.loads(result.text), indent=4))
```

    [
        {
            "recipe_description": "A delicate and fragrant cookie infused with the bright citrus notes of bergamot and the warming spice of cardamom.",
            "recipe_ingredients": [
                "Bergamot zest",
                "Cardamom pods",
                "Almond flour",
                "Butter",
                "Sugar"
            ],
            "recipe_name": "Bergamot Cardamom Sabl\u00e9s"
        },
        {
            "recipe_description": "A luxurious and decadent cookie with a rich chocolate flavor and a hint of salty caramel.",
            "recipe_ingredients": [
                "Dark chocolate",
                "Sea salt",
                "Caramel sauce",
                "Butter",
                "Flour"
            ],
            "recipe_name": "Salted Caramel Chocolate Chunk Cookies"
        },
        {
            "recipe_description": "A whimsical and colorful cookie inspired by a summer garden, featuring a blend of fresh herbs and bright citrus.",
            "recipe_ingredients": [
                "Lavender flowers",
                "Lemon zest",
                "Rosemary",
                "Butter",
                "Sugar",
                "Flour"
            ],
            "recipe_name": "Lavender Lemon Rosemary Shortbread"
        }
    ]
    

It is the recommended method if you're using a compatible model.

## Next Steps
### Useful API references:

Check the [structured ouput](https://ai.google.dev/gemini-api/docs/structured-output) documentation or the [`GenerationConfig`](https://ai.google.dev/api/generate-content#generationconfig) API reference for more details

### Related examples

* The constrained output is used in the [Text summarization](../examples/json_capabilities/Text_Summarization.ipynb) example to provide the model a format to summarize a story (genre, characters, etc...)
* The [Object detection](../examples/Object_detection.ipynb) examples are using the JSON constrained output to uniiformize the output of the detection.

### Continue your discovery of the Gemini API

JSON is not the only way to constrain the output of the model, you can also use an [Enum](../quickstarts/Enum.ipynb). [Function calling](../quickstarts/Function_calling.ipynb) and [Code execution](../quickstarts/Code_Execution.ipynb) are other ways to enhance your model by either using your own functions or by letting the model write and run them.
