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

# Gemini API: Entity Extraction

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/json_capabilities/Entity_Extraction_JSON.ipynb"><img src = "../../images/colab_logo_32px.png"/>Run in Google Colab</a>
  </td>
</table>

You will use Gemini to extract all fields that fit one of the predefined classes and label them.


```
!pip install -U -q "google-generativeai>=0.7.2"
```


```
import google.generativeai as genai

import json
from enum import Enum
from typing_extensions import TypedDict # in python 3.12 replace typing_extensions with typing
```

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
```

## Example


```
entity_recognition_text = "John Johnson, the CEO of the Oil Inc. and Coal Inc. companies, has unveiled plans to build a new factory in Houston, Texas."
prompt = f"""
Generate list of entities in text based on the following Python class structure:

class CategoryEnum(str, Enum):
    Person = 'Person'
    Company = 'Company'
    State = 'State'
    City = 'City'

class Entity(TypedDict):
  name: str
  category: CategoryEnum

class Entities(TypedDict):
  entities: list[Entity]

{entity_recognition_text}"""
model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest', generation_config={"temperature": 0})
response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
```


```
print(json.dumps(json.loads(response.text), indent=4))
```

    {
        "entities": [
            {
                "name": "John Johnson",
                "category": "Person"
            },
            {
                "name": "Oil Inc.",
                "category": "Company"
            },
            {
                "name": "Coal Inc.",
                "category": "Company"
            },
            {
                "name": "Houston",
                "category": "City"
            },
            {
                "name": "Texas",
                "category": "State"
            }
        ]
    }
    

## Summary
You have used the Gemini API to extract entities of predefined categories with their labels. You extracted every person, company, state, and country. You are not limited to these categories, as this should work with any category of your choice.

Please see the other notebooks in this directory to learn more about how you can use the Gemini API for other JSON related tasks.

