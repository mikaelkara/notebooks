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

# Gemini API: Few-shot prompting

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/prompting/Few_shot_prompting.ipynb"><img src = "../../images/colab_logo_32px.png"/>Run in Google Colab</a>
  </td>
</table>

Some prompts may need a bit more information or require a specific output schema for the LLM to understand and accomplish the requested task. In such cases, providing example questions with answers to the model may greatly increase the quality of the response.


```
!pip install -U -q "google-generativeai>=0.7.2"
```


```
import google.generativeai as genai
```

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
```

## Examples

Use Gemini 1.5 Flash as your model to run through the following examples.


```
model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
```


```
prompt = """
Sort the animals from biggest to smallest.
Question: Sort Tiger, Bear, Dog
Answer: Bear > Tiger > Dog}
Question: Sort Cat, Elephant, Zebra
Answer: Elephant > Zebra > Cat}
Question: Sort Whale, Goldfish, Monkey
Answer:"""
model.generate_content(prompt).text
```




    'Answer: Whale > Monkey > Goldfish \n'




```
prompt = """
Extract cities from text, include country they are in.
USER: I visited Mexico City and Poznan last year
MODEL: {"Mexico City": "Mexico", "Poznan": "Poland"}
USER: She wanted to visit Lviv, Monaco and Maputo
MODEL: {"Minsk": "Ukraine", "Monaco": "Monaco", "Maputo": "Mozambique"}
USER: I am currently in Austin, but I will be moving to Lisbon soon
MODEL:"""
model.generate_content(prompt, generation_config={'response_mime_type':'application/json'}).text
```




    '```json\n{"Austin": "United States", "Lisbon": "Portugal"}\n``` \n'



## Next steps

Be sure to explore other examples of prompting in the repository. Try writing prompts about classifying your own data, or try some of the other prompting techniques such as zero-shot prompting.
