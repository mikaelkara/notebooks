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

# Gemini API: Adding context information

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/prompting/Adding_context_information.ipynb"><img src = "../../images/colab_logo_32px.png"/>Run in Google Colab</a>
  </td>
</table>

While LLMs are trained extensively on various documents and data, the LLM does not know everything. New information or information that is not easily accessible cannot be known by the LLM, unless it was specifically added to its corpus of knowledge somehow. For this reason, it is sometimes necessary to provide the LLM, with information and context necessary to answer our queries by providing additional context.


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

Let's say you provide some statistics from a recent Olympics competition, and this data wasn't used to train the LLM. Insert it into the prompt, and input the prompt to the model.


```
# the list as of April 2024
prompt = """
QUERY: provide a list of atheletes that competed in olympics exactly 9 times.
CONTEXT:

Table title: Olympic athletes and number of times they've competed
Ian Millar, 10
Hubert Raudaschl, 9
Afanasijs Kuzmins, 9
Nino Salukvadze, 9
Piero d'Inzeo, 8
Raimondo d'Inzeo, 8
Claudia Pechstein, 8
Jaqueline Mourão, 8
Ivan Osiier, 7
François Lafortune, Jr, 7

"""
model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
Markdown(model.generate_content(prompt).text)
```




Here are the athletes who competed in the Olympics exactly 9 times:

* **Hubert Raudaschl**
* **Afanasijs Kuzmins**
* **Nino Salukvadze** 




## Next steps

While some information may be easily searchable online without the use of an LLM, consider data that is not found on the internet, such as private documentation, quickbooks, and forums. Use this code as a template to help you input that information into the Gemini model.

Be sure to explore other examples of prompting in the repository. Try writing prompts about classifying your own data, or try some of the other prompting techniques such as few-shot prompting.
