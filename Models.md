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

# Gemini API: List models


<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Models.ipynb"><img src="../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

This notebook demonstrates how to list the models that are available for you to use in the Gemini API, and how to find details about a model.


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

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
```

## List models

Use `list_models()` to see what models are available. These models support `generateContent`, the main method used for prompting.


```
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
```

    models/gemini-1.0-pro
    models/gemini-1.0-pro-001
    models/gemini-1.0-pro-latest
    models/gemini-1.5-flash
    models/gemini-1.5-flash-001
    models/gemini-1.5-flash-latest
    models/gemini-1.5-pro
    models/gemini-1.5-pro-001
    models/gemini-1.5-pro-latest
    models/gemini-pro
    

These models support `embedContent`, used for embeddings:


```
for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(m.name)
```

    models/embedding-001
    models/text-embedding-004
    

## Find details about a model

You can see more details about a model, including the `input_token_limit` and `output_token_limit` as follows.


```
for m in genai.list_models():
    if m.name == "models/gemini-1.5-flash":
        print(m)
```

    Model(name='models/gemini-1.5-flash',
          base_model_id='',
          version='001',
          display_name='Gemini 1.5 Flash',
          description='Fast and versatile multimodal model for scaling across diverse tasks',
          input_token_limit=1048576,
          output_token_limit=8192,
          supported_generation_methods=['generateContent', 'countTokens'],
          temperature=1.0,
          top_p=0.95,
          top_k=64)
    

## Get model

Use `get_model()` to retrieve the specific details of a model. You can iterate over all available models using `list_models()`, but if you already know the model name you can retrieve it directly with `get_model()`.


```
model_info = genai.get_model("models/aqa")
print(model_info)
```

    Model(name='models/aqa',
          base_model_id='',
          version='001',
          display_name='Model that performs Attributed Question Answering.',
          description=('Model trained to return answers to questions that are grounded in provided '
                       'sources, along with estimating answerable probability.'),
          input_token_limit=7168,
          output_token_limit=1024,
          supported_generation_methods=['generateAnswer'],
          temperature=0.2,
          top_p=1.0,
          top_k=40)
    

## Learning more

* To learn how use a model for prompting, see the [Prompting](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Prompting.ipynb) quickstart.

* To learn how use a model for embedding, see the [Embedding](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Embeddings.ipynb) quickstart.

* For more information on models, visit the [Gemini models](https://ai.google.dev/models/gemini) documentation.
