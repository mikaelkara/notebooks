```
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Intro to Controlled Generation with the Gemini API

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/controlled-generation/intro_controlled_generation.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fcontrolled-generation%2Fintro_controlled_generation.ipynb">
      <img width="32px" src="https://cloud.google.com/ml-engine/images/colab-enterprise-logo-32px.png" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/controlled-generation/intro_controlled_generation.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/controlled-generation/intro_controlled_generation.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Eric Dong](https://github.com/gericdong)|

## Overview

### Gemini

Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases.

### Controlled Generation

Depending on your application, you may want the model response to a prompt to be returned in a structured data format, particularly if you are using the responses for downstream processes, such as downstream modules that expect a specific format as input. The Gemini API provides the controlled generation capability to constraint the model output to a structured format. This capability is available in the following models:

- Gemini 1.5 Pro
- Gemini 1.5 Flash

Learn more about [control generated output](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output).


### Objectives

In this tutorial, you learn how to use the controlled generation capability in the Gemini API in Vertex AI to generate model responses in a structured data format.

You will complete the following tasks:

- Sending a prompt with a response schema
- Using controlled generation in use cases requiring output constraints


## Get started

### Install Vertex AI SDK and other required packages



```
%pip install --upgrade --user --quiet google-cloud-aiplatform
```

### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.


```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.


```
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

## Code Examples

### Import libraries


```
import json

from vertexai import generative_models
from vertexai.generative_models import GenerationConfig, GenerativeModel, Part
```

### Sending a prompt with a response schema

The Gemini 1.5 Pro (`gemini-1.5-pro`) and the Gemini 1.5 Flash (`gemini-1.5-flash`) models allow you define a response schema to specify the structure of a model's output, the field names, and the expected data type for each field. The response schema is specified in the `response_schema` parameter in `generation_config`, and the model output will strictly follow that schema.

The following examples use the Gemini 1.5 Flash (`gemini-1.5-flash`) model.


```
MODEL_ID = "gemini-1.5-flash"  # @param {type:"string"}

model = GenerativeModel(MODEL_ID)
```

Define a response schema for the model output. Use only the supported fields as listed below. All other fields are ignored.

- enum
- items
- maxItems
- nullable
- properties
- required


When a model generates a response, it uses the field name and context from your prompt. As such, we recommend that you use a clear structure and unambiguous field names so that your intent is clear.

By default, fields are optional, meaning the model can populate the fields or skip them. You can set fields as required to force the model to provide a value. If there's insufficient context in the associated input prompt, the model generates responses mainly based on the data it was trained on.

If you aren't seeing the results you expect, add more context to your input prompts or revise your response schema. For example, review the model's response without controlled generation to see how the model responds. You can then update your response schema that better fits the model's output.


```
response_schema = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "recipe_name": {
                "type": "STRING",
            },
        },
        "required": ["recipe_name"],
    },
}
```

When prompting the model to generate the content, pass the schema to the `response_schema` field of the `generation_config`. 

You also need to specify the model output format in the `response_mime_type` field. Output formats such as `application/json` and `text/x.enum` are supported.


```
response = model.generate_content(
    "List a few popular cookie recipes",
    generation_config=GenerationConfig(
        response_mime_type="application/json", response_schema=response_schema
    ),
)

print(response.text)
```

    [{"recipe_name": "Chocolate Chip Cookies"}, {"recipe_name": "Oatmeal Raisin Cookies"}, {"recipe_name": "Sugar Cookies"}, {"recipe_name": "Peanut Butter Cookies"}, {"recipe_name": "Snickerdoodles"}] 
    

You can parse the response string to JSON.


```
json_response = json.loads(response.text)
print(json_response)
```

    [{'recipe_name': 'Chocolate Chip Cookies'}, {'recipe_name': 'Oatmeal Raisin Cookies'}, {'recipe_name': 'Sugar Cookies'}, {'recipe_name': 'Peanut Butter Cookies'}, {'recipe_name': 'Snickerdoodles'}]
    

### Using controlled generation in use cases requiring output constraints

Controlled generation can be used to ensure that model outputs adhere to a specific structure (e.g., JSON), instruct the model to perform pure multiple choices (e.g., sentiment classification), or follow certain style or guidelines.

Let's use controlled generation in the following use cases that require output constraints.

#### **Example**: Generate game character profile

In this example, you instruct the model to create a game character profile with some specific requirements, and constraint the model output to a structured format. This example also demonstrates how to configure the `response_schema` and `response_mime_type` fields in `generative_config` in conjunction with `safety_settings`.


```
response_schema = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "name": {"type": "STRING"},
            "age": {"type": "INTEGER"},
            "occupation": {"type": "STRING"},
            "background": {"type": "STRING"},
            "playable": {"type": "BOOLEAN"},
            "children": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "age": {"type": "INTEGER"},
                    },
                    "required": ["name", "age"],
                },
            },
        },
        "required": ["name", "age", "occupation", "children"],
    },
}

prompt = """
    Generate a character profile for a video game, including the character's name, age, occupation, background, names of their
    three children, and whether they can be controlled by the player.
"""

response = model.generate_content(
    prompt,
    generation_config=GenerationConfig(
        response_mime_type="application/json", response_schema=response_schema
    ),
    safety_settings={
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    },
)

print(response.text)
```

    [{"age": 45, "children": [{"age": 18, "name": "Sarah"}, {"age": 16, "name": "Michael"}, {"age": 14, "name": "Emily"}], "name": "John Smith", "occupation": "Carpenter", "background": "John is a hard-working carpenter who loves spending time with his family.", "playable": false}] 
    

#### **Example**: Extract errors from log data

In this example, you use the model to pull out specific error messages from unstructured log data, extract key information, and constraint the model output to a structured format.

Some properties are set to nullable so the model can return a null value when it doesn't have enough context to generate a meaningful response.



```
response_schema = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "timestamp": {"type": "STRING"},
            "error_code": {"type": "INTEGER", "nullable": True},
            "error_message": {"type": "STRING"},
        },
        "required": ["timestamp", "error_message", "error_code"],
    },
}

prompt = """
[15:43:28] ERROR: Could not process image upload: Unsupported file format. (Error Code: 308)
[15:44:10] INFO: Search index updated successfully.
[15:45:02] ERROR: Service dependency unavailable (payment gateway). Retrying... (Error Code: 5522)
[15:45:33] ERROR: Application crashed due to out-of-memory exception. (Error Code: 9001)
"""

response = model.generate_content(
    prompt,
    generation_config=GenerationConfig(
        response_mime_type="application/json", response_schema=response_schema
    ),
)

print(response.text)
```

    [{"error_code": 308, "error_message": "Could not process image upload: Unsupported file format.", "timestamp": "15:43:28"}, {"error_code": null, "error_message": "Search index updated successfully.", "timestamp": "15:44:10"}, {"error_code": 5522, "error_message": "Service dependency unavailable (payment gateway). Retrying...", "timestamp": "15:45:02"}, {"error_code": 9001, "error_message": "Application crashed due to out-of-memory exception.", "timestamp": "15:45:33"}] 
    

#### **Example**: Analyze product review data

In this example, you instruct the model to analyze product review data, extract key entities, perform sentiment classification (multiple choices), provide additional explanation, and output the results in JSON format.


```
response_schema = {
    "type": "ARRAY",
    "items": {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "rating": {"type": "INTEGER"},
                "flavor": {"type": "STRING"},
                "sentiment": {
                    "type": "STRING",
                    "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
                },
                "explanation": {"type": "STRING"},
            },
            "required": ["rating", "flavor", "sentiment", "explanation"],
        },
    },
}

prompt = """
  Analyze the following product reviews, output the sentiment classification and give an explanation.
  
  - "Absolutely loved it! Best ice cream I've ever had." Rating: 4, Flavor: Strawberry Cheesecake
  - "Quite good, but a bit too sweet for my taste." Rating: 1, Flavor: Mango Tango
"""

response = model.generate_content(
    prompt,
    generation_config=GenerationConfig(
        response_mime_type="application/json", response_schema=response_schema
    ),
)

print(response.text)
```

    [
      [
        {
          "explanation": "The customer expresses extreme positive sentiment with words like \"loved\" and \"best\", indicating a high level of satisfaction.",
          "flavor": "Strawberry Cheesecake",
          "rating": 4,
          "sentiment": "POSITIVE"
        },
        {
          "explanation": "The customer expresses a somewhat positive sentiment, acknowledging the product is \"good\" but also noting a negative aspect (too sweet) indicating a less enthusiastic experience.",
          "flavor": "Mango Tango",
          "rating": 1,
          "sentiment": "NEGATIVE"
        }
      ]
    ] 
    

#### Example: Detect objects in images

You can also use controlled generation in multimodality use cases. In this example, you instruct the model to detect objects in the images and output the results in JSON format. These images are stored in a Google Storage bucket.

- [office-desk.jpeg](https://storage.googleapis.com/cloud-samples-data/generative-ai/image/office-desk.jpeg)
- [gardening-tools.jpeg](https://storage.googleapis.com/cloud-samples-data/generative-ai/image/gardening-tools.jpeg)


```
response_schema = {
    "type": "ARRAY",
    "items": {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "object": {"type": "STRING"},
            },
        },
    },
}

prompt = "Generate a list of objects in the images."

response = model.generate_content(
    [
        Part.from_uri(
            "gs://cloud-samples-data/generative-ai/image/office-desk.jpeg",
            "image/jpeg",
        ),
        Part.from_uri(
            "gs://cloud-samples-data/generative-ai/image/gardening-tools.jpeg",
            "image/jpeg",
        ),
        prompt,
    ],
    generation_config=GenerationConfig(
        response_mime_type="application/json", response_schema=response_schema
    ),
)

print(response.text)
```

    [
      [
        {
          "object": "globe"
        },
        {
          "object": "tablet"
        },
        {
          "object": "shopping cart"
        },
        {
          "object": "gift"
        },
        {
          "object": "coffee"
        },
        {
          "object": "cup"
        },
        {
          "object": "keyboard"
        },
        {
          "object": "mouse"
        },
        {
          "object": "passport"
        },
        {
          "object": "sunglasses"
        },
        {
          "object": "money"
        },
        {
          "object": "notebook"
        },
        {
          "object": "pen"
        },
        {
          "object": "eiffel tower"
        },
        {
          "object": "airplane"
        }
      ],
      [
        {
          "object": "watering can"
        },
        {
          "object": "plant"
        },
        {
          "object": "pot"
        },
        {
          "object": "gloves"
        },
        {
          "object": "hand trowel"
        }
      ]
    ] 
    

#### Example: Respond with a single plain text enum value

This example identifies the genre of a movie based on its description. The output is one plain-text enum value that the model selects from a list values that are defined in the response schema. Note that in this example, the `response_mime_type` field is set to `text/x.enum`.



```
response_schema = {"type": "STRING", "enum": ["drama", "comedy", "documentary"]}

prompt = (
    "The film aims to educate and inform viewers about real-life subjects, events, or people."
    "It offers a factual record of a particular topic by combining interviews, historical footage, "
    "and narration. The primary purpose of a film is to present information and provide insights "
    "into various aspects of reality."
)

response = model.generate_content(
    prompt,
    generation_config=GenerationConfig(
        response_mime_type="text/x.enum", response_schema=response_schema
    ),
)

print(response.text)
```

    documentary
    
