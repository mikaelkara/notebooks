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

# Code migration from PaLM to Gemini
<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/language/migration/PaLM_to_gemini_codemigration.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Flanguage%2Fmigration%2FPaLM_to_gemini_codemigration.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/migration/PaLM_to_gemini_codemigration.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/migration/PaLM_to_gemini_codemigration.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Neelima Reddy](https://github.com/NeelRed) |

## Overview


This guide shows how to migrate the Vertex AI SDK for Python from using the PaLM API to using the Gemini API. You can generate text, multi-turn conversations (chat), and code with Gemini. After you migrate, check your responses because the Gemini output might be different from PaLM output. For more information, see the Introduction to multimodal classes in the [Vertex AI SDK]( https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini).


## Differences between PaLM and Gemini

The following are some differences between Gemini and PaLM models:

- Their response structures are different. To learn about the Gemini response structure, see the [Gemini API model reference response body](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini#response_body).

- Their safety categories are different. To learn about differences between Gemini and PaLM safety settings, see [Key differences between Gemini and other model families](https://cloud.google.com/vertex-ai/generative-ai/docs/migrate/migrate-palm-to-gemini#:~:text=Key%20differences%20between%20Gemini%20and%20other%20model%20families).

- Gemini can't perform code completion. If you need to create a code completion application, use the `code-gecko` model. For more information, see [Codey code completion model](https://cloud.google.com/vertex-ai/generative-ai/docs/code/test-code-completion-prompts).

- For code generation, Gemini has a higher recitation block rate. The token generation stops if the response is flagged for unauthorised citations. For more information, see [Gemini API reference](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerateContentResponse#FinishReason)

- The confidence score in Codey code generation models that indicates how confident the model is in its response isn't exposed in Gemini.


## Update PaLM code to use Gemini models

The methods on the `GenerativeModel` class are mostly the same as the methods on the PaLM classes. For example, use `GenerativeModel.start_chat()` to replace the PaLM equivalent, `ChatModel.start_chat()`. However, because Google Cloud is always improving and updating Gemini, you might run into some differences. For more information, see the [Python SDK Reference](https://ai.google.dev/api/python/google/generativeai).

To migrate from the PaLM API to the Gemini API, the following code modifications are required:

- For all PaLM model classes, you use the `GenerativeModel` class in Gemini.

- To use the `GenerativeModel` class, run the following import statement:

  from `vertexai.generative_models` import `GenerativeModel`



- To generate text in Gemini, use the `GenerativeModel.generate_content()` method instead of the `predict()` method that's used on PaLM models.  


## Getting Started

### Install Vertex AI SDK and other required packages


```
%pip install --upgrade --user --quiet google-cloud-aiplatform
```

### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.


```
# Restart kernel after installs so that your environment can access the new packages
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the cell below to authenticate your environment. This step is not required if you are using [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench).


```
import sys

# Additional authentication is required for Google Colab
if "google.colab" in sys.modules:
    # Authenticate user to Google Cloud
    from google.colab import auth

    auth.authenticate_user()
```

### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and enable the Vertex AI API. Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).



```
# Define project information

PROJECT_ID = "your-project-id"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

# Initialize Vertex AI
import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

## Use case : Basic Text generation

The following code samples show the differences between the PaLM API and Gemini API for creating a text generation model.

#### Code sample using the PaLM API


```
from vertexai.language_models import TextGenerationModel

model = TextGenerationModel.from_pretrained("text-bison@002")

response = model.predict(prompt="The opposite of hot is")
print(response.text)
```

#### Code sample using the Gemini API

To load a Gemini model, replace the PaLM model class with the GenerativeModel class



```
from vertexai.generative_models import GenerativeModel
```


```
model = GenerativeModel("gemini-1.0-pro")
```

To generate text in Gemini, use the 'generate_content' method instead of the 'predict' method that's used in PaLM models.


```
response = model.generate_content("The opposite of hot is")


print(response.text)
```

## Use case : Text generation with parameters

The following code samples show the differences between the PaLM API and Gemini API for creating a text generation model, with optional parameters.

#### Code sample using the PaLM API


```
from vertexai.language_models import TextGenerationModel

model = TextGenerationModel.from_pretrained("text-bison@002")

prompt = """
You are an expert at solving word problems.

Solve the following problem:

I have three houses, each with three cats.
each cat owns 4 mittens, and a hat. Each mitten was
knit from 7m of yarn, each hat from 4m.
How much yarn was needed to make all the items?

Think about it step by step, and show your work.
"""

response = model.predict(
    prompt=prompt, temperature=0.1, max_output_tokens=800, top_p=1.0, top_k=40
)

print(response.text)
```

#### Code sample using the Gemini API


```
from vertexai.generative_models import GenerationConfig, GenerativeModel
```


```
model = GenerativeModel("gemini-1.0-pro")
```

To generate text in Gemini, use the generate_content method instead of the predict method that's used on PaLM models.


```
prompt = """
You are an expert at solving word problems.

Solve the following problem:

I have three houses, each with three cats.
each cat owns 4 mittens, and a hat. Each mitten was
knit from 7m of yarn, each hat from 4m.
How much yarn was needed to make all the items?

Think about it step by step, and show your work.
"""

generation_config = GenerationConfig(
    temperature=0.1,
    max_output_tokens=800,
    top_p=1.0,
    top_k=40,
)

gemini_response = model.generate_content(
    prompt,
    generation_config=generation_config,
)

print(gemini_response.text)
```

## Use case : Chat

The following code samples show the differences between the PaLM API and Gemini API for creating a chat model.

#### Code sample using the PaLM API


```
from vertexai.language_models import ChatModel

model = ChatModel.from_pretrained("chat-bison@002")

chat = model.start_chat()

response = chat.send_message(
    """
Hello! Can you write a 300 word abstract in paragraph format for a research paper I need to write about the impact of AI on society?
"""
)
print(response.text)

response = chat.send_message(
    """
Could you give me a catchy title for the paper?
"""
)
print(response.text)
```

#### Code sample using the Gemini API


```
from vertexai.generative_models import GenerativeModel

model = GenerativeModel("gemini-1.0-pro")

chat = model.start_chat()


response = chat.send_message(
    """
Hello! Can you write a 300 word abstract for a research paper I need to write about the impact of AI on society?
"""
)


print(response.text)


response = chat.send_message(
    """
Could you give me a catchy title for the paper?
"""
)


print(response.text)
```

## Use case : Code generation

The following code samples show the differences between the PaLM API and Gemini API for generating a function that predicts if a year is a leap year.

#### Code sample using the PaLM API


```
from vertexai.language_models import CodeGenerationModel

model = CodeGenerationModel.from_pretrained("code-bison@002")

response = model.predict(
    prefix="Write a function that checks if a year is a leap year."
)

print(response.text)
```

#### Code sample using the Gemini API


```
from vertexai.generative_models import GenerativeModel

model = GenerativeModel("gemini-1.0-pro-002")

response = model.generate_content(
    "Write a function that checks if a year is a leap year."
)

print(response.text)
```
