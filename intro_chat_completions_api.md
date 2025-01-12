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

# Call Gemini by using the OpenAI Library

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/chat-completions/intro_chat_completions_api.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fchat-completions%2Fintro_chat_completions_api.ipynb">
      <img width="32px" src="https://cloud.google.com/ml-engine/images/colab-enterprise-logo-32px.png" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/chat-completions/intro_chat_completions_api.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/chat-completions/intro_chat_completions_api.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Eric Dong](https://github.com/gericdong)|

## Overview

Developers already working with OpenAI's libraries can easily tap into the power of Gemini by leveraging the Gemini Chat Completions API. The Gemini Chat Completions API offers a streamlined way to experiment with and incorporate Gemini's capabilities into your existing AI applications. Learn more about [calling Gemini by using the OpenAI library](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-gemini-using-openai-library).

In this tutorial, you learn how to call Gemini using the OpenAI library. You will complete the following tasks:

- Configure OpenAI SDK for the Gemini Chat Completions API
- Send a chat completions request
- Stream chat completions response
- Send a multimodal request
- Send a function calling request
- Send a function calling request with the `tool_choice` parameter


## Get started

### Install required packages



```
%pip install --upgrade --quiet openai
```

### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.


```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. The restart might take a minute or longer. After it's restarted, continue to the next step. ⚠️</b>
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

Learn more about [setting up a project and development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
```

## Chat Completions API Examples

### Configure OpenAI SDK for the Gemini Chat Completions API


#### Import libraries

The `google-auth` library is used to programmatically get Google credentials. Colab already has this library pre-installed. If you don't have it in your environment, use the following command to install it.

`%pip install google-auth requests`


```
from google.auth import default, transport
import openai
```

#### Authentication

You can request an access token from the default credentials for the current environment. Note that the access token lives for 1 hour by default (https://cloud.google.com/docs/authentication/token-types#at-lifetime); after expiration, it must be refreshed.



```
credentials, _ = default()
auth_request = transport.requests.Request()
credentials.refresh(auth_request)
```

Then configure the OpenAI SDK to point to the Gemini Chat Completions API endpoint.


```
client = openai.OpenAI(
    base_url=f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi",
    api_key=credentials.token,
)
```

#### Gemini models

You can experiment with [various supported Gemini models](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-gemini-using-openai-library#supported_models). If you are not sure which model to use, then try `gemini-1.5-flash`. `gemini-1.5-flash` is optimized for multimodal use cases where speed and cost are important.


```
MODEL_ID = "google/gemini-1.5-flash"  # @param {type:"string"}
```

### [Example] Send a chat completions request

The Chat Completions API takes a list of messages as input and returns a generated message as output. Although the message format is designed to make multi-turn conversations easy, it's just as useful for single-turn tasks without any conversation.

In this example, you use the Chat Completions API to send a request to the Gemini model.


```
response = client.chat.completions.create(
    model=MODEL_ID, messages=[{"role": "user", "content": "Why is the sky blue?"}]
)
```

An example Chat Completions API response looks as follows:


```
print(response)
```

The generated content can be extracted with:


```
response.choices[0].message.content
```

You can use `Markdown` to display the formatted text.


```
from IPython.display import Markdown

Markdown(response.choices[0].message.content)
```

### [Example] Stream chat completions response

By default, the model returns a response after completing the entire generation process. You can also stream the response as it is being generated, and the model will return chunks of the response as soon as they are generated.


```
response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content)
```

### [Example] Send a multimodal request
You can send a multimodal prompt in a request to Gemini and get a text output.

In this example, you ask the model to create a blog post for [this image](https://storage.googleapis.com/cloud-samples-data/generative-ai/image/meal.png) stored in a Cloud Storage bucket.



```
response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Write a short, engaging blog post based on this picture.",
                },
                {
                    "type": "image_url",
                    "image_url": "gs://cloud-samples-data/generative-ai/image/meal.png",
                },
            ],
        }
    ],
)

print(response.choices[0].message.content)
```

### [Example] Send a function calling request

You can use the Chat Completions API for function calling with the Gemini models. The `tools` parameter in the API is used to provide function specifications. This is to enable models to generate function arguments which adhere to the provided specifications.

In this example, you create function specifications to interface with a hypothetical weather API, then pass these function specifications to the Chat Completions API to generate function arguments that adhere to the specification.

**Note** that in this example, the API will not actually execute any function calls. It is up to developers to execute function calls using model outputs.


```
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

messages = []
messages.append(
    {
        "role": "system",
        "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
    }
)
messages.append({"role": "user", "content": "What is the weather in Boston?"})

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=messages,
    tools=tools,
)

print(print(response.choices[0].message.tool_calls))
```

### [Example] Send a function calling request with the `tool_choice` parameter

Using the `tools` parameter, if the functions parameter is provided then by default the model will decide when it is appropriate to use one of the functions.

By default, `tool_choice` is set to `auto`. This lets the model decide whether to call functions and, if so, which functions to call. To disable function calling and force the model to only generate a user-facing message, you can set `tool_choice` to `none`.


```
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

messages = []
messages.append(
    {
        "role": "system",
        "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
    }
)
messages.append({"role": "user", "content": "What is the weather in Boston?"})

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

print(response.choices[0].message.tool_calls)
```
