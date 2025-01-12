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

# Getting Started with the Gemini API in Vertex AI with cURL

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_curl.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fgetting-started%2Fintro_gemini_curl.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>       
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_curl.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/getting-started/intro_gemini_curl.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Eric Dong](https://github.com/gericdong), [Polong Lin](https://github.com/polong-lin) |

## Overview

### Gemini
Gemini is a family of generative AI models developed by Google DeepMind. Gemini models support prompts that include text, image, and video as input and support text responses as output.

### Gemini API in Vertex AI

The Gemini API in Vertex AI provides a unified interface for interacting with Gemini models. You can interact with the Gemini API by using the following methods:

* Use [Vertex AI Studio](https://cloud.google.com/generative-ai-studio) for quick testing and command generation.
* Use cURL commands in Cloud Shell.
* Use the Vertex AI SDK for Python in a Jupyter notebook

This notebook focuses on using the **cURL commands** to call the Gemini API in Vertex AI.

For more information, see the [Generative AI on Vertex AI](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview) documentation.

### Objectives

In this tutorial, you learn how to use the Gemini API in Vertex AI with cURL commands to interact with the Gemini 1.5 Pro (`gemini-1.5-pro`) model.

You will complete the following tasks:

- Install the Python SDK.
- Use the Gemini API in Vertex AI to interact with each model.
  - Gemini 1.5 Pro (`gemini-1.5-pro`) model:
    - Generate text from text prompts.
    - Explore various features and configuration options.
    - Generate text from image(s) and text prompts.
    - Generate text from video.
  

### Costs
This tutorial uses billable components of Google Cloud:

- Vertex AI

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.

## Getting Started

### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the following cell to authenticate your environment.

This step is not required if you are using [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench).


```
import sys

# Additional authentication is required for Google Colab
if "google.colab" in sys.modules:
    # Authenticate user to Google Cloud
    from google.colab import auth

    auth.authenticate_user()
```

### Set Google Cloud project

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
PROJECT_ID = "[your-project_id]"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
```

### Defining environment variables for cURL commands

These environment variables are used to construct the cURL commands.


```
import os

os.environ["PROJECT_ID"] = PROJECT_ID
os.environ["LOCATION"] = LOCATION
os.environ["API_ENDPOINT"] = f"{LOCATION}-aiplatform.googleapis.com"
```

## Use the Gemini 1.5 Pro model


```
os.environ["MODEL_ID"] = "gemini-1.5-pro"
```

### Generate content

The generateContent method can handle a wide variety of use cases, including multi-turn chat and multimodal input, depending on what the underlying model supports. In this example, you send a text prompt using the `generateContent` method.


```bash
%%bash

curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://${API_ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:generateContent \
  -d '{
    "contents": {
      "role": "USER",
      "parts": { "text": "Why is the sky blue?" }
    }
  }'

```

### Streaming

The Gemini API provides a streaming response mechanism. With this approach, you don't need to wait for the complete response; you can start processing fragments as soon as they're accessible.


```bash
%%bash

curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://${API_ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:streamGenerateContent \
  -d '{
    "contents": {
      "role": "USER",
      "parts": { "text": "Why is the sky blue?" }
    }
  }'

```

### Model parameters

Every prompt you send to the model includes parameter values that control how the model generates a response. The model can generate different results for different parameter values. You can experiment with different model parameters to see how the results change. 


```bash
%%bash

curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://${API_ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:generateContent \
  -d '{
    "contents": {
      "role": "USER",
      "parts": [
        {"text": "Describe this image"},
        {"file_data": {
          "mime_type": "image/png",
          "file_uri": "gs://cloud-samples-data/generative-ai/image/320px-Felis_catus-cat_on_snow.jpg"
        }}
      ]
    },
    "generation_config": {
      "temperature": 0.2,
      "top_p": 0.1,
      "top_k": 16,
      "max_output_tokens": 2048,
      "candidate_count": 1,
      "stop_sequences": []
    },
    "safety_settings": {
      "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
      "threshold": "BLOCK_LOW_AND_ABOVE"
    }
  }'

```

### Chat

The Gemini API supports natural multi-turn conversations and is ideal for text tasks that require back-and-forth interactions.

Specify the `role` field only if the content represents a turn in a conversation. You can set `role` to one of the following values: `user`, `model`.


```bash
%%bash

curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://${API_ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:generateContent \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [
          { "text": "Hello" }
        ]
      },
      {
        "role": "model",
        "parts": [
          { "text": "Hello! I am glad you could both make it." }
        ]
      },
      {
        "role": "user",
        "parts": [
          { "text": "So what is the first order of business?" }
        ]
      }
    ]
  }'
```

### Function calling

Function calling lets you create a description of a function in their code, then pass that description to a language model in a request. This sample is an example of passing in a description of a function that returns information about where a movie is playing. Several function declarations are included in the request, such as `find_movies` and `find_theaters`.

Learn more about [function calling](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/function-calling).


```bash
%%bash

curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://${API_ENDPOINT}/v1beta1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:generateContent \
  -d '{
  "contents": {
    "role": "user",
    "parts": {
      "text": "Which theaters in Mountain View show Barbie movie?"
    }
  },
  "tools": [
    {
      "function_declarations": [
        {
          "name": "find_movies",
          "description": "find movie titles currently playing in theaters based on any description, genre, title words, etc.",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
              },
              "description": {
                "type": "string",
                "description": "Any kind of description including category or genre, title words, attributes, etc."
              }
            },
            "required": [
              "description"
            ]
          }
        },
        {
          "name": "find_theaters",
          "description": "find theaters based on location and optionally movie title which are is currently playing in theaters",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
              },
              "movie": {
                "type": "string",
                "description": "Any movie title"
              }
            },
            "required": [
              "location"
            ]
          }
        },
        {
          "name": "get_showtimes",
          "description": "Find the start times for movies playing in a specific theater",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
              },
              "movie": {
                "type": "string",
                "description": "Any movie title"
              },
              "theater": {
                "type": "string",
                "description": "Name of theater"
              },
              "date": {
                "type": "string",
                "description": "Date for requested showtime"
              }
            },
            "required": [
              "location",
              "movie",
              "theater",
              "date"
            ]
          }
        }
      ]
    }
  ]
}'
```

## Multimodal input

The Gemini 1.5 Pro  (`gemini-1.5-pro`) is a multimodal model that supports adding image and video in text or chat prompts for a text response.


### Download an image from Google Cloud Storage


```
! gsutil cp "gs://cloud-samples-data/generative-ai/image/320px-Felis_catus-cat_on_snow.jpg" ./image.jpg
```

### Generate text from a local image

Specify the [base64](https://en.wikipedia.org/wiki/Base64) encoding of the image or video to include inline in the prompt and the `mime_type` field. The supported [MIME types](https://en.wikipedia.org/wiki/Media_type) for images include `image/png` and `image/jpeg`.


```bash
%%bash

# Encode image data in base64
# NOTE: This command only works on Linux.
data=$(base64 -w 0 image.jpg)

curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://${API_ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:generateContent \
  -d "{
      'contents': {
        'role': 'USER',
        'parts': [
          {
            'text': 'Is it a cat?'
          },
          {
            'inline_data': {
              'data': '${data}',
              'mime_type':'image/jpeg'
            }
          }
        ]
       }
     }"
```

### Generate text from an image on Google Cloud Storage

Specify the Cloud Storage URI of the image to include in the prompt. The bucket that stores the file must be in the same Google Cloud project that's sending the request. You must also specify the `mime_type` field. The supported image MIME types include `image/png` and `image/jpeg`.


```bash
%%bash

MODEL_ID="gemini-1.5-pro"

curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://${API_ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:generateContent \
  -d '{
    "contents": {
      "role": "USER",
      "parts": [
        {
          "text": "Describe this image"
        },
        {
          "file_data": {
            "mime_type": "image/png",
            "file_uri": "gs://cloud-samples-data/generative-ai/image/320px-Felis_catus-cat_on_snow.jpg"
          }
        }
      ]
    },
    "generation_config": {
      "temperature": 0.2,
      "top_p": 0.1,
      "top_k": 16,
      "max_output_tokens": 2048,
      "candidate_count": 1,
      "stop_sequences": []
    },
    "safety_settings": {
      "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
      "threshold": "BLOCK_LOW_AND_ABOVE"
    }
  }'
```

### Generate text from a video file

Specify the Cloud Storage URI of the video to include in the prompt. The bucket that stores the file must be in the same Google Cloud project that's sending the request. You must also specify the `mime_type` field. The supported MIME types for video include `video/mp4`.



```bash
%%bash

curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://${API_ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:generateContent \
  -d \
'{
    "contents": {
      "role": "USER",
      "parts": [
        {
          "text": "Answer the following questions using the video only. What is the profession of the main person? What are the main features of the phone highlighted?Which city was this recorded in?Provide the answer JSON."
        },
        {
          "file_data": {
            "mime_type": "video/mp4",
            "file_uri": "gs://github-repo/img/gemini/multimodality_usecases_overview/pixel8.mp4"
          }
        }
      ]
    }
  }'
```
