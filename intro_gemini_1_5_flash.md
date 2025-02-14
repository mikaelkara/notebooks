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

# Getting started with Vertex AI Gemini 1.5 Flash


<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_1_5_flash.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fgetting-started%2Fintro_gemini_1_5_flash.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/getting-started/intro_gemini_1_5_flash.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_1_5_flash.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Eric Dong](https://github.com/gericdong), [Holt Skinner](https://github.com/holtskinner) |

## Overview

Gemini 1.5 Flash is a new language model from the Gemini family. This model includes the long context window of up to 1 million tokens from Gemini 1.5 Pro and is optimized for low-latency tasks. It can process text, images, audio, video, and code all together for deeper insights. Learn more about [Gemini 1.5 Flash](https://deepmind.google/technologies/gemini/flash/).

With this tutorial, you learn how to use the Gemini API in Vertex AI and the Vertex AI SDK to work with the Gemini 1.5 Flash model to:

- analyze audio for insights.
- understand videos (including their audio components).
- extract information from PDF documents.
- process images, video, audio, and text simultaneously.

## Getting Started

### Install Vertex AI SDK for Python



```
%pip install --upgrade --user --quiet google-cloud-aiplatform
```

### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.


```
import sys

if "google.colab" in sys.modules:
    import IPython

    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the cell below to authenticate your environment.



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
# Use the environment variable if the user doesn't provide Project ID.
import os

import vertexai

PROJECT_ID = "[your-project-id]"  # @param {type: "string", placeholder: "[your-project-id]" isTemplate: true}
if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries



```
import IPython.display
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
```

### Load the Gemini 1.5 Flash model

To learn more about all [Gemini API models on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-models).



```
MODEL_ID = "gemini-1.5-flash-002"  # @param {type:"string"}

model = GenerativeModel(MODEL_ID)
```

### Vertex AI SDK basic usage

Below is a simple example that demonstrates how to prompt the Gemini 1.5 Flash model using the Vertex AI SDK. Learn more about the [Gemini API parameters](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini#gemini-pro).


```
# Load a example model with system instructions
example_model = GenerativeModel(
    MODEL_ID,
    system_instruction=[
        "You are a helpful language translator.",
        "Your mission is to translate text in English to French.",
    ],
)

# Set model parameters
generation_config = GenerationConfig(
    temperature=0.9,
    top_p=1.0,
    top_k=32,
    candidate_count=1,
    max_output_tokens=8192,
)

# Set safety settings
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

prompt = """
  User input: I like bagels.
  Answer:
"""

# Set contents to send to the model
contents = [prompt]

# Counts tokens
print(example_model.count_tokens(contents))

# Prompt the model to generate content
response = example_model.generate_content(
    contents,
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# Print the model response
print(f"\nAnswer:\n{response.text}")
print(f'\nUsage metadata:\n{response.to_dict().get("usage_metadata")}')
print(f"\nFinish reason:\n{response.candidates[0].finish_reason}")
print(f"\nSafety settings:\n{response.candidates[0].safety_ratings}")
```

## Audio understanding

Gemini 1.5 Flash can directly process audio for long-context understanding.



```
audio_file_path = "cloud-samples-data/generative-ai/audio/pixel.mp3"
audio_file_uri = f"gs://{audio_file_path}"
audio_file_url = f"https://storage.googleapis.com/{audio_file_path}"

IPython.display.Audio(audio_file_url)
```

#### Example 1: Summarization


```
prompt = """
  Please provide a summary for the audio.
  Provide chapter titles, be concise and short, no need to provide chapter summaries.
  Do not make up any information that is not part of the audio and do not be verbose.
"""

audio_file = Part.from_uri(audio_file_uri, mime_type="audio/mpeg")
contents = [audio_file, prompt]

response = model.generate_content(contents)
print(response.text)
```

#### Example 2: Transcription


```
prompt = """
    Can you transcribe this interview, in the format of speaker, caption.
    Use speaker A, speaker B, etc. to identify the speakers.
"""

audio_file = Part.from_uri(audio_file_uri, mime_type="audio/mpeg")
contents = [audio_file, prompt]

responses = model.generate_content(contents, stream=True)

for response in responses:
    print(response.text)
```

## Video with audio understanding

Try out Gemini 1.5 Flash's native multimodal and long context capabilities on video interleaving with audio inputs.


```
video_file_path = "cloud-samples-data/generative-ai/video/pixel8.mp4"
video_file_uri = f"gs://{video_file_path}"
video_file_url = f"https://storage.googleapis.com/{video_file_path}"

IPython.display.Video(video_file_url, width=450)
```


```
prompt = """
  Provide a description of the video.
  The description should also contain anything important which people say in the video.
"""

video_file = Part.from_uri(video_file_uri, mime_type="video/mp4")
contents = [video_file, prompt]

response = model.generate_content(contents)
print(response.text)
```

Gemini 1.5 Flash model is able to process the video with audio, retrieve and extract textual and audio information.

## PDF document analysis

You can use Gemini 1.5 Flash to process PDF documents, and analyze content, retain information, and provide answers to queries regarding the documents.

The PDF document example used here is the Gemini 1.5 paper (https://arxiv.org/pdf/2403.05530.pdf).

![image.png](https://storage.googleapis.com/cloud-samples-data/generative-ai/image/gemini1.5-paper-2403.05530.png)


```
pdf_file_uri = "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf"

prompt = """
  You are a very professional document summarization specialist.
  Please summarize the given document.
"""

pdf_file = Part.from_uri(pdf_file_uri, mime_type="application/pdf")
contents = [pdf_file, prompt]

response = model.generate_content(contents)
print(response.text)
```

## All modalities (images, video, audio, text) at once

Gemini 1.5 Flash is natively multimodal and supports interleaving of data from different modalities, it can support a mix of audio, visual, text, and
code inputs in the same input sequence.


```
video_file_path = "cloud-samples-data/generative-ai/video/behind_the_scenes_pixel.mp4"
video_file_uri = f"gs://{video_file_path}"
video_file_url = f"https://storage.googleapis.com/{video_file_path}"

IPython.display.Video(video_file_url, width=450)
```


```
image_file_path = "cloud-samples-data/generative-ai/image/a-man-and-a-dog.png"
image_file_uri = f"gs://{image_file_path}"
image_file_url = f"https://storage.googleapis.com/{image_file_path}"

IPython.display.Image(image_file_url, width=450)
```


```
video_file = Part.from_uri(video_file_uri, mime_type="video/mp4")
image_file = Part.from_uri(image_file_uri, mime_type="image/png")

prompt = """
  Look through each frame in the video carefully and answer the questions.
  Only base your answers strictly on what information is available in the video attached.
  Do not make up any information that is not part of the video and do not be too
  verbose, be to the point.

  Questions:
  - When is the moment in the image happening in the video? Provide a timestamp.
  - What is the context of the moment and what does the narrator say about it?
"""

contents = [video_file, image_file, prompt]

response = model.generate_content(contents)
print(response.text)
```

## Conclusion

In this tutorial, you've learned how to use Gemini 1.5 Flash with the Vertex AI SDK to:

- analyze audio for insights.
- understand videos (including their audio components).
- extract information from PDF documents.
- process images, video, audio, and text simultaneously.
