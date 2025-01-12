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

# Basic prototyping of the solution with Gemini API in Vertex AI


<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/workshops/rag-ops/1_prototyping_gemini.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fworkshops%2Frag-ops%2F1_prototyping_gemini.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/workshops/rag-ops/1_prototyping_gemini.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/workshops/rag-ops/1_prototyping_gemini.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

## Overview

Gemini 1.5 Pro is a new language model from the Gemini family. This model introduces a breakthrough long context window of up to 1 million tokens that can help seamlessly analyze large amounts of information and long-context understanding. It can process text, images, audio, video, and code all together for deeper insights. Learn more about [Gemini 1.5](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/).

With this tutorial, you learn how to use the Gemini API in Vertex AI and the Vertex AI SDK to work with the Gemini 1.5 Pro model to:

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
import os
import sys

import vertexai

PROJECT_ID = "[your-project-id]"  # @param {type:"string", isTemplate: true}

if PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

if not PROJECT_ID or PROJECT_ID == "[your-project-id]" or PROJECT_ID == "None":
    raise ValueError("Please set your PROJECT_ID")

print(f"Project ID: {PROJECT_ID}")

LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries



```
from IPython.core.interactiveshell import InteractiveShell
import IPython.display

InteractiveShell.ast_node_interactivity = "all"

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

The Gemini model family has several model versions. You will start by using Gemini 1.5 Flash. Gemini 1.5 Flash is a more lightweight, fast, and cost-efficient model. This makes it a great option for prototyping.



```
MODEL_ID = "gemini-1.5-flash-002"  # @param {type:"string"}

model = GenerativeModel(MODEL_ID)
```

## Vertex AI SDK basic usage

Prototyping starts with calling the API to see if we get a response. Lets learn some of the funamentals of the Vertex AI Gemini model.

Below is a simple example that demonstrates how to prompt the Gemini 1.5 Pro model using the Vertex AI SDK. Learn more about the [Gemini API parameters](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini#gemini-pro).


```
%time

# Load a example model with system instructions
example_model = GenerativeModel(
    MODEL_ID,
    system_instruction=[
        "You are a financial analyst specialized in earnings reports",
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
  User input: Explain the steps you would take to review a companies earnings report?
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

### Enable streaming

You can also enable streaming to receive the output while its being generated.


```
%%time
response = example_model.generate_content(
    contents,
    stream=True,
    generation_config=generation_config,
    safety_settings=safety_settings,
)

for chunk in response:
    print(chunk.text)
    print("_" * 80)
```

### Token count
**Important:** A token is equivalent to about 4 characters for Gemini models. 100 tokens are about 60-80 English words.
[Cloud pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) is done on characters and not tokens.

Using token count can help you with understanding the cost of running your prompt.


```
# Get token count
response = model.count_tokens(prompt)
print(f"Prompt Token Count: {response.total_tokens}")
print(f"Prompt Character Count: {response.total_billable_characters}")
```

### Prompt design fundamentals
Here are some fundamentals for designing your prompts throughout this workshop:
- Be specific in your instructions: Craft clear and concise instructions that leave minimal room for misinterpretation.
- - Add a few examples to your prompt: Use realistic few-shot examples to illustrate what you want to achieve.
Break it down step-by-step: Divide complex tasks into manageable sub-goals, guiding the model through the process.
- Specify the output format: In your prompt, ask for the output to be in the format you want, like markdown, JSON, HTML and more.
- Put your image first for single-image prompts: While Gemini can handle image and text inputs in any order, for prompts containing a single image, it might perform better if that image (or video) is placed before the text prompt. However, for prompts that require images to be highly interleaved with texts to make sense, use whatever order is most natural.

## Working with multimodal data

Lets now dive into the world of multimodal where we will use different types of data, like a pdf and audio, and ask Gemini to reasons across these modalities.

### Text and PDF

Next, you can ask a question about the PDF document. You will use the PDF format of an earnings report. To answer this question the model has to process the PDF and find the information in the pdf.


```
ROLE = """
You are a financial analyst. You specialize in Give me a summary of the earnings report
Only base your answer on the information provided.
"""
```


```
question_1 = "Question: How many shares of each class of Alphabet stock were outstanding as of July 22, 2022?"
```

Next you can combine the prompts with the PDF and send it to the Gemini API.


```
pdf_file_uri = "gs://mlops-for-genai/multimodal-finanace-qa/data/unstructured/prototype/goog-10-q-q2-2023-4-1-15.pdf"
pdf_file = Part.from_uri(pdf_file_uri, mime_type="application/pdf")
contents = [pdf_file, ROLE, question_1]

response = model.generate_content(contents)
print(response.text)
```


```
question_2 = (
    "Summarize the earnings call provided and provide us with your top 3 take aways"
)

audio_file_uri = "gs://mlops-for-genai/multimodal-finanace-qa/data/unstructured/prototype/alphabet_2023_q4_earnings_call.mp3"
audio_file = Part.from_uri(audio_file_uri, mime_type="audio/mpeg")

contents = [audio_file, ROLE, question_2]

response = model.generate_content(contents)
print(response.text)
```

### Troubleshooting your multimodal prompt
While prototyping you also might have to troubleshoot your multimodal prompts. Here are some tips:

- If the model is not drawing information from the relevant part of the image: Drop hints with which aspects of the image you want the prompt to draw information from.
- If the model output is too generic (not tailored enough to the image/video input): At the start of the prompt, try asking the model to describe the image(s) or video before providing the task instruction, or try asking the model to refer to what's in the image.
- To troubleshoot which part failed: Ask the model to describe the image, or ask the model to explain its reasoning, to gauge the model's initial understanding.
- If your prompt results in hallucinated content: Try dialing down the temperature setting or asking the model for shorter descriptions so that it's less likely to extrapolate additional details.
- Tuning the sampling parameters: Experiment with different temperature settings and top-k selections to adjust the model's creativity.

## Advanced reasoning across text, audio, video and PDF.

Let's do something more complex. You will now let the model reason across different modalities to find the answer to the question: audio, PDF, and video.

Lets also switch models and use `gemini-1.5-pro-001`


```
MODEL_ID = "gemini-1.5-pro-001"
```


```
ROLE = "You are a financial analyst at a hedge fund"
question_2 = "How is Google using AI to optimize advertisement experience and what strategic partnership did Google announce in February 2023 in the automotive industry??"

audio_uri_1 = "gs://mlops-for-genai/multimodal-finanace-qa/data/unstructured/prototype/Alphabet_2023_Q1_Earnings_Call.mp3"
audio_uri_2 = "gs://mlops-for-genai/multimodal-finanace-qa/data/unstructured/prototype/episode1.mp3"

pdf_uri_2 = "gs://mlops-for-genai/multimodal-finanace-qa/data/unstructured/prototype/20230426-alphabet-10q.pdf"
video_uri_1 = "gs://mlops-for-genai/multimodal-finanace-qa/data/unstructured/prototype/changing-the-way-scientists-research-Gemini.mp4"

pdf_file = Part.from_uri(pdf_uri_2, mime_type="application/pdf")
video_file = Part.from_uri(video_uri_1, mime_type="video/mp4")
audio_file_1 = Part.from_uri(audio_uri_1, mime_type="audio/mpeg")
audio_file_2 = Part.from_uri(audio_uri_2, mime_type="audio/mpeg")

contents = [audio_file_1, audio_file_2, video_file, pdf_file, ROLE, question_2]

response = model.generate_content(contents)
print(response.text)
```

## Experimentation

Now its time for you to improve the prompts. You can update the prompts below and re-run the code to see the response from Gemini.


```
ROLE = "<update-this>"
question_2 = "<update-this>"
```


```
audio_uri_1 = "gs://mlops-for-genai/multimodal-finanace-qa/data/unstructured/prototype/Alphabet_2023_Q1_Earnings_Call.mp3"
audio_uri_2 = "gs://mlops-for-genai/multimodal-finanace-qa/data/unstructured/prototype/episode1.mp3"

pdf_uri_2 = "gs://mlops-for-genai/multimodal-finanace-qa/data/unstructured/prototype/20230426-alphabet-10q.pdf"
video_uri_1 = "gs://mlops-for-genai/multimodal-finanace-qa/data/unstructured/prototype/changing-the-way-scientists-research-Gemini.mp4"

pdf_file = Part.from_uri(pdf_uri_2, mime_type="application/pdf")
video_file = Part.from_uri(video_uri_1, mime_type="video/mp4")
audio_file_1 = Part.from_uri(audio_uri_1, mime_type="audio/mpeg")
audio_file_2 = Part.from_uri(audio_uri_2, mime_type="audio/mpeg")

# contents = [audio_file_1, audio_file_2, video_file, pdf_file, ROLE, question_2]
contents = [audio_file_1, audio_file_2, ROLE, question_2]

response = model.generate_content(contents)
print(response.text)
```
