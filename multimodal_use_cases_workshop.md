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

# Gemini 1.5: A workshop in multimodal use cases

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/multimodal_use_cases_workshop.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fuse-cases%2Fmultimodal_use_cases_workshop.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/use-cases/multimodal_use_cases_workshop.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/multimodal_use_cases_workshop.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Katie Nguyen](https://github.com/katiemn) |

## Overview

### Gemini 1.5 Pro

Gemini 1.5 Pro is a new language model from the Gemini family. This model introduces a long context window of up to 1 million tokens that can seamlessly analyze large amounts of information. Additionally, it is multimodal with the ability to process text, images, audio, video, and code. Learn more about [Gemini 1.5 Pro](https://deepmind.google/technologies/gemini/pro/).

### Gemini 1.5 Flash

This smaller Gemini model is optimized for high-frequency tasks to prioritize the model's response time. This model has superior speed and efficiency with a context window of up to 1 million tokens for all modalities. Learn more about [Gemini 1.5 Flash](https://deepmind.google/technologies/gemini/flash/).

In this workshop tutorial, you will learn how to use the Vertex AI SDK for Python to interact with the Gemini 1.5 Pro and Gemini 1.5 Flash models to:
  - Cover individual text, PDF, image, video, code, and audio scenarios
  - Consider different modality combinations
  - Run through an e-commerce use case


## Getting Started


### Install Vertex AI SDK for Python



```
%pip install --upgrade --user google-cloud-aiplatform\
                                        gitpython \
                                        magika
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
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the following cell to authenticate your environment.



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

### Import libraries



```
from vertexai.generative_models import GenerativeModel, Image, Part
```

### Use the Gemini 1.5 models

Gemini 1.5 Pro and Gemini 1.5 Flash are multimodal models that support multimodal prompts. You can include text, image(s), and video in your prompt requests.



```
multimodal_model = GenerativeModel("gemini-1.5-pro")

multimodal_model_flash = GenerativeModel("gemini-1.5-flash")
```

### Define helper functions



```
import http.client
import typing
import urllib.request

from IPython.core.interactiveshell import InteractiveShell
import IPython.display

InteractiveShell.ast_node_interactivity = "all"


def get_image_bytes_from_url(image_url: str) -> bytes:
    with urllib.request.urlopen(image_url) as response:
        response = typing.cast(http.client.HTTPResponse, response)
        image_bytes = response.read()
    return image_bytes


def load_image_from_url(image_url: str) -> Image:
    image_bytes = get_image_bytes_from_url(image_url)
    return Image.from_bytes(image_bytes)


def display_content_as_video(content: str | Image | Part):
    if not isinstance(content, Part):
        return False
    part = typing.cast(Part, content)
    file_path = part.file_data.file_uri.removeprefix("gs://")
    video_url = f"https://storage.googleapis.com/{file_path}"
    IPython.display.display(IPython.display.Video(video_url, width=350))
```

## Individual Modalities

### Textual understanding

Gemini 1.5 Pro can parse textual questions and retain that context across following prompts.


```
question = "What is the average weather in Mountain View, CA in the middle of May?"
prompt = """
Considering the weather, please provide some outfit suggestions.

Give examples for the daytime and the evening.
"""

contents = [question, prompt]
response = multimodal_model.generate_content(contents)
display(IPython.display.Markdown(response.text))
```

### Document Summarization

You can use Gemini 1.5 Pro to process PDF documents, and analyze content, retain information, and provide answers to queries regarding the documents.

The PDF document example used here is the Gemini 1.5 paper (https://arxiv.org/pdf/2403.05530.pdf).

![image.png](https://storage.googleapis.com/cloud-samples-data/generative-ai/image/gemini1.5-paper-2403.05530.png)


```
pdf_file_uri = "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf"
pdf_file = Part.from_uri(pdf_file_uri, mime_type="application/pdf")

prompt = "How many tokens can the model process?"

contents = [pdf_file, prompt]

response = multimodal_model.generate_content(contents)
display(IPython.display.Markdown(response.text))
```


```
prompt = """
  You are a professional document summarization specialist.
  Please summarize the given document.
"""

contents = [pdf_file, prompt]

response = multimodal_model.generate_content(contents)
display(IPython.display.Markdown(response.text))
```

### Image understanding across multiple images

One of Gemini's capabilities is being able to reason across multiple images to provide recommendations.

This is an example using Gemini 1.5 Pro to reason which glasses would be more suitable for an oval face shape:



```
image_glasses1_url = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/glasses1.jpg"
image_glasses2_url = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/glasses2.jpg"
image_glasses1 = load_image_from_url(image_glasses1_url)
image_glasses2 = load_image_from_url(image_glasses2_url)

prompt = """
I have an oval face. Given my face shape, which glasses would be more suitable?

Explain how you reached this decision.
Provide your recommendation based on my face shape, and please give an explanation for each.
"""

IPython.display.Image(image_glasses1_url, width=150)
IPython.display.Image(image_glasses2_url, width=150)

contents = [prompt, image_glasses1, image_glasses2]
responses = multimodal_model.generate_content(contents)
display(IPython.display.Markdown(responses.text))
```

### Generating a video description

Gemini can also extract tags throughout a video:

> Video: https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/mediterraneansea.mp4


```
prompt = """
What is shown in this video?
Where should I go to see it?
What are the top 5 places in the world that look like this?
"""
video = Part.from_uri(
    uri="gs://github-repo/img/gemini/multimodality_usecases_overview/mediterraneansea.mp4",
    mime_type="video/mp4",
)
contents = [prompt, video]

responses = multimodal_model.generate_content(contents)

display_content_as_video(video)
display(IPython.display.Markdown(responses.text))
```

> You can confirm that the location is indeed Antalya, Turkey by visiting the Wikipedia page: https://en.wikipedia.org/wiki/Antalya


You can also use Gemini 1.5 Pro to retrieve extra information beyond the video contents.

> Video: https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/ottawatrain3.mp4



```
prompt = """
Which line is this?
Where does it go?
What are the stations/stops?
"""
video = Part.from_uri(
    uri="gs://github-repo/img/gemini/multimodality_usecases_overview/ottawatrain3.mp4",
    mime_type="video/mp4",
)
contents = [prompt, video]

responses = multimodal_model.generate_content(contents)

display_content_as_video(video)
display(IPython.display.Markdown(responses.text))
```

> You can confirm that this is indeed the Confederation Line on Wikipedia here: https://en.wikipedia.org/wiki/Confederation_Line


### Reason across a codebase

You will use the Online Boutique repo as an example in this notebook. Online Boutique is a cloud-first microservices demo application. The application is a web-based e-commerce app where users can browse items, add them to the cart, and purchase them. This application consists of 11 microservices across multiple languages.


```
# The GitHub repository URL
repo_url = "https://github.com/GoogleCloudPlatform/microservices-demo"  # @param {type:"string"}

# The location to clone the repo
repo_dir = "./repo"
```

#### Define helper functions for processing GitHub repository



```
import os
from pathlib import Path
import shutil

import git
import magika

m = magika.Magika()


def clone_repo(repo_url, repo_dir):
    """Clone a GitHub repository"""

    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    os.makedirs(repo_dir)
    git.Repo.clone_from(repo_url, repo_dir)


def extract_code(repo_dir):
    """Create an index, extract content of code/text files"""

    code_index = []
    code_text = ""
    for root, _, files in os.walk(repo_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, repo_dir)
            code_index.append(relative_path)

            file_type = m.identify_path(Path(file_path))
            if file_type.output.group in ("text", "code"):
                try:
                    with open(file_path) as f:
                        code_text += f"----- File: {relative_path} -----\n"
                        code_text += f.read()
                        code_text += "\n-------------------------\n"
                except Exception:
                    pass

    return code_index, code_text
```

#### Create an index and extract the contents of a codebase

Clone the repo and create an index and extract content of code/text files.


```
clone_repo(repo_url, repo_dir)

code_index, code_text = extract_code(repo_dir)
```

#### Define a helper function to generate a prompt to a code related question



```
def get_code_prompt(question):
    """Generates a prompt to a code related question."""

    prompt = f"""
    Questions: {question}

    Context:
    - The entire codebase is provided below.
    - Here is an index of all of the files in the codebase:
      \n\n{code_index}\n\n.
    - Then each of the files is concatenated together. You will find all of the code you need:
      \n\n{code_text}\n\n

    Answer:
  """

    return prompt
```

#### Create a developer getting started guide


```
question = """
  Provide a getting started guide to onboard new developers to the codebase.
"""

prompt = get_code_prompt(question)
contents = [prompt]

responses = multimodal_model.generate_content(contents, stream=True)
for response in responses:
    IPython.display.Markdown(response.text)
```

#### Finding bugs in the code


```
question = """
  Find the top 3 most severe issues in the codebase.
"""

prompt = get_code_prompt(question)
contents = [prompt]

responses = multimodal_model.generate_content(contents, stream=True)
for response in responses:
    IPython.display.Markdown(response.text)
```

#### Summarizing the codebase with Gemini 1.5 Flash


```
question = """
  Give me a summary of this codebase, and tell me the top 3 things that I can learn from it.
"""

prompt = get_code_prompt(question)
contents = [prompt]

# Generate text using non-streaming method
response = multimodal_model_flash.generate_content(contents)
IPython.display.Markdown(response.text)
```

### Audio understanding

Gemini 1.5 Pro can directly process audio for long-context understanding.


```
audio_file_path = "cloud-samples-data/generative-ai/audio/pixel.mp3"
audio_file_uri = f"gs://{audio_file_path}"
audio_file_url = f"https://storage.googleapis.com/{audio_file_path}"

IPython.display.Audio(audio_file_url)
```

#### Summarization


```
prompt = """
  Please provide a short summary and title for the audio.
  Provide chapter titles, be concise and short, no need to provide chapter summaries.
  Provide each of the chapter titles in a numbered list.
  Do not make up any information that is not part of the audio and do not be verbose.
"""

audio_file = Part.from_uri(audio_file_uri, mime_type="audio/mpeg")
contents = [audio_file, prompt]

response = multimodal_model.generate_content(contents)
IPython.display.Markdown(response.text)
```

#### Transcription using Gemini 1.5 Flash


```
prompt = """
    Can you transcribe this interview, in the format of timecode, speaker, caption.
    Use speaker A, speaker B, etc. to identify the speakers.
    Please provide each piece of information on a separate bullet point.
"""

audio_file = Part.from_uri(audio_file_uri, mime_type="audio/mpeg")
contents = [audio_file, prompt]

responses = multimodal_model_flash.generate_content(contents)

IPython.display.Markdown(responses.text)
```

## Combining multiple modalities

### Video and audio understanding


Try out Gemini 1.5 Pro's native multimodal and long context capabilities on video interleaving with audio inputs.


```
video_file_path = "cloud-samples-data/generative-ai/video/pixel8.mp4"
video_file_uri = f"gs://{video_file_path}"
video_file_url = f"https://storage.googleapis.com/{video_file_path}"

IPython.display.Video(video_file_url, width=350)
```


```
prompt = """
  Provide a description of the video.
  The description should also contain any important dialogue from the video.
"""

video_file = Part.from_uri(video_file_uri, mime_type="video/mp4")
contents = [video_file, prompt]

response = multimodal_model.generate_content(contents)
IPython.display.Markdown(response.text)
```

### All modalities (images, video, audio, text) at once

Gemini 1.5 Pro is natively multimodal and supports interleaving of data from different modalities. It can support a mix of audio, visual, text, and code inputs in the same input sequence.


```
video_file_path = "cloud-samples-data/generative-ai/video/behind_the_scenes_pixel.mp4"
video_file_uri = f"gs://{video_file_path}"
video_file_url = f"https://storage.googleapis.com/{video_file_path}"

IPython.display.Video(video_file_url, width=350)
```


```
image_file_path = "cloud-samples-data/generative-ai/image/a-man-and-a-dog.png"
image_file_uri = f"gs://{image_file_path}"
image_file_url = f"https://storage.googleapis.com/{image_file_path}"

IPython.display.Image(image_file_url, width=350)
```


```
video_file = Part.from_uri(video_file_uri, mime_type="video/mp4")
image_file = Part.from_uri(image_file_uri, mime_type="image/png")

prompt = """
  Look through each frame in the video carefully and answer the questions.
  Only base your answers strictly on what information is available in the video attached.
  Do not make up any information that is not part of the video and do not be too
  verbose, be straightforward.

  Questions:
  - When is the moment in the image happening in the video? Provide a timestamp.
  - What is the context of the moment and what does the narrator say about it?
"""

contents = [video_file, image_file, prompt]

response = multimodal_model.generate_content(contents)
IPython.display.Markdown(response.text)
```

## Use Case: retail / e-commerce

The customer shows you their living room:

|Customer photo |
|:-----:|
|<img src="https://storage.googleapis.com/cloud-samples-data/generative-ai/image/living-room.png" width="50%">  |



Below are four wall art options that the customer is trying to decide between:

|Art 1| Art 2 | Art 3 | Art 4 |
|:-----:|:----:|:-----:|:----:|
| <img src="https://storage.googleapis.com/cloud-samples-data/generative-ai/image/room-art-1.png" width="60%">|<img src="https://storage.googleapis.com/cloud-samples-data/generative-ai/image/room-art-2.png" width="100%">|<img src="https://storage.googleapis.com/cloud-samples-data/generative-ai/image/room-art-3.png" width="60%">|<img src="https://storage.googleapis.com/cloud-samples-data/generative-ai/image/room-art-4.png" width="60%">|


How can you use Gemini 1.5 Pro, a multimodal model, to help the customer choose the best option?

### Generating open recommendations

Using the same image, you can ask the model to recommend a piece of furniture that would make sense in the space.

Note that the model can choose any furniture in this case, and can do so only from its built-in knowledge.


```
# urls for room images
room_image_url = "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/living-room.png"

# load room images as Image Objects
room_image = load_image_from_url(room_image_url)

prompt = "Describe this room"
contents = [prompt, room_image]

IPython.display.Image(room_image_url, width=350)
responses = multimodal_model.generate_content(contents)
IPython.display.Markdown(responses.text)
```


```
prompt1 = "Recommend a new piece of furniture for this room"
prompt2 = "Explain the reason in detail"
contents = [prompt1, room_image, prompt2]

responses = multimodal_model.generate_content(contents)
IPython.display.Markdown(responses.text)
```

### Generating recommendations based on provided images

Instead of keeping the recommendation open, you can also provide a list of items for the model to choose from. Here, you will download a few art images that the Gemini model can recommend. This is particularly useful for retail companies who want to provide product recommendations to users based on their current setup.


```
# Download and display sample artwork
art_image_urls = [
    "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/room-art-1.png",
    "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/room-art-2.png",
    "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/room-art-3.png",
    "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/room-art-4.png",
]

# Load wall art images as Image Objects
art_images = [load_image_from_url(url) for url in art_image_urls]

# To recommend an item from a selection, you will need to label the item number within the prompt.
# That way you are providing the model with a way to reference each image as you pose a question.
# Labeling images within your prompt also helps reduce hallucinations and produce better results.
prompt = """
  You are an interior designer.
  For each piece of wall art, explain whether it would be appropriate for the style of the room.
  Rank each piece according to how well it would be compatible in the room.
"""
contents = [
    "Consider the following art pieces:",
    "art 1:",
    art_images[0],
    "art 2:",
    art_images[1],
    "art 3:",
    art_images[2],
    "art 4:",
    art_images[3],
    "room:",
    room_image,
    prompt,
]

IPython.display.Image(room_image_url, width=350)
print("\n------Art1:-------")
IPython.display.Image(art_image_urls[0], width=150)
print("\n------Art2:-------")
IPython.display.Image(art_image_urls[1], width=150)
print("\n------Art3:-------")
IPython.display.Image(art_image_urls[2], width=150)
print("\n------Art4:-------")
IPython.display.Image(art_image_urls[3], width=150)

responses = multimodal_model.generate_content(contents)
IPython.display.Markdown(responses.text)
```
