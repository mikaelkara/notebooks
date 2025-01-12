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

# Getting Started with the Gemini Pro Vision Model

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_pro_vision_python.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fgetting-started%2Fintro_gemini_pro_vision_python.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>       
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_pro_vision_python.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/getting-started/intro_gemini_pro_vision_python.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Eric Dong](https://github.com/gericdong), [Polong Lin](https://github.com/polong-lin), [Wanheng Li](https://github.com/wanhengli) |

## Overview

### Gemini

Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases. The Gemini API gives you access to the Gemini Pro Vision and Gemini Pro models.

### Gemini API in Vertex AI

The Gemini API in Vertex AI provides a unified interface for interacting with Gemini models. There are two Gemini 1.0 Pro models available in the Gemini API:

- **Gemini 1.0 Pro model** (`gemini-1.0-pro`): Designed to handle natural language tasks, multi-turn text and code chat, and code generation.
- **Gemini 1.0 Pro Vision model** (`gemini-1.0-pro-vision`): Supports multimodal prompts. You can include text, images, and video in your prompt requests and get text or code responses.

You can interact with the Gemini API using the following methods:

- Use [Vertex AI Studio](https://cloud.google.com/generative-ai-studio) for quick testing and command generation
- Use cURL commands
- Use the Vertex AI SDK

This notebook focuses on using the **Vertex AI SDK for Python** to call the Gemini API in Vertex AI with the Gemini 1.0 Pro Vision model.

For more information, see the [Generative AI on Vertex AI](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview) documentation.


### Objectives

In this tutorial, you will learn how to use the Gemini API in Vertex AI with the Vertex AI SDK for Python to interact with the Gemini 1.0 Pro Vision (`gemini-1.0-pro-vision`) model.

You will complete the following tasks:

- Install the Vertex AI SDK for Python
- Use the Gemini API in Vertex AI to interact with Gemini 1.0 Pro Vision (`gemini-1.0-pro-vision`) model:
  - Generate text from image(s) and text prompts
  - Generate text from video and text prompts


### Costs

This tutorial uses billable components of Google Cloud:

- Vertex AI

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.


## Getting Started


### Install Vertex AI SDK for Python



```
%pip install --upgrade --user google-cloud-aiplatform
```

### Restart current runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.


```
# Restart kernel after installs so that your environment can access the new packages
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the following cell to authenticate your environment. This step is not required if you are using [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench).



```
import sys

# Additional authentication is required for Google Colab
if "google.colab" in sys.modules:
    # Authenticate user to Google Cloud
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
from vertexai.generative_models import GenerativeModel, Image, Part
```

## Use the Gemini 1.0 Pro Vision model

Gemini 1.0 Pro Vision (`gemini-1.0-pro-vision`) is a multimodal model that supports multimodal prompts. You can include text, image(s), and video in your prompt requests and get text or code responses.


### Load the Gemini 1.0 Pro Vision model



```
multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
```

### Define helper functions

Define helper functions to load and display images.



```
import http.client
import typing
import urllib.request

import IPython.display
from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps


def display_images(
    images: typing.Iterable[Image],
    max_width: int = 600,
    max_height: int = 350,
) -> None:
    for image in images:
        pil_image = typing.cast(PIL_Image.Image, image._pil_image)
        if pil_image.mode != "RGB":
            # RGB is supported by all Jupyter environments (e.g. RGBA is not yet)
            pil_image = pil_image.convert("RGB")
        image_width, image_height = pil_image.size
        if max_width < image_width or max_height < image_height:
            # Resize to display a smaller notebook image
            pil_image = PIL_ImageOps.contain(pil_image, (max_width, max_height))
        IPython.display.display(pil_image)


def get_image_bytes_from_url(image_url: str) -> bytes:
    with urllib.request.urlopen(image_url) as response:
        response = typing.cast(http.client.HTTPResponse, response)
        image_bytes = response.read()
    return image_bytes


def load_image_from_url(image_url: str) -> Image:
    image_bytes = get_image_bytes_from_url(image_url)
    return Image.from_bytes(image_bytes)


def get_url_from_gcs(gcs_uri: str) -> str:
    # converts GCS uri to url for image display.
    url = "https://storage.googleapis.com/" + gcs_uri.replace("gs://", "").replace(
        " ", "%20"
    )
    return url


def print_multimodal_prompt(contents: list):
    """
    Given contents that would be sent to Gemini,
    output the full multimodal prompt for ease of readability.
    """
    for content in contents:
        if isinstance(content, Image):
            display_images([content])
        elif isinstance(content, Part):
            url = get_url_from_gcs(content.file_data.file_uri)
            IPython.display.display(load_image_from_url(url))
        else:
            print(content)
```

### Generate text from local image and text

Use the `Image.load_from_file` method to load a local file as the image to generate text for.



```
# Download an image from Google Cloud Storage
! gsutil cp "gs://cloud-samples-data/generative-ai/image/320px-Felis_catus-cat_on_snow.jpg" ./image.jpg

# Load from local file
image = Image.load_from_file("image.jpg")

# Prepare contents
prompt = "Describe this image?"
contents = [image, prompt]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")
```

    Copying gs://cloud-samples-data/generative-ai/image/320px-Felis_catus-cat_on_snow.jpg...
    / [1 files][ 17.4 KiB/ 17.4 KiB]                                                
    Operation completed over 1 objects/17.4 KiB.                                     
    -------Prompt--------
    


    
![png](output_24_1.png)
    


    Describe this image?
    
    -------Response--------
     A cat with gray and black stripes is walking in the snow.

### Generate text from text & image(s)


#### Images with Cloud Storage URIs

If your images are stored in [Cloud Storage](https://cloud.google.com/storage/docs), you can specify the Cloud Storage URI of the image to include in the prompt. You must also specify the `mime_type` field. The supported MIME types for images include `image/png` and `image/jpeg`.

Note that the URI (not to be confused with URL) for a Cloud Storage object should always start with `gs://`.


```
# Load image from Cloud Storage URI
gcs_uri = "gs://cloud-samples-data/generative-ai/image/boats.jpeg"

# Prepare contents
image = Part.from_uri(gcs_uri, mime_type="image/jpeg")
prompt = "Describe the scene?"
contents = [image, prompt]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")
```

    -------Prompt--------
    


    
![png](output_27_1.png)
    


    Describe the scene?
    
    -------Response--------
     The image shows a pontoon boat on the Charles River in Boston, Massachusetts. The boat is anchored and there are two other boats in the background. The Zakim Bridge and Longfellow Bridge are in the distance.

#### Images with direct links

You can also use direct links to images, as shown below. The helper function `load_image_from_url()` (that was declared earlier) converts the image to bytes and returns it as an Image object that can be then be sent to Gemini 1.0 Pro Vision along with the text prompt.


```
# Load image from Cloud Storage URI
image_url = (
    "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/boats.jpeg"
)
image = load_image_from_url(image_url)  # convert to bytes

# Prepare contents
prompt = "Describe the scene?"
contents = [image, prompt]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")
```

    -------Prompt--------
    


    
![png](output_29_1.png)
    


    Describe the scene?
    
    -------Response--------
     Two pontoon boats are anchored in the Charles River in Boston, Massachusetts. In the background are two bridges and the Boston skyline.

#### Combining multiple images and text prompts for few-shot prompting

You can send more than one image at a time, and also place your images anywhere alongside your text prompt.

In the example below, few-shot prompting is performed to have Gemini 1.0 Pro Vision return the city and landmark in a specific JSON format.


```
# Load images from Cloud Storage URI
image1_url = "https://storage.googleapis.com/github-repo/img/gemini/intro/landmark1.jpg"
image2_url = "https://storage.googleapis.com/github-repo/img/gemini/intro/landmark2.jpg"
image3_url = "https://storage.googleapis.com/github-repo/img/gemini/intro/landmark3.jpg"
image1 = load_image_from_url(image1_url)
image2 = load_image_from_url(image2_url)
image3 = load_image_from_url(image3_url)

# Prepare prompts
prompt1 = """{"city": "London", "Landmark:", "Big Ben"}"""
prompt2 = """{"city": "Paris", "Landmark:", "Eiffel Tower"}"""

# Prepare contents
contents = [image1, prompt1, image2, prompt2, image3]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")
```

    -------Prompt--------
    


    
![png](output_32_1.png)
    


    {"city": "London", "Landmark:", "Big Ben"}
    


    
![png](output_32_3.png)
    


    {"city": "Paris", "Landmark:", "Eiffel Tower"}
    


    
![png](output_32_5.png)
    


    
    -------Response--------
     {"city": "Rome", "Landmark:", "Colosseum"}

### Generate text from a video file

Specify the Cloud Storage URI of the video to include in the prompt. The bucket that stores the file must be in the same Google Cloud project that's sending the request. You must also specify the `mime_type` field. The supported MIME type for video includes `video/mp4`.



```
file_path = "github-repo/img/gemini/multimodality_usecases_overview/pixel8.mp4"
video_uri = f"gs://{file_path}"
video_url = f"https://storage.googleapis.com/{file_path}"

IPython.display.Video(video_url, width=450)
```




<video src="https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/pixel8.mp4" controls  width="450" >
      Your browser does not support the <code>video</code> element.
    </video>




```
prompt = """
Answer the following questions using the video only:
What is the profession of the main person?
What are the main features of the phone highlighted?
Which city was this recorded in?
Provide the answer JSON.
"""

video = Part.from_uri(video_uri, mime_type="video/mp4")
contents = [prompt, video]

responses = multimodal_model.generate_content(contents, stream=True)

for response in responses:
    print(response.text, end="")
```

     ```json
    {
      "profession": "photographer",
      "features": "Night Sight, Video Boost",
      "city": "Tokyo"
    }
    ```
