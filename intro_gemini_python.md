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

# Getting Started with the Gemini API in Vertex AI & Python SDK

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_python.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fgetting-started%2Fintro_gemini_python.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_python.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/getting-started/intro_gemini_python.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Eric Dong](https://github.com/gericdong), [Polong Lin](https://github.com/polong-lin) |

## Overview

### Gemini

Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases. The Gemini API gives you access to the Gemini models.

### Gemini API in Vertex AI

The Gemini API in Vertex AI provides a unified interface for interacting with Gemini models. You can interact with the Gemini API using the following methods:

- Use [Vertex AI Studio](https://cloud.google.com/generative-ai-studio) for quick testing and command generation
- Use cURL commands
- Use the Vertex AI SDK

This notebook focuses on using the **Vertex AI SDK for Python** to call the Gemini API in Vertex AI.

For more information, see the [Generative AI on Vertex AI](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview) documentation.


### Objectives

In this tutorial, you will learn how to use the Gemini API in Vertex AI with the Vertex AI SDK for Python to interact with the Gemini 1.5 Pro (`gemini-1.5-pro`) model.

You will complete the following tasks:

- Install the Vertex AI SDK for Python
- Use the Gemini API in Vertex AI to interact with the Gemini 1.5 models
    - Generate text from text prompt
    - Explore various features and configuration options
    - Generate text from image(s) and text prompt
    - Generate text from video and text prompt


## Getting Started


### Install Vertex AI SDK for Python



```
%pip install --upgrade --user google-cloud-aiplatform
```

### Restart current runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.


```
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
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Image,
    Part,
    SafetySetting,
)
```

## Use the Gemini 1.5 Pro model

The Gemini 1.5 Pro (`gemini-1.5-pro`) model is a foundation model that performs well at a variety of multimodal tasks such as visual understanding, classification, summarization, and creating content from image, audio and video. It's adept at processing visual and text inputs such as photographs, documents, infographics, and screenshots.


### Load the Gemini 1.5 Pro model



```
model = GenerativeModel("gemini-1.5-pro")
```

### Generate text from text prompt

Send a text prompt to the model using the `generate_content` method. The `generate_content` method can handle a wide variety of use cases, including multi-turn chat and multimodal input, depending on what the underlying model supports.



```
response = model.generate_content("Why is the sky blue?")

print(response.text)
```

    The sky appears blue due to a phenomenon called **Rayleigh scattering**. Here's a breakdown:
    
    * **Sunlight Enters the Atmosphere:** Sunlight, appearing white to our eyes, is actually a mix of all colors of the rainbow. When this light enters Earth's atmosphere, it encounters various particles and gases.
    
    * **Shorter Wavelengths Scatter More:**  Blue and violet light have shorter wavelengths compared to other colors in the spectrum. As sunlight interacts with the tiny particles in the atmosphere (mainly nitrogen and oxygen molecules), the shorter wavelengths (blue and violet) are scattered more effectively in all directions.
    
    * **Our Eyes' Perception:** While violet light is scattered even more than blue, our eyes are more sensitive to blue light. This is why we perceive the sky as blue rather than violet.
    
    **In Summary:** The blue color of the sky is a result of the preferential scattering of shorter wavelength blue light by the Earth's atmosphere. 
    
    

### Streaming

By default, the model returns a response after completing the entire generation process. You can also stream the response as it is being generated, and the model will return chunks of the response as soon as they are generated.


```
responses = model.generate_content("Why is the sky blue?", stream=True)

for response in responses:
    print(response.text, end="")
```

    The sky appears blue due to a phenomenon called **Rayleigh scattering**. Here's a breakdown:
    
    **1. Sunlight and its Colors:**
       - Sunlight appears white but is actually a mixture of all the colors of the rainbow.
    
    **2. Earth's Atmosphere:**
       - Our atmosphere is made up of tiny particles, primarily nitrogen and oxygen molecules.
    
    **3. Scattering of Light:**
       - As sunlight enters the atmosphere, it collides with these molecules. 
       - This collision causes the light to scatter in different directions.
    
    **4. Rayleigh Scattering:**
       -  Shorter wavelengths of light, like blue and violet, are scattered more strongly by these molecules than longer wavelengths, like red and orange.
    
    **5. Why We See Blue:**
       - Our eyes are more sensitive to blue light. 
       -  The scattered blue light reaches our eyes from all directions, making the sky appear blue.
    
    **Why not violet?**
       - While violet light is scattered even more than blue, our eyes are less sensitive to violet and the sun emits slightly less violet light compared to blue.
    
    **Sunrise and Sunset Colors:**
       - During sunrise and sunset, the sunlight travels through more of the atmosphere to reach our eyes. 
       - This causes the blue light to be scattered away, allowing us to see the longer wavelengths like red and orange, creating those beautiful colors. 
    

#### Try your own prompts

- What are the biggest challenges facing the healthcare industry?
- What are the latest developments in the automotive industry?
- What are the biggest opportunities in retail industry?
- (Try your own prompts!)



```
prompt = """Create a numbered list of 10 items. Each item in the list should be a trend in the tech industry.

Each trend should be less than 5 words."""  # try your own prompt

response = model.generate_content(prompt)

print(response.text)
```

    Here are 10 tech trends, each in 5 words or less:
    
    1. **AI everywhere** 
    2. **Edge computing grows**
    3. **Sustainable technology** 
    4. **Metaverse expands** 
    5. **Web3 development** 
    6. **Privacy focus increases**
    7. **Low-code/no-code platforms** 
    8. **Hyperautomation accelerates** 
    9. **Everything as a service**
    10. **Quantum computing emerges** 
    
    

#### Model parameters

Every prompt you send to the model includes parameter values that control how the model generates a response. The model can generate different results for different parameter values. You can experiment with different model parameters to see how the results change.



```
generation_config = GenerationConfig(
    temperature=0.9,
    top_p=1.0,
    top_k=32,
    candidate_count=1,
    max_output_tokens=8192,
)

response = model.generate_content(
    "Why is the sky blue?",
    generation_config=generation_config,
)

print(response.text)
```

    The sky appears blue due to a phenomenon called **Rayleigh scattering**. Here's how it works:
    
    1. **Sunlight Enters the Atmosphere:** Sunlight, which appears white to us, is actually a mixture of all colors of the rainbow. 
    
    2. **Scattering of Light:** As sunlight enters the Earth's atmosphere, it collides with tiny air molecules (mostly nitrogen and oxygen). This causes the light to scatter in different directions.
    
    3. **Blue Light Scatters More:**  Blue and violet light have shorter wavelengths compared to other colors in the visible spectrum.  Shorter wavelengths are scattered more strongly by the air molecules. 
    
    4. **Our Eyes' Perception:**  While violet light is scattered even more than blue, our eyes are more sensitive to blue light.  Therefore, we perceive the sky as blue.
    
    **Why not violet then?**
    
    Although violet light is scattered more, our eyes are less sensitive to violet wavelengths and more sensitive to blue. Additionally, sunlight contains a broader range of blue wavelengths compared to violet.
    
    **Why does the sky appear different colors at sunrise and sunset?**
    
    During sunrise and sunset, the sunlight has to travel through a larger portion of the atmosphere to reach our eyes. This longer path results in more scattering of the shorter wavelengths (blue and violet), leaving the longer wavelengths (orange and red) to dominate our perception, creating those beautiful warm hues. 
    
    

### Safety filters

The Gemini API provides safety filters that you can adjust across multiple filter categories to restrict or allow certain types of content. You can use these filters to adjust what's appropriate for your use case. See the [Configure safety filters](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/configure-safety-filters) page for details.

When you make a request to Gemini, the content is analyzed and assigned a safety rating. You can inspect the safety ratings of the generated content by printing out the model responses, as in this example:


```
response = model.generate_content("Why is the sky blue?")

print(f"Safety ratings:\n{response.candidates[0].safety_ratings}")
```

    Safety ratings:
    [category: HARM_CATEGORY_HATE_SPEECH
    probability: NEGLIGIBLE
    probability_score: 0.0693359375
    severity: HARM_SEVERITY_NEGLIGIBLE
    severity_score: 0.046630859375
    , category: HARM_CATEGORY_DANGEROUS_CONTENT
    probability: NEGLIGIBLE
    probability_score: 0.09130859375
    severity: HARM_SEVERITY_NEGLIGIBLE
    severity_score: 0.0693359375
    , category: HARM_CATEGORY_HARASSMENT
    probability: NEGLIGIBLE
    probability_score: 0.11767578125
    severity: HARM_SEVERITY_NEGLIGIBLE
    severity_score: 0.0267333984375
    , category: HARM_CATEGORY_SEXUALLY_EXPLICIT
    probability: NEGLIGIBLE
    probability_score: 0.1435546875
    severity: HARM_SEVERITY_NEGLIGIBLE
    severity_score: 0.0289306640625
    ]
    

In Gemini 1.5 Flash 002 and Gemini 1.5 Pro 002, the safety settings are `OFF` by default and the default block thresholds are `BLOCK_NONE`.

You can use `safety_settings` to adjust the safety settings for each request you make to the API. This example demonstrates how you set the block threshold to BLOCK_ONLY_HIGH for the dangerous content category:


```
safety_settings = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
]

prompt = """
    Write a list of 2 disrespectful things that I might say to the universe after stubbing my toe in the dark.
"""

response = model.generate_content(
    prompt,
    safety_settings=safety_settings,
)

print(response)
```

### Test chat prompts

The Gemini API supports natural multi-turn conversations and is ideal for text tasks that require back-and-forth interactions. The following examples show how the model responds during a multi-turn conversation.



```
chat = model.start_chat()

prompt = """My name is Ned. You are my personal assistant. My favorite movies are Lord of the Rings and Hobbit.

Suggest another movie I might like.
"""

response = chat.send_message(prompt)

print(response.text)
```

    Hi Ned, since you love Lord of the Rings and The Hobbit, both fantasy epics with grand adventures, mythical creatures, and themes of good versus evil, you might enjoy:
    
    * **Willow (1988):**  A classic fantasy film with a similar whimsical feel to LOTR, featuring a dwarf warrior on a quest to protect a baby from an evil queen. 
    * **The Chronicles of Narnia (series):**  Another epic tale about a group of children who discover a magical world.
    
    Let me know what you think!  I can offer more suggestions based on what you liked most about LOTR and The Hobbit.  Did you enjoy the battles, the friendships, the magical creatures, or something else entirely?  😊 
    
    

This follow-up prompt shows how the model responds based on the previous prompt:



```
prompt = "Are my favorite movies based on a book series?"

responses = chat.send_message(prompt)

print(response.text)
```

    Hi Ned, since you love Lord of the Rings and The Hobbit, both fantasy epics with grand adventures, mythical creatures, and themes of good versus evil, you might enjoy:
    
    * **Willow (1988):**  A classic fantasy film with a similar whimsical feel to LOTR, featuring a dwarf warrior on a quest to protect a baby from an evil queen. 
    * **The Chronicles of Narnia (series):**  Another epic tale about a group of children who discover a magical world.
    
    Let me know what you think!  I can offer more suggestions based on what you liked most about LOTR and The Hobbit.  Did you enjoy the battles, the friendships, the magical creatures, or something else entirely?  😊 
    
    

You can also view the chat history:



```
print(chat.history)
```

    [role: "user"
    parts {
      text: "My name is Ned. You are my personal assistant. My favorite movies are Lord of the Rings and Hobbit.\n\nSuggest another movie I might like.\n"
    }
    , role: "model"
    parts {
      text: "Hi Ned, since you love Lord of the Rings and The Hobbit, both fantasy epics with grand adventures, mythical creatures, and themes of good versus evil, you might enjoy:\n\n* **Willow (1988):**  A classic fantasy film with a similar whimsical feel to LOTR, featuring a dwarf warrior on a quest to protect a baby from an evil queen. \n* **The Chronicles of Narnia (series):**  Another epic tale about a group of children who discover a magical world.\n\nLet me know what you think!  I can offer more suggestions based on what you liked most about LOTR and The Hobbit.  Did you enjoy the battles, the friendships, the magical creatures, or something else entirely?  \360\237\230\212 \n"
    }
    , role: "user"
    parts {
      text: "Are my favorite movies based on a book series?"
    }
    , role: "model"
    parts {
      text: "Yes, Ned, both *The Lord of the Rings* and *The Hobbit* are based on book series written by J.R.R. Tolkien! \n\n* **The Hobbit** is a standalone book.\n* **The Lord of the Rings** is a trilogy, consisting of:\n    * *The Fellowship of the Ring*\n    * *The Two Towers*\n    * *The Return of the King*\n\nMany people consider them to be some of the greatest fantasy novels ever written! Have you read any of them? \n"
    }
    ]
    

## Generate text from multimodal prompt

Gemini 1.5 Pro (`gemini-1.5-pro`) is a multimodal model that supports multimodal prompts. You can include text, image(s), and video in your prompt requests and get text or code responses.


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

response = model.generate_content(contents)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
print(response.text)
```

    Copying gs://cloud-samples-data/generative-ai/image/320px-Felis_catus-cat_on_snow.jpg...
    / [1 files][ 17.4 KiB/ 17.4 KiB]                                                
    Operation completed over 1 objects/17.4 KiB.                                     
    -------Prompt--------
    


    
![png](output_42_1.png)
    


    Describe this image?
    
    -------Response--------
    A brown tabby cat is standing on a snow-covered ground and looking directly at the camera.  The cat's fur has black stripes and it has a bushy tail that's curved to the right.  The cat's left paw is slightly raised as if it's about to take a step. The background is blurry, emphasizing the cat as the focal point.  It appears to be a sunny winter day with crisp white snow. 
    
    

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

response = model.generate_content(contents)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
print(response.text, end="")
```

    -------Prompt--------
    


    
![png](output_45_1.png)
    


    Describe the scene?
    
    -------Response--------
    The photo shows two boats on a river with a bridge and city skyline in the background. The nearest boat is a small pontoon boat with a dark green hull and a white canopy. Behind it is a smaller open motorboat with an outboard motor. Both boats appear to be at rest. In the background is a wide river spanned by a multi-arched bridge.  Beyond the bridge is the skyline of a city with numerous buildings of various sizes and designs. The sky is overcast. The scene gives a sense of a peaceful day on the water. 
    

#### Images with direct links

You can also use direct links to images, as shown below. The helper function `load_image_from_url()` (that was declared earlier) converts the image to bytes and returns it as an Image object that can be then be sent to the Gemini model with the text prompt.


```
# Load image from Cloud Storage URI
image_url = (
    "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/boats.jpeg"
)
image = load_image_from_url(image_url)  # convert to bytes

# Prepare contents
prompt = "Describe the scene?"
contents = [image, prompt]

response = model.generate_content(contents)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
print(response.text)
```

    -------Prompt--------
    


    
![png](output_47_1.png)
    


    Describe the scene?
    
    -------Response--------
    The photo shows two boats on a cloudy day. In the background, you can see the Boston city skyline, with several bridges connecting the two sides of the city. The bridge closest to the camera is made of concrete and steel, with several arches. The water is choppy and gray.
    

#### Combining multiple images and text prompts for few-shot prompting

You can send more than one image at a time, and also place your images anywhere alongside your text prompt.

In the example below, few-shot prompting is performed to have the Gemini model return the city and landmark in a specific JSON format.


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

responses = model.generate_content(contents)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
print(response.text)
```

    -------Prompt--------
    


    
![png](output_50_1.png)
    


    {"city": "London", "Landmark:", "Big Ben"}
    


    
![png](output_50_3.png)
    


    {"city": "Paris", "Landmark:", "Eiffel Tower"}
    


    
![png](output_50_5.png)
    


    
    -------Response--------
    The photo shows two boats on a cloudy day. In the background, you can see the Boston city skyline, with several bridges connecting the two sides of the city. The bridge closest to the camera is made of concrete and steel, with several arches. The water is choppy and gray.
    

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
Provide the answer in JSON.
"""

video = Part.from_uri(video_uri, mime_type="video/mp4")
contents = [prompt, video]

response = model.generate_content(contents)

print(response.text)
```

    ```json
    {
     "profession": "Photographer",
     "phone_features": "Video Boost and Night Sight",
     "city": "Tokyo"
    }
    ```
    

### Direct analysis of publicly available web media

This new feature enables you to directly process publicly available URL resources including images, text, video and audio with Gemini. This feature supports all currently [supported modalities and file formats](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#blob).

In this example, you add the file URL of a publicly available image file to the request to identify what's in the image.


```
prompt = """
Extract the objects in the given image and output them in a list in alphabetical order.
"""

image_file = Part.from_uri(
    "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/office-desk.jpeg",
    "image/jpeg",
)

response = model.generate_content([image_file, prompt])

print(response.text)
```

This example demonstrates how to add the file URL of a publicly available video file to the request, and use the [controlled generation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output) capability to constraint the model output to a structured format.


```
response_schema = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "timecode": {
                "type": "STRING",
            },
            "chapter_summary": {
                "type": "STRING",
            },
        },
        "required": ["timecode", "chapter_summary"],
    },
}

prompt = """
Chapterize this video content by grouping the video content into chapters and providing a brief summary for each chapter. 
Please only capture key events and highlights. If you are not sure about any info, please do not make it up. 
"""

video_file = Part.from_uri(
    "https://storage.googleapis.com/cloud-samples-data/generative-ai/video/rio_de_janeiro_beyond_the_map_rio.mp4",
    "video/mp4",
)

response = model.generate_content(
    contents=[video_file, prompt],
    generation_config=GenerationConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
    ),
)

print(response.text)
```
