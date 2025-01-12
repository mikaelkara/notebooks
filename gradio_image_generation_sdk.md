```
# Copyright 2023 Google LLC
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

# Using a Gradio app and Vertex AI for image generation

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/vision/gradio/gradio_image_generation_sdk.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/vision/gradio/gradio_image_generation_sdk.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/vision/gradio/gradio_image_generation_sdk.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Jose Brache](https://github.com/jbrache) |

## Overview

This notebook will create a [Gradio app](https://www.gradio.app/) (frontend) that integrates with [Imagen on Vertex AI](https://cloud.google.com/vertex-ai/docs/generative-ai/image/overview) to generate high quality images using natural language prompts.

This notebook only focuses on **image generation** features from Imagen. Note that [image generation](https://cloud.google.com/vertex-ai/docs/generative-ai/image/overview#feature-launch-stage) is currently under **Restricted General Availability (approved users)**. In order to use the API you will need to request access in the [Google Cloud console](https://console.cloud.google.com/vertex-ai/generative/vision) via the request form in the `Generate` tab under **Vertex AI Studio &rarr; Vision**.

For more information about writing text prompts for image generation, see the [prompt guide](https://cloud.google.com/vertex-ai/docs/generative-ai/image/img-gen-prompt-guide) and these resources:
- When you generate images there are several standard and advanced [parameters](https://cloud.google.com/vertex-ai/docs/generative-ai/image/generate-images#use-params) you can set depending on your use case.
- There are various versions of the `imagegeneration` model you can use. For general information on Imagen model versioning, see the [official documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/image/generate-images#model-versions).

Imagen can be accessed via the [Google Cloud console](https://console.cloud.google.com/vertex-ai/generative/vision) or by calling the Vertex AI API. More information about image generation with Imagen on Vertex AI can be found in the [official documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/image/generate-images).

### Objectives

In this notebook, you will learn how to:

- Generate new images from a text prompt using [Imagen 2](https://cloud.google.com/blog/products/ai-machine-learning/imagen-2-on-vertex-ai-is-now-generally-available) (`imagegeneration@005`) with the Vertex AI SDK

- Experiment with different parameters, such as:
    - Using example or your own text prompts to generate images
    - Version of the model used to generate images
    - Providing a seed to reproduce the same image output from inputs

- Launch a [Gradio app](https://www.gradio.app/) to access Imagen


### Costs

- This notebook uses billable components of Google Cloud:
  - Vertex AI (Imagen)

- Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.

## Getting Started

### Install Vertex AI SDK, other packages and their dependencies

[Gradio](https://pypi.org/project/gradio/) is used to interactively use Imagen with a user interface, tested with versions `gradio==4.11.0` and `google-cloud-aiplatform==1.38.1`.


```
%pip install --upgrade --user gradio
%pip install --upgrade --user google-cloud-aiplatform
```

### Restart current runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.

The restart process might take a minute or so.


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

If you are running this notebook on Google Colab, you will need to authenticate your environment. To do this, run the new cell below. This step is not required if you are using [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench).


```
import sys

# Additional authentication is required for Google Colab
if "google.colab" in sys.modules:
    # Authenticate user to Google Cloud
    from google.colab import auth

    auth.authenticate_user()
```

### Define Google Cloud project information (Colab only)

If you are running this notebook on Google Colab, you need to define Google Cloud project information to be used. In the following cell, you will define the information, import Vertex AI package, and initialize it. This step is also not required if you are using [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench).


```
if "google.colab" in sys.modules:
    # Define project information
    PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
    LOCATION = "us-central1"  # @param {type:"string"}

    # Initialize Vertex AI
    import vertexai

    vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries


```
import traceback

import gradio as gr
from vertexai.preview.vision_models import ImageGenerationModel
```

# Gradio app

[Imagen 2](https://cloud.google.com/blog/products/ai-machine-learning/imagen-2-on-vertex-ai-is-now-generally-available) (`imagegeneration@005`) is designed for generating high-quality, photorealistic, high-resolution, aesthetically pleasing images from natural language prompts.

This section packages up the text to image generation capabilities from Imagen into a [Gradio app](https://www.gradio.app/docs/interface) for interactive use with example prompts. Imagen has [model versions](https://cloud.google.com/vertex-ai/docs/generative-ai/image/generate-images) supporting different features, see [Imagen on Vertex AI model versions and lifecycle](https://cloud.google.com/vertex-ai/docs/generative-ai/image/model-versioning) for more information.

### Define helper functions

Define helper functions for Gradio and the Vertex AI SDK to load and display images.


```
# @title Helper functions
# Wrapper around the Vertex AI SDK to return PIL images


def imagen_generate(
    model_name: str,
    prompt: str,
    negative_prompt: str,
    sampleImageSize: int,
    sampleCount: int,
    seed=None,
):
    model = ImageGenerationModel.from_pretrained(model_name)

    generate_response = model.generate_images(
        prompt=prompt,
        negative_prompt=negative_prompt,
        number_of_images=sampleCount,
        guidance_scale=float(sampleImageSize),
        seed=seed,
    )

    images = []
    for index, result in enumerate(generate_response):
        images.append(generate_response[index]._pil_image)
    return images, generate_response


# Update function called by Gradio
def update(
    model_name,
    prompt,
    negative_prompt,
    sampleImageSize="1536",
    sampleCount=4,
    seed=None,
):
    if len(negative_prompt) == 0:
        negative_prompt = None

    print("prompt:", prompt)
    print("negative_prompt:", negative_prompt)

    # Advanced option, try different the seed numbers
    # any random integer number range: (0, 2147483647)
    if seed < 0 or seed > 2147483647:
        seed = None

    # Use & provide a seed, if possible, so that we can reproduce the results when needed.
    images = []
    error_message = ""
    try:
        images, generate_response = imagen_generate(
            model_name, prompt, negative_prompt, sampleImageSize, sampleCount, seed
        )
    except Exception as e:
        print(e)
        error_message = """An error occurred calling the API.
      1. Check if response was not blocked based on policy violation, check if the UI behaves the same way.
      2. Try a different prompt to see if that was the problem.
      """
        error_message += "\n" + traceback.format_exc()
        # raise gr.Error(str(e))

    return images, error_message
```

### Define Gradio examples

Example text prompts are provided to generate images, you can also try your own text prompts as well.


```
examples = [
    [
        "imagegeneration@005",
        """A studio portrait of a man with a grizzly beard eating a sandwich with his hands, a dramatic skewed angled photography, film noir.""",
        "",
        "1536",
        4,
        -1,
    ],
    [
        "imagegeneration@005",
        """A mosaic-inspired portrait of a person, their features formed by a collection of small, colourful tiles.""",
        "",
        "1536",
        4,
        -1,
    ],
    [
        "imagegeneration@005",
        """A modern house on a coastal cliff, sunset, reflections in the water, bright stylized, architectural magazine photo.""",
        "",
        "1536",
        4,
        -1,
    ],
    [
        "imagegeneration@005",
        """Isometric 3d rendering of a car driving in the countryside surrounded by trees, bright colors, puffy clouds overhead.""",
        "",
        "1536",
        4,
        -1,
    ],
    [
        "imagegeneration@005",
        """A tube of toothpaste with the words "CYMBAL" written on it, on a bathroom counter, advertisement.""",
        "",
        "1536",
        4,
        -1,
    ],
    [
        "imagegeneration@005",
        """A cup of strawberry yogurt with the word "Delicious" written on its side, sitting on a wooden tabletop. Next to the cup of yogurt is a plat with toast and a glass of orange juice.""",
        "",
        "1536",
        4,
        -1,
    ],
    [
        "imagegeneration@005",
        """A clean minimal emblem style logo for an ice cream shop, cream background.""",
        "",
        "1536",
        4,
        -1,
    ],
    [
        "imagegeneration@005",
        """An abstract logo representing intelligence for an enterprise AI platform, "Vertex AI" written under the logo.""",
        "",
        "1536",
        4,
        -1,
    ],
    [
        "imagegeneration@002",
        """A line drawing of a duck boat tour in Boston, with a colorful background of the city.""",
        "",
        "1024",
        4,
        -1,
    ],
    [
        "imagegeneration@002",
        """A raccoon wearing formal clothes, wearing a top hat. Oil painting in the style of Vincent Van Gogh.""",
        "",
        "1024",
        4,
        -1,
    ],
]
```

## Gradio Interface

This section launches a [Gradio Interface](https://www.gradio.app/docs/interface) which can be opened via a public URL or used directly from the notebook. Feel free to experiment with different text prompts to generate images.


```
# https://gradio.app/docs/#gallery
iface = gr.Interface(
    fn=update,
    inputs=[
        gr.Dropdown(
            label="Model Name",
            choices=["imagegeneration@002", "imagegeneration@005"],
            value="imagegeneration@005",
        ),
        gr.Textbox(
            placeholder="Try: A studio portrait of a man with a grizzly beard eating a sandwich with his hands, a dramatic skewed angled photography, film noir.",
            label="Text Prompt",
            value="A studio portrait of a man with a grizzly beard eating a sandwich with his hands, a dramatic skewed angled photography, film noir.",
        ),
        gr.Textbox(placeholder="", label="Negative Prompt", value=""),
        gr.Dropdown(label="ImageSize", choices=["256", "1024", "1536"], value="1536"),
        gr.Number(label="sampleCount", value=4),
        gr.Number(
            label="seed",
            info="Use & provide a seed, if possible, so that we can reproduce the results when needed. Integer number range: (0, 2147483647)",
            value=-1,
        ),
    ],
    outputs=[
        gr.Gallery(
            label="Generated Images",
            show_label=True,
            elem_id="gallery",
            columns=[2],
            object_fit="contain",
            height="auto",
        ),
        gr.Textbox(label="Error Messages"),
    ],
    examples=examples,
    title="Imagen",
    description="""Image generation from a text prompt. Look at [this link](https://cloud.google.com/vertex-ai/docs/generative-ai/image/generate-images) for Imagen documentation.
                     """,
    allow_flagging="never",
    theme=gr.themes.Soft(),
)
```

### Launch the Gradio app and start generating images!


```
# Set debug=True in Colab for live debugging
iface.launch(debug=True)
```


```
# (Optional) Make your Gradio app link publicly accessible by uncommenting the line below and running this cell
# iface.launch(share=True, debug=True)
```
