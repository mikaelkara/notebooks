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

# Imagen 3 Image Generation

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/vision/getting-started/imagen3_image_generation.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fvision%2Fgetting-started%2Fimagen3_image_generation.ipynb">
      <img width="32px" src="https://cloud.google.com/ml-engine/images/colab-enterprise-logo-32px.png" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/vision/getting-started/imagen3_image_generation.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/vision/getting-started/imagen3_image_generation.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Katie Nguyen](https://github.com/katiemn) |

## Overview

### Imagen 3

Imagen 3 on Vertex AI brings Google's state of the art generative AI capabilities to application developers. Imagen 3 is Google's highest quality text-to-image model to date. It's capable of creating images with astonishing detail. Thus, developers have more control when building next-generation AI products that transform their imagination into high quality visual assets. Learn more about [Imagen on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/image/overview).


In this tutorial, you will learn how to use the Vertex AI SDK for Python to interact with the Imagen 3 and Imagen 3 Fast models to generate images showcasing:

- Photorealistic scenes
- Text rendered within images
- Quality and latency comparisons within the two models

## Get started


### Install Vertex AI SDK for Python



```
%pip install --upgrade --user google-cloud-aiplatform
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
from vertexai.preview.vision_models import ImageGenerationModel
```

### Define a helper function


```
import typing

import IPython.display
from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps


def display_image(
    image,
    max_width: int = 600,
    max_height: int = 350,
) -> None:
    pil_image = typing.cast(PIL_Image.Image, image._pil_image)
    if pil_image.mode != "RGB":
        # RGB is supported by all Jupyter environments (e.g. RGBA is not yet)
        pil_image = pil_image.convert("RGB")
    image_width, image_height = pil_image.size
    if max_width < image_width or max_height < image_height:
        # Resize to display a smaller notebook image
        pil_image = PIL_ImageOps.contain(pil_image, (max_width, max_height))
    IPython.display.display(pil_image)
```

### Load the image generation models

Imagen 3: `imagen-3.0-generate-001`

Imagen 3 Fast: `imagen-3.0-fast-generate-001`


```
generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
generation_model_fast = ImageGenerationModel.from_pretrained(
    "imagen-3.0-fast-generate-001"
)
```

### Imagen 3 & Imagen 3 Fast

With Imagen 3, you also have the option to use Imagen 3 Fast. These two model options give you the choice to optimize for quality and latency, depending on your use case.

**Imagen 3:** Generates high quality images with natural lighting and increased photorealism.

**Imagen 3 Fast:** Suitable for creating brighter images with a higher contrast. Overall, you can see a 40% decrease in latency in Imagen 3 Fast compared to Imagen 2.

With Imagen 3 and Imagen 3 Fast, you can also configure the `aspect ratio` to any of the following:
* 1:1
* 9:16
* 16:9
* 3:4
* 4:3


```
import matplotlib.pyplot as plt

prompt = """
a photorealistic image of the inside of an amethyst crystal on display in a museum
"""

# Imagen 3 image generation
image = generation_model.generate_images(
    prompt=prompt,
    number_of_images=1,
    aspect_ratio="3:4",
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

# Imagen 3 Fast image generation
fast_image = generation_model_fast.generate_images(
    prompt=prompt,
    number_of_images=1,
    aspect_ratio="3:4",
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

# Display generated images
fig, axis = plt.subplots(1, 2, figsize=(12, 6))
axis[0].imshow(image[0]._pil_image)
axis[0].set_title("Imagen 3")
axis[1].imshow(fast_image[0]._pil_image)
axis[1].set_title("Imagen 3 Fast")
for ax in axis:
    ax.axis("off")
plt.show()
```

### Photorealism and prompt understanding

**Photorealism:** Imagen 3 is capable of generating photorealistic, lifelike images with fewer distracting visual artifacts than our previous models. This increased quality is especially prevalent when generating images of multiple people, animals, and landscapes.

**Prompt adherence:** It's also better at understanding natural language and the intent behind your prompts. Thus, they can be written in everyday language and can include specific details including camera angles, lens types, lighting, and stylistic features.

When generating images of people you can also set the `safety_filter_level` and `person_generation` parameters accordingly:
* `person_generation`: Allow (All ages), Allow (Adults only), Don't allow
* `safety_filter_level`: Block most, Block some, Block few


```
prompt = """
a photorealistic image of a happy family sitting around a table eating dinner, a yellow dog is at their feet begging for food,
the table is brown with white comfy chairs,
a grand chandelier above their heads casting a warm glow on the family,
they are eating pizza and salad,
large windows behind them with green curtains pulled off to either side
"""

images = generation_model.generate_images(
    prompt=prompt,
    number_of_images=1,
    aspect_ratio="1:1",
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

display_image(images[0])
```

### Better text rendering

Imagen 3 also does a great job accurately rendering small words and phrases in images. This could be particularly useful for generating business cards, posters, banners, product designs, or greeting cards.


```
prompt = """
a beige baseball cap with 'good vibes' written on top in white bubbly stitched letters that are outlined in neon green,
display it against a pool background with palm trees and pool floats
"""

images = generation_model.generate_images(
    prompt=prompt,
    number_of_images=1,
    aspect_ratio="1:1",
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

display_image(images[0])
```

### Add image watermark

By default, a digital watermark, or [SynthID](https://deepmind.google/technologies/synthid/), is added to Imagen 3 images. If you would like to explicitly set the watermark to True, you can do so with the `add_watermark` parameter. You can also [verify a watermarked image](https://cloud.google.com/vertex-ai/generative-ai/docs/image/generate-images#watermark).


```
prompt = """
Generate a photo of an old fashioned ice cream shop with white and pastel pink on the outside,
located on the street in a sunny beach town,
'the sweet stuff' displayed on the storefront with illustrations of pink and blue ice cream cones
"""

images = generation_model.generate_images(
    prompt=prompt,
    number_of_images=1,
    aspect_ratio="1:1",
    safety_filter_level="block_some",
    person_generation="allow_adult",
    add_watermark=True,
)

display_image(images[0])
```
