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

# Create a Photoshop Document with Image Segmentation on Vertex AI

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/vision/use-cases/image_segmentation_layers.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fvision%2Fuse-cases%2Fimage_segmentation_layers.ipynb">
      <img width="32px" src="https://cloud.google.com/ml-engine/images/colab-enterprise-logo-32px.png" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/vision/use-cases/image_segmentation_layers.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/vision/use-cases/image_segmentation_layers.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Katie Nguyen](https://github.com/katiemn) |

## Overview

### Image Segmentation

Image Segmentation on Vertex AI brings Google's state of the art segmentation models to developers as a scalable and reliable service.

With the Vertex AI Image Segmentation API, developers can choose from five different modes to segment images and build AI products.


In this tutorial, you will learn how to use the Vertex AI API to interact with the Image Segmentation model to:

- Segment images using different modes to create image masks
- Turn those image masks to individual layers in a PSD file
- Save the PSD file to a Cloud Storage bucket

## Get Started

### Install Vertex AI SDK for Python and Wand


```
!sudo apt-get install libmagickwand-dev

%pip install --upgrade --user --quiet google-cloud-aiplatform Wand
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

### Set Google Cloud project information and initialize Vertex AI API

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
from google.cloud import aiplatform

PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

aiplatform.init(project=PROJECT_ID, location=LOCATION)

api_regional_endpoint = f"{LOCATION}-aiplatform.googleapis.com"
client_options = {"api_endpoint": api_regional_endpoint}
client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

model_endpoint = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/image-segmentation-001"
```

### Import libraries


```
import base64
import io
import math
import os
from random import randrange
import re
import typing

import IPython
from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps
from google.cloud import storage
import ipywidgets as widgets
import matplotlib.pyplot as plt
from vertexai.preview.vision_models import Image as VertexAI_Image
from wand.image import Image as Wand_Image
```

### Define helper functions


```
# Parses the mask bytes from the response and converts it to an Image PIL object


def prediction_to_mask_pil(prediction) -> PIL_Image:
    encoded_mask_string = prediction["bytesBase64Encoded"]
    mask_bytes = base64.b64decode(encoded_mask_string)
    mask_pil = PIL_Image.open(io.BytesIO(mask_bytes))
    mask_pil.thumbnail((4096, 4096))
    return mask_pil


# Displays a PIL image horizontally next to a generated mask from the response
def display_horizontally(input_images: list, figsize: tuple[int, int] = (20, 5)):
    rows: int = math.ceil(len(input_images) / 4)  # Display at most 4 images per row
    cols: int = min(
        len(input_images) + 1, 4
    )  # Adjust columns based on the number of images
    fig, axis = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)

    for i, ax in enumerate(axis.flat):
        if i < len(input_images):
            cmap = "gray" if i > 0 else None
            ax.imshow(input_images[i], cmap)
            # Adjust the axis aspect ratio to maintain image proportions
            ax.set_aspect("equal")
            # Disable axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
        else:
            # Hide empty subplots
            ax.axis("off")

    # Adjust the layout to minimize whitespace between subplots.
    plt.tight_layout()
    plt.show()


def display_image(
    image: VertexAI_Image,
    max_width: int = 4096,
    max_height: int = 4096,
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


# Constructs a Vertex AI PredictRequest for the Image Segmentation model
def call_vertex_image_segmentation(
    gcs_uri=None,
    mode="foreground",
    prompt=None,
):
    instances = []
    if gcs_uri:
        instances.append(
            {
                "image": {"gcsUri": gcs_uri},
            }
        )
    if prompt:
        instances[0]["prompt"] = prompt

    parameters = {"mode": mode}
    response = client.predict(
        endpoint=model_endpoint, instances=instances, parameters=parameters
    )

    return response
```

### Select an image to segment from a Google Cloud Storage URI


```
file_path = "gs://"  # @param {type:"string"}

# Load the image file as Image object
image_file = VertexAI_Image.load_from_file(file_path)
display_image(image_file)

image_file.save("original.png")
```

### Segment images using different modes

You can generate image masks with different Image Segmentation features by setting the `mode` field to one of the available options:
* **Foreground**: Generate a mask of the segmented foreground of the image.
* **Background**: Generate a mask of the segmented background of the image.
* **Semantic**: Select the items in an image to segment from a set of 194 classes.
* **Prompt**: Use an open-vocabulary text prompt to guide the image segmentation.


### Foreground segmentation request


```
gcs_uri = file_path
mode = "foreground"
prompt = None  # Prompt to guide segmentation for `semantic` and `prompt` modes

response = call_vertex_image_segmentation(gcs_uri, mode, prompt)

MASK_PIL = prediction_to_mask_pil(response.predictions[0])
MASK_PIL.save("foreground.png")
BACKGROUND_PIL = PIL_Image.open("original.png")
display_horizontally([BACKGROUND_PIL, MASK_PIL])
```

### Background segmentation request


```
gcs_uri = file_path
mode = "background"
prompt = None  # Prompt to guide segmentation for `semantic` and `prompt` modes

response = call_vertex_image_segmentation(gcs_uri, mode, prompt)

MASK_PIL = prediction_to_mask_pil(response.predictions[0])
MASK_PIL.save("background.png")
BACKGROUND_PIL = PIL_Image.open("original.png")
display_horizontally([BACKGROUND_PIL, MASK_PIL])
```

### Semantic segmentation request

Specify the objects to segment from the set of 194 classes. For your convenience, the classes have been arranged into seven separate categories. Run the cell below and select your classes. To select multiple options from the same category, press ctrl or command and click on your selections.


```
from IPython.display import display

home_and_furniture = widgets.SelectMultiple(
    options=[
        "oven",
        "toaster",
        "ottoman",
        "sink",
        "wardrobe",
        "refrigerator",
        "chest_of_drawers",
        "dishwasher",
        "bookshelf",
        "armchair",
        "toilet",
        "counter_other",
        "bathtub",
        "bathroom_counter",
        "shower",
        "kitchen_island",
        "hair_dryer",
        "door",
        "couch",
        "toothbrush",
        "light_other",
        "lamp",
        "sconce",
        "nightstand",
        "microwave",
        "bed",
        "ceiling",
        "mirror",
        "cup",
        "shelf",
        "knife",
        "stairs",
        "fork",
        "spoon",
        "curtain_other",
        "cabinet",
        "bowl",
        "television",
        "fireplace",
        "tray",
        "floor",
        "stove",
        "range_hood",
        "towel",
        "plate",
        "rug_floormat",
        "wall",
        "window",
        "washer_dryer",
    ],
    value=[],
    rows=5,
    description="Home",
    disabled=False,
)

food = widgets.SelectMultiple(
    options=[
        "broccoli",
        "carrot",
        "hot_dog",
        "pizza",
        "donut",
        "cake",
        "fruit_other",
        "food_other",
        "bottle",
        "wine_glass",
        "banana",
        "apple",
        "sandwich",
        "orange",
    ],
    value=[],
    rows=5,
    description="Food",
    disabled=False,
)

outdoor_and_recreation = widgets.SelectMultiple(
    options=[
        "road",
        "mountain_hill",
        "snow",
        "rock",
        "sidewalk_pavement",
        "frisbee",
        "runway",
        "skis",
        "terrain",
        "snowboard",
        "sports_ball",
        "baseball_bat",
        "baseball_glove",
        "skateboard",
        "surfboard",
        "tennis_racket",
        "net",
        "tunnel",
        "bridge",
        "tent",
        "awning",
        "river_lake",
        "sea",
        "bus",
        "bench",
        "train",
        "bike_rack",
        "vegetation",
        "truck",
        "waterfall",
        "bicycle",
        "trailer",
        "sky",
        "car",
        "traffic_sign",
        "boat_ship",
        "autorickshaw",
        "traffic_light",
        "motorcycle",
        "airplane",
    ],
    value=[],
    rows=5,
    description="Outdoor",
    disabled=False,
)

office_and_work = widgets.SelectMultiple(
    options=[
        "storage_tank",
        "desk",
        "conveyor_belt",
        "suitcase",
        "chair_other",
        "swivel_chair",
        "laptop",
        "whiteboard",
        "keyboard",
        "mouse",
    ],
    value=[],
    rows=5,
    description="Office",
    disabled=False,
)
clothing_and_accessories = widgets.SelectMultiple(
    options=["backpack", "bag", "tie", "apparel"],
    value=[],
    rows=5,
    description="Clothing",
    disabled=False,
)

animals = widgets.SelectMultiple(
    options=[
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "animal_other",
    ],
    value=[],
    rows=5,
    description="Animals",
    disabled=False,
)

miscellaneous = widgets.SelectMultiple(
    options=[
        "pool_table",
        "umbrella",
        "barrel",
        "case",
        "book",
        "crib",
        "box",
        "kite",
        "basket",
        "clock",
        "fan",
        "vase",
        "scissors",
        "plaything_other",
        "stool",
        "teddy_bear",
        "seat",
        "base",
        "trash_can",
        "painting",
        "sculpture",
        "pier_wharf",
        "potted_plant",
        "poster",
        "column",
        "bulletin_board",
        "fountain",
        "building",
        "chandelier",
        "radiator",
        "table",
        "stage",
        "arcade_machine",
        "banner",
        "gravel",
        "flag",
        "platform",
        "blanket",
        "remote",
        "escalator",
        "playingfield",
        "cell phone",
        "railroad",
        "shower_curtain",
        "fire_hydrant",
        "pillow",
        "parking_meter",
        "road_barrier",
        "water_other",
        "mailbox",
        "swimming_pool",
        "person",
        "cctv_camera",
        "billboard",
        "rider_other",
        "junction_box",
        "bicyclist",
        "pole",
        "motorcyclist",
        "slow_wheeled_object",
        "fence",
        "window_blind",
        "paper",
        "streetlight",
        "railing_banister",
        "guard_rail",
    ],
    value=[],
    rows=5,
    description="Miscellaneous",
    disabled=False,
)

display(home_and_furniture)
display(food)
display(outdoor_and_recreation)
display(office_and_work)
display(clothing_and_accessories)
display(animals)
display(miscellaneous)
```

Combine all your segmentation class selections into a single string for the request.


```
item_string = ",".join(
    home_and_furniture.value
    + food.value
    + outdoor_and_recreation.value
    + office_and_work.value
    + clothing_and_accessories.value
    + animals.value
    + miscellaneous.value
)
print(item_string)
```

Regardless of the number of classes, a semantic segmentation request will return a single image mask with all detected items from the request.


```
gcs_uri = file_path
mode = "semantic"
prompt = item_string

response = call_vertex_image_segmentation(gcs_uri, mode, prompt)

MASK_PIL = prediction_to_mask_pil(response.predictions[0])
MASK_PIL.save("semantic.png")
BACKGROUND_PIL = PIL_Image.open("original.png")
display_horizontally([BACKGROUND_PIL, MASK_PIL])
```

### Open vocabulary segmentation request

Provide a prompt to guide the image segmentation. Unlike other modes, an open vocabulary request will produce multiple image masks based on the prompt.


```
# Delete local prompt based masks from previous runs
pattern = re.compile("prompt*")
for file in os.listdir("."):
    if pattern.match(file):
        os.remove(file)

gcs_uri = file_path
mode = "prompt"
prompt = "[your-prompt]"  # @param {type:"string"}

response = call_vertex_image_segmentation(gcs_uri, mode, prompt)

BACKGROUND_PIL = PIL_Image.open("original.png")
images = [BACKGROUND_PIL]
for i in range(len(response.predictions)):
    MASK_PIL = prediction_to_mask_pil(response.predictions[i])
    MASK_PIL.save("prompt" + str(i) + ".png")
    images.append(MASK_PIL)

display_horizontally(images)
```

### Select masks to apply to PSD file

Run the following cell to generate a checklist of all possible segmentation modes you may have previously generated. Then, select all modes you would like to be included in the final PSD file. All of the specified image masks will be included as separate layers.


```
from IPython.display import display

foreground_checkbox = widgets.Checkbox(
    value=True, description="Foreground Mask", disabled=False
)
background_checkbox = widgets.Checkbox(
    value=True, description="Background Mask", disabled=False
)
semantic_checkbox = widgets.Checkbox(
    value=True, description="Semantic Mask", disabled=False
)
prompt_checkbox = widgets.Checkbox(
    value=True, description="Prompt Mask", disabled=False
)


display(foreground_checkbox)
display(background_checkbox)
display(semantic_checkbox)
display(prompt_checkbox)
```

### Add selected mask images as layers

Once the layers are added, you will save the final PSD file.


```
with Wand_Image(filename="original.png") as img:
    img.read(filename="original.png")

    if foreground_checkbox.value:
        img.read(filename="foreground.png")
    if background_checkbox.value:
        img.read(filename="background.png")
    if semantic_checkbox.value:
        img.read(filename="semantic.png")
    if prompt_checkbox.value:
        pattern = re.compile("prompt*")
        for file in os.listdir("."):
            if pattern.match(file):
                img.read(filename=file)

    img.save(filename="output.psd")
```

### Upload the PSD file to Google Cloud Storage bucket


```
prefix = "psd_" + str(randrange(10000, 100000))
bucket_name = "[your-bucket-name]"  # @param {type:"string"}

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(prefix)

blob.upload_from_filename("output.psd")
print("Uploaded " + prefix)
```
