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

# Gemini: An Overview of Multimodal Use Cases

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/intro_multimodal_use_cases.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fuse-cases%2Fintro_multimodal_use_cases.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/use-cases/intro_multimodal_use_cases.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/intro_multimodal_use_cases.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Saeed Aghabozorgi](https://github.com/saeedaghabozorgi) |

## Overview

In this notebook, you will explore a variety of different use cases enabled by multimodality with Gemini 1.5 Flash.

### Gemini

Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases. The Gemini API gives you access to the Gemini 1.0 Pro Vision, Gemini 1.0 Pro, Gemini 1.5 Pro and Gemini 1.5 Flash models.

### Gemini API in Vertex AI

The Gemini API in Vertex AI provides a unified interface for interacting with Gemini models. There are currently four models available in the Gemini API:

- **Gemini 1.0 Pro model** (`gemini-1.0-pro`): Designed to handle natural language tasks, multiturn text and code chat, and code generation.
- **Gemini 1.0 Pro Vision model** (`gemini-1.0-pro-vision`): Supports multimodal prompts. You can include text, images, and video in your prompt requests and get text or code responses.
- **Gemini 1.5 Pro model** (`gemini-1.5-pro`): A foundation model that performs well at a variety of multimodal tasks such as visual understanding, classification, summarization, and creating content from image, audio and video..
- **Gemini 1.5 Flash model** (`gemini-1.5-flash`): A purpose-built multimodal model that provides speed and efficiency for high-volume, quality, cost-effective apps.

For more information, see the [Generative AI on Vertex AI](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview) documentation.


### Objectives

This notebook demonstrates a variety of multimodal use cases that Gemini can be used for.

#### Multimodal use cases

Compared to text-only LLMs, Gemini 1.5's multimodality can be used for many new use-cases:

Example use cases with **text and image(s)** as input:

- Detecting objects in photos
- Understanding screens and interfaces
- Understanding of drawing and abstraction
- Understanding charts and diagrams
- Recommendation of images based on user preferences
- Comparing images for similarities, anomalies, or differences

Example use cases with **text and video** as input:

- Generating a video description
- Extracting tags of objects throughout a video
- Extracting highlights/messaging of a video


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




    {'status': 'ok', 'restart': True}



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

### Define Google Cloud project information and initialize Vertex AI

Initialize the Vertex AI SDK for Python for your project:


```
# Define project information
PROJECT_ID = ""  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

# Initialize Vertex AI
import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries



```
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image, Part
```

## Use the Gemini 1.5 Flash model

Gemini 1.5 Flash (`gemini-1.5-flash`) is a multimodal model that supports multimodal prompts. You can include text, image(s), and video in your prompt requests and get text or code responses.


### Load Gemini 1.5 Flash model



```
multimodal_model = GenerativeModel("gemini-1.5-flash")
```

### Define helper functions



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


def display_content_as_image(content: str | Image | Part) -> bool:
    if not isinstance(content, Image):
        return False
    display_images([content])
    return True


def display_content_as_video(content: str | Image | Part) -> bool:
    if not isinstance(content, Part):
        return False
    part = typing.cast(Part, content)
    file_path = part.file_data.file_uri.removeprefix("gs://")
    video_url = f"https://storage.googleapis.com/{file_path}"
    IPython.display.display(IPython.display.Video(video_url, width=600))
    return True


def print_multimodal_prompt(contents: list[str | Image | Part]):
    """
    Given contents that would be sent to Gemini,
    output the full multimodal prompt for ease of readability.
    """
    for content in contents:
        if display_content_as_image(content):
            continue
        if display_content_as_video(content):
            continue
        print(content)
```

## Image understanding across multiple images

One of the capabilities of Gemini is being able to reason across multiple images.

This is an example of using Gemini to calculate the total cost of groceries using an image of fruits and a price list:



```
image_grocery_url = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/banana-apple.jpg"
image_prices_url = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/pricelist.jpg"
image_grocery = load_image_from_url(image_grocery_url)
image_prices = load_image_from_url(image_prices_url)

instructions = "Instructions: Consider the following image that contains fruits:"
prompt1 = "How much should I pay for the fruits given the following price list?"
prompt2 = """
Answer the question through these steps:
Step 1: Identify what kind of fruits there are in the first image.
Step 2: Count the quantity of each fruit.
Step 3: For each grocery in first image, check the price of the grocery in the price list.
Step 4: Calculate the subtotal price for each type of fruit.
Step 5: Calculate the total price of fruits using the subtotals.

Answer and describe the steps taken:
"""

contents = [
    instructions,
    image_grocery,
    prompt1,
    image_prices,
    prompt2,
]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")
```

    -------Prompt--------
    Instructions: Consider the following image that contains fruits:
    


    
![png](output_24_1.png)
    


    How much should I pay for the fruits given the following price list?
    


    
![png](output_24_3.png)
    


    
    Answer the question through these steps:
    Step 1: Identify what kind of fruits there are in the first image.
    Step 2: Count the quantity of each fruit.
    Step 3: For each grocery in first image, check the price of the grocery in the price list.
    Step 4: Calculate the subtotal price for each type of fruit.
    Step 5: Calculate the total price of fruits using the subtotals.
    
    Answer and describe the steps taken:
    
    
    -------Response--------
    Step 1: The image contains apples and bananas.
    Step 2: There are 3 apples and 2 bananas.
    Step 3: The price of each apple is $1.50 and the price of each banana is $0.80.
    Step 4: The subtotal price of apples is 3 * $1.50 = $4.50. The subtotal price of bananas is 2 * $0.80 = $1.60.
    Step 5: The total price of fruits is $4.50 + $1.60 = $6.10.
    
    Therefore, you should pay $6.10 for the fruits.

## Understanding Screens and Interfaces

Gemini can also extract information from appliance screens, UIs, screenshots, icons, and layouts.

For example, if you input an image of a stove, you can ask Gemini to provide instructions to help a user navigate the UI and respond in different languages:



```
image_stove_url = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/stove.jpg"
image_stove = load_image_from_url(image_stove_url)

prompt = """Help me to reset the clock on this appliance?
Provide the instructions in English and French.
If instructions include buttons, also explain where those buttons are physically located.
"""

contents = [image_stove, prompt]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")
```

    -------Prompt--------
    


    
![png](output_26_1.png)
    


    Help me to reset the clock on this appliance?
    Provide the instructions in English and French.
    If instructions include buttons, also explain where those buttons are physically located.
    
    
    -------Response--------
    To reset the clock on this appliance, follow these steps:
    
    1. **Press the "CLEAR/OFF" button.** This button is located in the bottom right corner of the appliance's control panel.
    2. **Press the "CLOCK" button.** This button is located in the top right corner of the appliance's control panel.
    3. **Enter the current time.** Use the number buttons (located in the middle of the appliance's control panel) to input the desired hour and minutes.
    4. **Press the "CLOCK" button again.** This will confirm the time and exit the clock setting mode. 
    
    
    ## French
    
    Pour réinitialiser l'horloge de cet appareil, suivez ces étapes :
    
    1. **Appuyez sur le bouton « CLEAR/OFF ».** Ce bouton se trouve dans le coin inférieur droit du panneau de commande de l'appareil.
    2. **Appuyez sur le bouton « CLOCK ».** Ce bouton se trouve dans le coin supérieur droit du panneau de commande de l'appareil.
    3. **Entrez l'heure actuelle.** Utilisez les boutons numériques (situés au milieu du panneau de commande de l'appareil) pour saisir l'heure et les minutes souhaitées.
    4. **Appuyez à nouveau sur le bouton « CLOCK ».** Cela confirmera l'heure et quittera le mode de réglage de l'horloge. 
    

Note: The response may not be completely accurate, as the model may hallucinate; however, the model is able to identify the location of buttons and translate in a single query. To mitigate hallucinations, one approach is to ground the LLM with retrieval-augmented generation, which is outside the scope of this notebook.


## Understanding entity relationships in technical diagrams

Gemini has multimodal capabilities that enable it to understand diagrams and take actionable steps, such as optimization or code generation. This example demonstrates how Gemini can decipher an entity relationship (ER) diagram, understand the relationships between tables, identify requirements for optimization in a specific environment like BigQuery, and even generate corresponding code.



```
image_er_url = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/er.png"
image_er = load_image_from_url(image_er_url)

prompt = "Document the entities and relationships in this ER diagram."

contents = [prompt, image_er]

# Use a more deterministic configuration with a low temperature
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    candidate_count=1,
    max_output_tokens=2048,
)

responses = multimodal_model.generate_content(
    contents,
    generation_config=generation_config,
    stream=True,
)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")
```

    -------Prompt--------
    Document the entities and relationships in this ER diagram.
    


    
![png](output_29_1.png)
    


    
    -------Response--------
    ## Entities
    
    - **Category:** Represents different categories of items.
        - **Attributes:** category, category_name
    - **Vendor:** Represents different vendors supplying items.
        - **Attributes:** vendor_no, vendor
    - **Item:** Represents different items sold.
        - **Attributes:** item, description, pack, liter_size
    - **Sales:** Represents sales transactions.
        - **Attributes:** date, store, category, vendor_no, item, state_btl_cost, btl_price, bottle_qty, total
    - **Convenience_store:** Represents convenience stores.
        - **Attributes:** store
    - **Store:** Represents stores.
        - **Attributes:** store, name, address, city, zipcode, store_location, county_number
    - **County:** Represents counties.
        - **Attributes:** county_number, county
    
    ## Relationships
    
    - **Category** **1:N** **Sales:** A category can have many sales transactions.
    - **Vendor** **1:N** **Sales:** A vendor can have many sales transactions.
    - **Item** **1:N** **Sales:** An item can have many sales transactions.
    - **Sales** **1:1** **Convenience_store:** A sales transaction belongs to one convenience store.
    - **Convenience_store** **1:1** **Store:** A convenience store is a type of store.
    - **County** **1:N** **Store:** A county can have many stores.
    - **Item** **1:N** **Sales:** An item can be sold in many sales transactions.
    - **County** **1:N** **Sales:** A county can have many sales transactions.
    
    ## Notes
    
    - The relationship between **Sales** and **Convenience_store** is a weak entity relationship, as a sales transaction cannot exist without a convenience store.
    - The relationship between **Convenience_store** and **Store** is a specialization relationship, as a convenience store is a specific type of store.
    - The relationship between **Item** and **Sales** is a many-to-many relationship, as an item can be sold in many sales transactions and a sales transaction can include many items.
    - The relationship between **County** and **Sales** is a many-to-many relationship, as a county can have many sales transactions and a sales transaction can occur in many counties.
    

## Recommendations based on multiple images

Gemini is capable of image comparison and providing recommendations. This may be useful in industries like e-commerce and retail.

Below is an example of choosing which pair of glasses would be better suited to an oval-shaped face:



```
image_glasses1_url = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/glasses1.jpg"
image_glasses2_url = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/glasses2.jpg"
image_glasses1 = load_image_from_url(image_glasses1_url)
image_glasses2 = load_image_from_url(image_glasses2_url)

prompt1 = """
Which of these glasses you recommend for me based on the shape of my face?
I have an oval shape face.
----
Glasses 1:
"""
prompt2 = """
----
Glasses 2:
"""
prompt3 = """
----
Explain how you reach out to this decision.
Provide your recommendation based on my face shape, and reasoning for each in JSON format.
"""

contents = [prompt1, image_glasses1, prompt2, image_glasses2, prompt3]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")
```

    -------Prompt--------
    
    Which of these glasses you recommend for me based on the shape of my face?
    I have an oval shape face.
    ----
    Glasses 1:
    
    


    
![png](output_31_1.png)
    


    
    ----
    Glasses 2:
    
    


    
![png](output_31_3.png)
    


    
    ----
    Explain how you reach out to this decision.
    Provide your recommendation based on my face shape, and reasoning for each in JSON format.
    
    
    -------Response--------
    ```json
    {
      "recommendation": "Glasses 2",
      "reasoning": {
        "Glasses 1": "Square shaped glasses can make an oval face appear wider, which can be undesirable. They can also make the face look boxy.",
        "Glasses 2": "Round glasses are a great choice for oval faces as they complement the natural shape of the face, creating a balanced and harmonious look."
      }
    }
    ```

## Similarity/Differences

Gemini can compare images and identify similarities or differences between objects.

The following example shows two scenes from Marienplatz in Munich, Germany that are slightly different. Gemini can compare between the images and find similarities/differences:



```
image_landmark1_url = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/landmark1.jpg"
image_landmark2_url = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/landmark2.jpg"
image_landmark1 = load_image_from_url(image_landmark1_url)
image_landmark2 = load_image_from_url(image_landmark2_url)

prompt1 = """
Consider the following two images:
Image 1:
"""
prompt2 = """
Image 2:
"""
prompt3 = """
1. What is shown in Image 1? Where is it?
2. What is similar between the two images?
3. What is difference between Image 1 and Image 2 in terms of the contents or people shown?
"""

contents = [prompt1, image_landmark1, prompt2, image_landmark2, prompt3]

generation_config = GenerationConfig(
    temperature=0.0,
    top_p=0.8,
    top_k=40,
    candidate_count=1,
    max_output_tokens=2048,
)

responses = multimodal_model.generate_content(
    contents,
    generation_config=generation_config,
    stream=True,
)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")
```

    -------Prompt--------
    
    Consider the following two images:
    Image 1:
    
    


    
![png](output_33_1.png)
    


    
    Image 2:
    
    


    
![png](output_33_3.png)
    


    
    1. What is shown in Image 1? Where is it?
    2. What is similar between the two images?
    3. What is difference between Image 1 and Image 2 in terms of the contents or people shown?
    
    
    -------Response--------
    1. Image 1 shows the Feldherrnhalle, a building in Munich, Germany. It is located on the Odeonsplatz, a square in the city center.
    2. Both images show the same scene, the Feldherrnhalle and the Odeonsplatz.
    3. Image 1 shows more people than Image 2. In Image 1, there are people walking around the square, sitting on benches, and standing in front of the Feldherrnhalle. In Image 2, there are fewer people, and they are mostly walking in the street.

## Generating a video description

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

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")
```

    -------Prompt--------
    
    What is shown in this video?
    Where should I go to see it?
    What are the top 5 places in the world that look like this?
    
    


<video src="https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/mediterraneansea.mp4" controls  width="600" >
      Your browser does not support the <code>video</code> element.
    </video>


    
    -------Response--------
    This video shows a pier in Antalya, Turkey.
    To see the pier, go to Antalya, Turkey.
    
    Here are some top 5 places in the world that look like this:
    
    1. **Amalfi Coast, Italy:** The Amalfi Coast is known for its dramatic cliffs, beautiful beaches, and charming towns. 
    2. **Dubrovnik, Croatia:** Dubrovnik is a walled city on the Adriatic Sea, with a stunning coastline and picturesque harbor.
    3. **Santorini, Greece:** Santorini is a volcanic island in the Aegean Sea, with whitewashed houses and stunning views of the caldera.
    4. **Positano, Italy:** Positano is a town on the Amalfi Coast, known for its colorful houses and breathtaking views.
    5. **Nice, France:** Nice is a city on the French Riviera, with a beautiful coastline, charming old town, and vibrant culture. 
    

> You can confirm that the location is indeed Antalya, Turkey by visiting the Wikipedia page: https://en.wikipedia.org/wiki/Antalya


## Extracting tags of objects throughout the video

Gemini can also extract tags throughout a video.

> Video: https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/photography.mp4



```
prompt = """
Answer the following questions using the video only:
- What is in the video?
- What is the action in the video?
- Provide 10 best tags for this video?
"""
video = Part.from_uri(
    uri="gs://github-repo/img/gemini/multimodality_usecases_overview/photography.mp4",
    mime_type="video/mp4",
)
contents = [prompt, video]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")
```

    -------Prompt--------
    
    Answer the following questions using the video only:
    - What is in the video?
    - What is the action in the video?
    - Provide 10 best tags for this video?
    
    


<video src="https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/photography.mp4" controls  width="600" >
      Your browser does not support the <code>video</code> element.
    </video>


    
    -------Response--------
    - The video shows a man taking pictures with his camera in a room with rustic furniture and decor.
    - The action is the man taking pictures.
    - Here are 10 tags for the video:
        - photography
        - camera
        - rustic
        - decor
        - interior design
        - home
        - man
        - vintage
        - travel
        - lifestyle

## Asking more questions about a video

Below is another example of using Gemini to ask questions the video and return a JSON response.

> Video: https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/pixel8.mp4  
> _Note: Although this video contains audio, Gemini does not currently support audio input and will only answer based on the video._


```
prompt = """
Answer the following questions using the video only:
What is the profession of the main person?
What are the main features of the phone highlighted?
Which city was this recorded in?
Provide the answer JSON.
"""
video = Part.from_uri(
    uri="gs://github-repo/img/gemini/multimodality_usecases_overview/pixel8.mp4",
    mime_type="video/mp4",
)
contents = [prompt, video]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")
```

    -------Prompt--------
    
    Answer the following questions using the video only:
    What is the profession of the main person?
    What are the main features of the phone highlighted?
    Which city was this recorded in?
    Provide the answer JSON.
    
    


<video src="https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/pixel8.mp4" controls  width="600" >
      Your browser does not support the <code>video</code> element.
    </video>


    
    -------Response--------
    ```json
    {
     "profession": "photographer",
     "features": [
      "Video Boost",
      "Night Sight"
     ],
     "city": "Tokyo"
    }
    ```

## Retrieving extra information beyond the video


> Video: https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/ottawatrain3.mp4



```
prompt = """
Which line is this?
where does it go?
What are the stations/stops of this line?
"""
video = Part.from_uri(
    uri="gs://github-repo/img/gemini/multimodality_usecases_overview/ottawatrain3.mp4",
    mime_type="video/mp4",
)
contents = [prompt, video]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")
```

    -------Prompt--------
    
    Which line is this?
    where does it go?
    What are the stations/stops of this line?
    
    


<video src="https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/ottawatrain3.mp4" controls  width="600" >
      Your browser does not support the <code>video</code> element.
    </video>


    
    -------Response--------
    This is the **Rideau LRT** in **Ottawa**, Canada. It runs between **Tunney's Pasture** and **Blair Station**. 
    
    Some of the stations it goes through include:
    
    * Tunney's Pasture
    * Bayview
    * Lebreton
    * Pimisi
    * Parliament
    * Rideau
    * uOttawa
    * Lees
    * Billings Bridge
    * St. Laurent
    *  Hintonburg
    *  Lyon
    *  Parliament
    *  Rideau
    *  uOttawa
    *  Lees
    *  Heron
    *  Cumberland
    *  Blair
    
    Please note that this list may not be exhaustive as the line has many stops. For a full list of stations, it is best to consult the official Rideau LRT website.

> You can confirm that this is indeed the Confederation Line on Wikipedia here: https://en.wikipedia.org/wiki/Confederation_Line

