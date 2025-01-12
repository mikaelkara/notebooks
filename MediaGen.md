# Building a Video Generation Pipeline with Llama3

<!--
## Video Walkthrough
You can follow along this notebook on our [video walkthrough](https://youtu.be/UO8QZ2qBonw).

[![Workshop Walkthrough Still](http://img.youtube.com/vi/UO8QZ2qBonw/0.jpg)](http://www.youtube.com/watch?v=UO8QZ2qBonw "Workshop Walkthrough") -->

## Overview
In this notebook you'll learn how to build a powerful media generation pipeline in a few simple steps. More specifically, this pipeline will generate a ~1min long food recipe video entirely from just the name of a dish.

This demo in particular showcases the ability for Llama3 to produce creative recipes while following JSON formatting guidelines very well.

[Example Video Output for "dorritos consomme"](https://drive.google.com/file/d/1AP3VUlAmOUU6rcZp1wQ4v4Fyf5-0tky_/view?usp=drive_link)

![overview](https://raw.githubusercontent.com/tmoreau89/image-assets/main/llama3_hackathon/mediagen_llama3.png)

Let's take a look at the high level steps needed to go from the name of a dish, e.g. "baked alaska" to a fully fledged recipe video:
1. We use a Llama3-70b-instruct LLM to generate a recipe from the name of a dish. The recipe is formatted in JSON which breaks down the recipe into the following fields: recipe title, prep time, cooking time, difficulty, ingredients list and instruction steps.
2. We use SDXL to generate a frame for the finished dish, each one of the ingredients, and each of the recipe steps.
3. We use Stable Video Diffusion 1.1 to animate each frame into a short 4 second video.
4. Finally we stitch all of the videos together using MoviePy, add subtitles and a soundtrack.

## Pre-requisites

### OctoAI
We'll use [OctoAI](https://octo.ai/) to power all of the GenAI needs of this notebook: LLMs, image gen, image animation.
* To use OctoAI, you'll need to go to https://octoai.cloud/ and sign in using your Google or GitHub account.
* Next you'll need to generate an OctoAI API token by following these [instructions](https://octo.ai/docs/getting-started/how-to-create-an-octoai-access-token). Keep the API token in hand, we'll need it further down in this notebook.

In this example we will use the Llama 3 70b instruct model. You can find more on Llama models on the [OctoAI text generation solution page](https://octoai.cloud/text).

At the time of writing this notebook the following Llama models are available on OctoAI:
* meta-llama-3-8b-instruct
* meta-llama-3-70b-instruct
* codellama-7b-instruct
* codellama-13b-instruct
* codellama-34b-instruct
* llama-2-13b-chat
* llama-2-70b-chat
* llamaguard-7b

### Local Python Notebook
We highly recommend launching this notebook from a fresh python environment, for instance you can run the following:
```
python3 -m venv .venv         
source .venv/bin/activate
```
All you need to run this notebook is to install jupyter notebook with `python3 -m pip install notebook` then run `jupyter notebook` ([link](https://jupyter.org/install)) in the same directory as this `.ipynb` file.
You don't need to install additional pip packages ahead of running the notebook, since those will be installed right at the beginning. You will need to ensure your system has `imagemagick` installed by following the [instructions](https://imagemagick.org/script/download.php).


```python
# This can take a few minutes on Colab, please be patient!
# Note: in colab you may have to restart the runtime to get all of the
# dependencies set up properly (a message will instruct you to do so)
import platform
if platform.system() == "Linux":
    # Tested on colab - requires a few steps to get imagemagick installed correctly
    # https://github.com/Zulko/moviepy/issues/693#issuecomment-622997875
    ! apt install imagemagick &> /dev/null
    ! apt install ffmpeg &> /dev/null
    ! pip install moviepy[optional] &> /dev/null
    ! sed -i '/<policy domain="path" rights="none" pattern="@\*"/d' /etc/ImageMagick-6/policy.xml
elif platform.system() == "Darwin":
    # Tested on a macbook on macOS Sonoma
    ! brew install imagemagick
    ! brew reinstall ffmpeg
    ! pip install moviepy
else:
    print("Please install imagemagick on your system by following the instructions above")
# Let's proceed by installing the necessary pip packages
! pip install langchain==0.1.19 octoai===1.0.2 openai pillow ffmpeg devtools
```


```python
# Next let's use the getpass library to enter the OctoAI API token you just
# obtained in the pre-requisite step
from getpass import getpass
import os

OCTOAI_API_TOKEN = getpass()
os.environ["OCTOAI_API_TOKEN"] = OCTOAI_API_TOKEN
```

# 1. Recipe Generation with Langchain using a Llama3-70b-instruct hosted on OctoAI

In this first section, we're going to show how you can use Llama3-70b-instruct LLM hosted on OctoAI. Here we're using Langchain, a popular Python based library to build LLM-powered application.

[Llama 3](https://llama.meta.com/llama3/) is Meta AI's latest open source model in the Llama family.

The key here is to rely on the OctoAIEndpoint LLM by adding the following line to your python script:
```python
from langchain.llms.octoai_endpoint import OctoAIEndpoint
```

Then you can instantiate your `OctoAIEndpoint` LLM by passing in under the `model_kwargs` dictionary what model you wish to use (there is a rather wide selection you can consult [here](https://octo.ai/docs/text-gen-solution/getting-started#self-service-models)), and what the maximum number of tokens should be set to.

Next you need to define your prompt template. The key here is to provide enough rules to guide the LLM into generating a recipe with just the right amount of information and detail. This will make the text generated by the LLM usable in the next generation steps (image generation, image animation etc.).

> ⚠️ Note that we're generating intentionally a short recipe according to the prompt template - this is to ensure we can go through this notebook fairly quickly the first time. If you want to generate a full recipe, delete the following line from the prompt template.
```
Use only two ingredients, and two instruction steps.
```

Finally we create an LLM chain by passing in the LLM and the prompt template we just instantiated.

This chain is now ready to be invoked by passing in the user input, namely: the name of the dish to generate a  recipe for. Let's invoke the chain and see what recipe our LLM just thought about.


```python
import json
from langchain.llms.octoai_endpoint import OctoAIEndpoint
from langchain.output_parsers import PydanticOutputParser
from langchain import PromptTemplate, LLMChain
from pydantic import BaseModel, Field
from typing import List

# OctoAI LLM endpoint
llm = OctoAIEndpoint(
    model = "meta-llama-3-70b-instruct",
    max_tokens = 1024,
    temperature = 0.01
)

# Define a JSON format for our recipe using Pydantic to declare our data model
class Ingredient(BaseModel):
    """The object representing an ingredient"""
    item: str = Field(description="Ingredient")
    illustration: str = Field(description="Text-based detailed visual description of the ingredient for a photograph or illustrator")

class RecipeStep(BaseModel):
    """The object representing a recipe steps"""
    item: str = Field(description="Recipe step/instruction")
    illustration: str = Field(description="Text-based detailed visual description of the instruction for a photograph or illustrator")

class Recipe(BaseModel):
    """The format of the recipe answer."""
    dish_name: str = Field(description="Name of the dish")
    ingredient_list: List[Ingredient] = Field(description="List of the ingredients")
    recipe_steps: List[RecipeStep] = Field(description="List of the recipe steps")
    prep_time: int = Field(description="Recipe prep time in minutes")
    cook_time: int = Field(description="Recipe cooking time in minutes")
    difficulty: str = Field(description="Rating in difficulty, can be easy, medium, hard")

# Pydantic output parser
parser = PydanticOutputParser(pydantic_object=Recipe)

# Define a recipe template
template = """
You are a food recipe generator. 

Given the name of a dish, generate a recipe that's easy to follow and leads to a delicious and creative dish.

Use only two ingredients, and two instruction steps.

Here are some rules to follow at all costs:
0. Respond back only as only JSON!!!
1. Provide a list of ingredients needed for the recipe.
2. Provide a list of instructions to follow the recipe.
3. Each instruction should be concise (1 sentence max) yet informative. It's preferred to provide more instruction steps with shorter instructions than fewer steps with longer instructions.
4. For the whole recipe, provide the amount of prep and cooking time, with a classification of the recipe difficulty from easy to hard.

{format_instructions}

Human: Generate a recipe for a dish called {human_input}
AI: """

prompt = PromptTemplate(
    template=template,
    input_variables=["human_input"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Set up the language model chain
llm_chain = prompt | llm | parser
```


```python
# Let's request user input for the recipe name
print("Provide a recipe name, e.g. llama 3 spice omelette")
recipe_title = input()
```


```python
from devtools import pprint

# Invoke the LLM chain, extract the JSON and print the response
recipe_dict = llm_chain.invoke({"human_input": recipe_title})
recipe_dict = json.loads(recipe_dict.json())
```

# 2. Generate images that narrate the recipe with SDXL hosted on OctoAI

In this section we'll rely on OctoAI's SDK to invoke the image generation endpoint powered by Stable Diffusion XL. Now that we have our recipe stored in JSON object we'll generate the following images:
* A set of images for every ingredient used in the recipe, stored in `ingredient_images`
* A set of images for every step in the recipe, stored in `step_images`
* An image of the final dish, stored under `final_dish_still`

We rely on the OctoAI Python SDK to generate those images with SDXL. You just need to instantiate the OctoAI ImageGenerator with your OctoAI API token, then invoke the `generate` method for each set of images you want to produce. You'll need to pass in the following arguments:
* `engine` which selects what model to use - we use SDXL here
* `prompt` which describes the image we want to generate
* `negative_prompt` which provides image attributes/keywords that we absolutely don't want to have in our final image
* `width`, `height` which helps us specify a resolution and aspect ratio of the final image
* `sampler` which is what's used in every denoising step, you can read more about them [here](https://stable-diffusion-art.com/samplers/)
* `steps` which specifies the number of denoising steps to obtain the final image
* `cfg_scale` which specifies the configuration scale, which defines how closely to adhere to the original prompt
* `num_images` which specifies the number of images to generate at once
* `use_refiner` which when turned on lets us use the SDXL refiner model which enhances the quality of the image
* `high_noise_frac` which specifies the ratio of steps to perform with the base SDXL model vs. refiner model
* `style_preset` which specifies a stype preset to apply to the negative and positive prompts, you can read more about them [here](https://stable-diffusion-art.com/sdxl-styles/)

To read more about the API and what options are supported in OctoAI, head over to this [link](https://octoai.cloud/media/image-gen?mode=api).

**Note:** Looking to use a specific SDXL checkpoint, LoRA or controlnet for your image generation needs? You can manage and upload your own collection of stable diffusion assets via the [OctoAI CLI](https://octo.ai/docs/media-gen-solution/uploading-a-custom-asset-to-the-octoai-asset-library), or via the [web UI](https://octoai.cloud/assets?isPublic=false). You can then invoke your own [checkpoint](https://octo.ai/docs/media-gen-solution/customizations/checkpoints), [LoRA](https://octo.ai/docs/media-gen-solution/customizations/loras), [textual inversion](https://octo.ai/docs/media-gen-solution/customizations/textual-inversions), or [controlnet](https://octo.ai/docs/media-gen-solution/customizations/controlnets) via the `ImageGenerator` API.


```python
from PIL import Image
from io import BytesIO
from base64 import b64encode, b64decode
from octoai import client as octo_client

# Instantiate the OctoAI SDK image generator
octo_client = octo_client.OctoAI(api_key=OCTOAI_API_TOKEN)

# Ingredients stills dictionary (Ingredient -> Image)
ingredient_images = {}
# Recipe steps stills dictionary (Step -> Image)
step_images = {}

# Iterate through ingredients and recipe steps
for recipe_list, dst in zip(["ingredient_list", "recipe_steps"], [ingredient_images, step_images]):
  for element in recipe_dict[recipe_list]:
      # We do some simple prompt engineering to achieve a consistent style
      prompt = "RAW photo, Fujifilm XT, clean bright modern kitchen photograph, ({})".format(element["illustration"])
      # The parameters below can be tweaked as needed, the resolution is intentionally set to portrait mode
      image_resp = octo_client.image_gen.generate_sdxl(
          prompt=prompt,
          negative_prompt="Blurry photo, distortion, low-res, poor quality, watermark",
          width=768,
          height=1344,
          num_images=1,
          sampler="DPM_PLUS_PLUS_2M_KARRAS",
          steps=30,
          cfg_scale=12,
          use_refiner=True,
          high_noise_frac=0.8,
          style_preset="Food Photography",
      )
      image_str = image_resp.images[0].image_b64
      image = Image.open(BytesIO(b64decode(image_str)))
      dst[element["item"]] = image
      display(dst[element["item"]])
```


```python
# Final dish in all of its glory
prompt = "RAW photo, Fujifilm XT, clean bright modern kitchen photograph, professionally presented ({})".format(recipe_dict["dish_name"])
image_resp = octo_client.image_gen.generate_sdxl(
    prompt=prompt,
    negative_prompt="Blurry photo, distortion, low-res, poor quality",
    width=768,
    height=1344,
    num_images=1,
    sampler="DPM_PLUS_PLUS_2M_KARRAS",
    steps=30,
    cfg_scale=12,
    use_refiner=True,
    high_noise_frac=0.8,
    style_preset="Food Photography",
)
image_str = image_resp.images[0].image_b64
final_dish_still = Image.open(BytesIO(b64decode(image_str)))
display(final_dish_still)
```

# 3. Animate the images with Stable Video Diffusion 1.1 hosted on OctoAI

In this section we'll rely once again on OctoAI's SDK to invoke the image animation endpoint powered by Stable Video Diffusion 1.1. In the last section we generated a handful of images which we're now going to animate:
* A set of videos for every ingredient used in the recipe, stored in `ingredient_videos`
* A set of videos for every step in the recipe, stored in `steps_videos`
* An videos of the final dish, stored under `final_dish_video`

From these we'll be generating 25-frame videos using the image animation API in OctoAI's Python SDK. You just need to instantiate the OctoAI VideoGenerator with yout OctoAI API token, then invoke the `generate` method for each animation you want to produce. You'll need to pass in the following arguments:
* `engine` which selects what model to use - we use SVD here
* `image` which encodes the input image we want to animate as a base64 string
* `steps` which specifies the number of denoising steps to obtain each frame in the video
* `cfg_scale` which specifies the configuration scale, which defines how closely to adhere to the image description
* `fps` which specifies the numbers of frames per second
* `motion scale` which indicates how much motion should be in the generated animation
* `noise_aug_strength` which specifies how much noise to add to the initial images - a higher value encourages more creative videos
* `num_video` which represents how many output animations to generate

To read more about the API and what options are supported in OctoAI, head over to this [link](https://octoai.cloud/media/animate?mode=api).

**Note:** this step will take a few minutes, as each video takes about 30s to generate and that we're generating each video sequentially. For faster execution time all of these video generation calls can be done asynchronously, or in multiple threads.


```python
# We'll need this helper to convert PIL images into a base64 encoded string
def image_to_base64(image: Image) -> str:
  buffered = BytesIO()
  image.save(buffered, format="JPEG")
  img_b64 = b64encode(buffered.getvalue()).decode("utf-8")
  return img_b64
```


```python
# Generate a video for the final dish presentation (it'll be used in the intro and at the end)
video_resp = octo_client.image_gen.generate_svd(
    image=image_to_base64(final_dish_still),
    steps=25,
    cfg_scale=3,
    fps=6,
    motion_scale=0.5,
    noise_aug_strength=0.02,
    num_videos=1,
)
final_dish_video = video_resp.videos[0]
```


```python
from IPython.display import HTML
from moviepy.editor import *

# This is a helper function that gets the video dumped locally
def getVideoFileClip(video, fn):
    with open(fn, 'wb') as wfile:
        wfile.write(b64decode(video.video))
    vfc = VideoFileClip(fn)
    return vfc

# View the video to confirm
getVideoFileClip(final_dish_video, "final_dish.mp4")
mp4 = open('final_dish.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)
```


```python
# Generate the ingredients videos (doing synchronously, so it's gonna be slow)
# TODO: parallelize to make this run faster!

# Dictionary that stores the videos for ingredients (ingredient -> video)
ingredient_videos = {}
# Dictionary that stores the videos for recipe steps (step -> video)
steps_videos = {}

# Iterate through ingredients and recipe steps
for recipe_list, src, dst in zip(["ingredient_list", "recipe_steps"], [ingredient_images, step_images], [ingredient_videos, steps_videos]):
    # Iterate through each ingredient / step
    for item in recipe_dict[recipe_list]:
        key = item["item"]
        # Retrieve the image from the ingredient_images dict
        still = src[key]
        # Generate a video with the OctoAI video generator
        video_resp = octo_client.image_gen.generate_svd(
            image=image_to_base64(still),
            steps=25,
            cfg_scale=3,
            fps=6,
            motion_scale=0.5,
            noise_aug_strength=0.02,
            num_videos=1,
        )
        dst[key] = video_resp.videos[0]
```

# 4. Create a video montage with MoviePy

In this section we're going to rely on the MoviePy library to create a montage of the videos.

For each short animation (dish, ingredients, steps), we also have corresponding text that goes with it from the original `recipe_dict` JSON object. This allows us to generate a montage captions.

Each video having 25 frames and being a 6FPS video, they will last 4.167s each. Because the ingredients list can be rather long, we crop each video to a duration of 2s to keep the flow of the video going. For the steps video, we play 4s of each clip given that we need to give the viewer time to read the instructions.




```python
from IPython.display import Video
from moviepy.video.tools.subtitles import SubtitlesClip
import textwrap

# Video collage
collage = []

# To prepare the closed caption of the video, we define
# two durations: short duration (2.0s) and long duration (4.0s)
short_duration = 2
long_duration = 4
# We keep track of the time ellapsed
t = 0
# This sub list will contain tuples in the following form:
# ((t_start, t_end), "caption")
subs = []

# Let's create the intro clip presenting the final dish
vfc = getVideoFileClip(final_dish_video, "final_dish.mp4")
collage.append(vfc.subclip(0, long_duration))
# Add the subtitle which provides the name of the dish, along with prep time, cook time and difficulty
subs.append(((t, t+long_duration), "{} Recipe\nPrep: {}min\nCook: {}min\nDifficulty: {}".format(
    recipe_dict["dish_name"].title(), recipe_dict["prep_time"], recipe_dict["cook_time"], recipe_dict["difficulty"]))
)
t += long_duration

# Go through the ingredients list to stich together the ingredients clip
for idx, ingredient in enumerate(recipe_dict["ingredient_list"]):
    # Write the video to disk and load it as a VideoFileClip
    key = ingredient["item"]
    vfc = getVideoFileClip(ingredient_videos[key], 'clip_ingredient_{}.mp4'.format(idx))
    collage.append(vfc.subclip(0, short_duration))
    # Add the subtitle which just provides each ingredient
    subs.append(((t, t+short_duration), "Ingredients:\n{}".format(textwrap.fill(key, 35))))
    t += short_duration

# Go through the recipe steps to stitch together each step of the recipe video
for idx, step in enumerate(recipe_dict["recipe_steps"]):
    # Write the video to disk and load it as a VideoFileClip
    key = step["item"]
    vfc = getVideoFileClip(steps_videos[key], 'clip_step_{}.mp4'.format(idx))
    collage.append(vfc.subclip(0, long_duration))
    # Add the subtitle which just provides each recipe step
    subs.append(((t, t+long_duration), "Step {}:\n{}".format(idx, textwrap.fill(key, 35))))
    t += long_duration

# Add the outtro clip
vfc = VideoFileClip('final_dish.mp4'.format(idx))
collage.append(vfc.subclip(0, long_duration))
# Add the subtitle: Enjoy your {dish_name}
subs.append(((t, t+long_duration), "Enjoy your {}!".format(recipe_title.title())))
t += long_duration
```


```python
# Concatenate the clips into one initial collage
final_clip = concatenate_videoclips(collage)
final_clip.to_videofile("collage.mp4", fps=vfc.fps)

# Preview the video
mp4 = open('collage.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)
```


```python
# Add subtitles to the collage
generator = lambda txt: TextClip(
    txt,
    font='Century-Schoolbook-Roman',
    fontsize=30,
    color='white',
    stroke_color='black',
    stroke_width=1.5,
    method='label',
    transparent=True
)
subtitles = SubtitlesClip(subs, generator)
result = CompositeVideoClip([final_clip, subtitles.margin(bottom=70, opacity=0).set_pos(('center','bottom'))])
result.write_videofile("collage_sub.mp4", fps=vfc.fps)
```


```python
# Now add a soundtrack: you can browse https://pixabay.com for a track you like
# I'm downloading a track called "once in paris" by artist pumpupthemind
import subprocess

subprocess.run(["wget", "-O", "audio_track.mp3", "http://cdn.pixabay.com/download/audio/2023/09/29/audio_0eaceb1002.mp3"])

# Add the soundtrack to the video
videoclip = VideoFileClip("collage_sub.mp4")
audioclip = AudioFileClip("audio_track.mp3").subclip(0, videoclip.duration)
video = videoclip.set_audio(audioclip)
video.write_videofile("collage_sub_sound.mp4")
```


```python
# Enjoy your video!
mp4 = open('collage_sub_sound.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)
```

**Authors**
- Thierry Moreau, OctoAI - tmoreau@octo.ai
- Pedro Toruella, OctoAI - ptoruella@octo.ai

Join [OctoAI Discord](https://discord.com/invite/rXTPeRBcG7)
