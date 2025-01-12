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

# Brochure Creation Tool using Imagen

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/vision/use-cases/brochure-creation-using-imagen/brochure_creation_using_imagen.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fvision%2Fuse-cases%2Fbrochure-creation-using-imagen%2Fbrochure_creation_using_imagen.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/vision/use-cases/brochure-creation-using-imagen/brochure_creation_using_imagen.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/vision/use-cases/brochure-creation-using-imagen/brochure_creation_using_imagen.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Sanchit Latawa](https://github.com/slatawa) , [Divya Veerapandian](https://github.com/divyapandian5)

# Brochure Creation Tool using Imagen

A Generative AI driven tool utilizing Google Vertex AI and Imagen to create product Brochure material for promotional materials.

# Objectives

- Create background images for product brochures
- Place product image on top of the background along with text like company name
- Add company logo image, if provided


# Solution Architecture

![SOlution](./architecture.png) 

# Getting Started

# Install Vertex AI SDK & Other dependencies


```
%pip install --upgrade --user -q google-cloud-aiplatform rembg Pillow opencv-python numpy requests gradio
```

    Note: you may need to restart the kernel to use updated packages.
    

Colab only: Run the following cell to restart the kernel. For Vertex AI Workbench you can restart the terminal using the button on top.


```
import sys

if "google.colab" in sys.modules:
    # Automatically restart kernel after installs so that your environment can access the new packages
    import IPython

    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)
```

# Colab Only
You will need to run the following cell to authenticates your Colab environment with your Google Cloud account.


```
if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Import Required Libraries


```
from enum import Enum
import os
import random
import time

from IPython import display
from PIL import Image
from PIL import Image as PIL_Image
from PIL import ImageColor
import gradio as gr
import matplotlib.pyplot as plt
from rembg import remove
import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.preview.vision_models import ImageGenerationModel
```

### ALl set lets run the pipeline 

Set these variables below:

1. Enter `YOUR_PROJECT_ID` in project_id
2. Enter `REGION` in REGION

# Env Variables


```
PROJECT_ID = "[your-project-id]"
# "sl-test-project-353312"
REGION = "[your-region]"
# "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)
```

### Get token

### Sample Background prompts to choose from 

These prompts would be used to generate Image using Imagen and used as background of the product.


```
EXAMPLE_PROMPTS = [
    """Display a vibrant, high-resolution background illuminated by a combination of soft, diffused light and a touch of dramatic side lighting.The background consists of a clean, minimalist setting with subtle geometric patterns""",
    """Nature's Touch Neutral beige background with a subtle gradient from a light brown to an off-white A faint, organic pattern of leaves and branches creates a natural and serene ambiance The pattern is intentionally blurred to prevent it from overpowering the product""",
    """Modern Simplicity Clean, white background with a soft gradient from a light blue to a pale gray Minimalistic lines form an abstract, geometric pattern in the background The pattern is subtle and fades into the background, allowing the product to take center stage""",
    """Neutral Paper with Faint Linen Texture A neutral beige background resembling paper or canvas, providing a warm and inviting backdrop. A faint linen texture adds a touch of sophistication and depth, creating a subtle visual interest. The background conveys a sense of timeless elegance and quality, reflecting Company's commitment to craftsmanship and enduring products.""",
    """Soft Pastel Gradient with Organic Shapes A soft gradient background transitioning between steel and blue, creating a sense of tranquility and optimism. Organic shapes, such as flowing lines or abstract curves, add a touch of movement and playfulness. The background evokes a sense of harmony, innovation, and a brighter future, embodying Company's dedication to societal advancement.""",
    """Light of the Future"*** **Minimalist Design:** Simple, flowing curves convey a sense of progress and fluidity.* **Subtle Gradient:** A subtle gradient from light blue to white creates a luminous backdrop, suggesting a bright future ahead.* **Neutral Colors:** Light blue and white provide a clean and airy neutral background.* **Subtle Pattern:** A faint geometric pattern of hexagons symbolizes technology and connectivity, reflecting Company's problem-solving nature.* **Company's Philosophy:** The flowing curves evoke a sense of optimism and a journey toward a brighter tomorrow.""",
]
FUTURISTIC_EXAMPLE_PROMPTS = [
    """The background features a pristine, ultra-high-definition image of a CVD Coated Carbide surface. The surface is meticulously rendered with a high degree of precision, showcasing its smooth, lustrous finish.""",
    """A grid of abstract electronic circuit patterns in neon hues, forming a dynamic and futuristic backdrop""",
    """A seamless blend of organic and technological imagery against a matte gray background..""",
    """Crisp blue gradient background reminiscent of a clear night sky.""",
    """Subtle geometric shapes in soft blue and white, representing the harmony and fusion of technology and nature""",
    """Delicately sketched soft light blue and white circuit patterns interwoven into the gradient, resembling ethereal constellations""",
    """A shimmering, metallic horizon line at the base of the image, representing the intersection between cutting-edge technology and the boundless possibilities of the future.""",
    """A vast, white expanse stretching out to the horizon, with faint blue lines representing connectivity and growth.""",
    """Concept : Binary Landscape Image: A minimalist landscape composed of binary code that gradually morphs into intricate patterns. Color Palette: Blue and white, with subtle hints of gray to convey the contrast between technology and nature. Technical Specifications: 4K resolution, dynamic motion to symbolize the constant evolution of technology.""",
    """Concept : Problem-Solving Matrix Image: An abstract grid of interconnected nodes and lines. The grid represents a complex problem-solving space. Color Palette: White with accents of blue Futuristic Element: The grid is constantly shifting and adapting, symbolizing Company's iterative approach to problem-solving. Company's Philosophy: The image highlights Company's dedication to finding innovative solutions through meticulous analysis and collaboration. Color Palette: Bright blue and white, emphasizing the idea of unity and collaboration. Technical Specifications: 4K resolution, close-up perspective for an intimate and impactful connection.""",
]
```

### Use this code to get possible prompt to generate an Image background. Tune the prompt based on your idea

The below code uses Gemini to generate a prompt which can then be used with Imagen to generate a background image for the product brochure.


```
def generate_imagen_prompt_using_llm(
    temperature: str,  # "Cool", "Warm", or "Neutral"
    colors: str,  # Comma-separated string of colors
    is_blurred: bool,  # "True" or "False"
    patterns: str,  # String representing a single pattern
    style: str,  # String representing the style
) -> str:
    model = generative_models.GenerativeModel("gemini-1.0-pro-001")

    add_blur_text = "Blur the generated Image" if is_blurred else ""

    llm_prompt = f"""Background: 
    As a distinguished marketing expert renowned for crafting captivating visuals, you excel in transforming intricate concepts into sleek, minimalist imagery tailored for brochures, PR materials, and sales collateral.Your expertise shines particularly bright in the electronics industry.
    Client: Company, a global technology trailblazer, epitomizes a profound dedication to leveraging technology for societal advancement. Their ethos underscores a commitment to shaping a better tomorrow through innovation and problem-solving. Company unveils a cutting-edge product and seeks your adept touch for a forward-thinking PR campaign. 
    
    Challenge: Your task is to conceive one background image concept that embody the following principles:
    Colors to be used: shades of {colors}
    Temperature: {temperature}
    Pattern to be used: {patterns}
    Style of the image: {style}
    Minimalist Design: Craft backgrounds with clean lines and simple shapes, eschewing clutter or intricate patterns that may divert attention from the product.
    Subtle Gradient: Infuse backgrounds with gentle gradients, seamlessly transitioning between hues to lend depth without overpowering the focal point. These gradients should evoke a sense of dimensionality, subtly enhancing the overall visual appeal.
    Subtle Texture or Pattern: Introduce faint patterns or subtle textures such as linen or paper to add visual interest without detracting from the product\'s prominence.
    Additionally, your designs should reflect Company\'s philosophy, embodying their commitment to a brighter future through technological innovation and problem-solving. 
    Ensure your backgrounds convey a sense of optimism, progress, and societal betterment.  Lastly, aim for high-quality, 4K resolution images to meet the client\'s expectations 
    for technical specifications. {add_blur_text}"""

    generation_config = generative_models.GenerationConfig(
        max_output_tokens=2048, temperature=0.9, top_p=1
    )

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    # Neutral Colors: Opt for soft gray, light blue, beige, or pastel tones to provide a neutral backdrop that accentuates the product without overshadowing it.
    response = model.generate_content(
        llm_prompt, generation_config=generation_config, safety_settings=safety_settings
    )

    return response.text
```

### Test Prompt creation using LLM for Imagen


```
temperature = "Cool"  ##  warm , cool ,neutral
colors = "white,blue"
is_blurred = False
patterns = "waves"
style = "Minimalistic"

prompt_for_imagen = generate_imagen_prompt_using_llm(
    temperature, colors, is_blurred, patterns, style
)
print(prompt_for_imagen)
```

    **Concept:**
    
    **Title:** "Waves of Progress"
    
    **Description:**
    
    The background image depicts a series of white waves gently undulating against a cool blue backdrop. The waves are rendered in a minimalist style, with clean lines and a subtle gradient that creates a sense of depth. The image conveys a sense of optimism and progress, as the waves represent the ongoing advancements in technology that Company is driving.
    
    **Visual Elements:**
    
    * **Colors:** Shades of white and blue
    * **Temperature:** Cool
    * **Pattern:** Waves
    * **Style:** Minimalistic
    * **Resolution:** 4K
    
    **Technical Specifications:**
    
    * Dimensions: 3840 x 2160 pixels
    * File size: 10 MB
    * File format: JPEG
    
    **Symbolism:**
    
    The waves in the image symbolize the relentless waves of technological innovation that Company is driving. The white color represents purity and progress, while the blue color represents the company's commitment to shaping a better future. The overall image conveys a sense of optimism and a belief in the power of technology to solve problems and improve society.
    

## Methods required to Generate Image using Imagen API


```
def imagen_generate_sdk(
    model_name: str,
    prompt: str,
    negativePrompt: str,
    sampleCount: int,
    seed: int | None = None,
    disablePersonFace: bool | None = False,  # Whether to disable generating faces
    sampleImageStyle: str | None = None,
) -> list[Image.Image]:
    model = ImageGenerationModel.from_pretrained(model_name)
    response = model.generate_images(
        prompt=prompt, number_of_images=sampleCount, negative_prompt=negativePrompt
    )

    pillow_images = []
    for image in response.images:
        # image_bytes = io.BytesIO(image._image_bytes)  # Convert to in-memory file
        # pillow_images.append(Image.open(image_bytes))
        pillow_images.append(image._pil_image)
    return pillow_images


# Images should be an array of images in PIL image format
def display_images(pil_images):
    scale = 0.25
    sampleImageSize = 1536
    width = int(float(sampleImageSize) * scale)
    height = int(float(sampleImageSize) * scale)
    for index, result in enumerate(pil_images):
        width, height = pil_images[index].size
        print(index)
        display.display(
            pil_images[index].resize(
                (
                    int(pil_images[index].size[0] * scale),
                    int(pil_images[index].size[1] * scale),
                ),
                0,
            )
        )
        print()


def display_image(image):
    plt.imshow(image, cmap="gray")
    plt.axis("off")  # Turn off axes
    plt.show()
```

## Methods required to make Brochure


```
from PIL import Image, ImageColor


def make_brochure(
    subject_image: Image.Image,
    bg_image: Image.Image,
    logo_image: Image.Image,
    banner_color: tuple[int, int, int],  # Assuming RGB tuple
) -> Image.Image:
    # Call overlay to superimpose product image with background
    brochure_image = overlay(subject_image, bg_image, "middle right", None)

    # Define the size of the blue banners
    banner_height = 100  # Height of the banners in pixels
    # banner_color = (41,106,145)  # Teal Blue color for the banners
    banner_image = Image.new("RGB", (842, 100), banner_color)

    # Place the banner images on the top and bottom of the background image
    brochure_image.paste(banner_image, (0, 0))
    brochure_image.paste(banner_image, (0, brochure_image.height - banner_image.height))

    # Add Company Logo
    final_image = add_logo(brochure_image, logo_image, "top left", 0, banner_height)

    return final_image


def add_logo(image, logo, corner, padding, banner_height):
    """Adds the Company logo to one of the corners of an image."""
    # logo = remove(logo, alpha_matting = True,alpha_matting_foreground_threshold=40) # remove background
    logo = logo.resize(
        (int(image.width / 6), banner_height)
    )  ##(int(image.height * 0.1), int(image.height * 0.1)))
    logo = logo.convert("RGBA")
    if corner == "top left":
        image.paste(logo, (padding, padding), logo)
    elif corner == "top right":
        image.paste(logo, (image.width - logo.width - padding, padding), logo)
    elif corner == "bottom left":
        image.paste(logo, (padding, image.height - logo.height - padding), logo)
    elif corner == "bottom right":
        image.paste(
            logo,
            (image.width - logo.width - padding, image.height - logo.height - padding),
            logo,
        )
    return image


class Position(Enum):
    TOP_RIGHT = "top right"
    TOP_CENTER = "top center"
    TOP_LEFT = "top left"
    MIDDLE_RIGHT = "middle right"
    CENTER = "center"
    MIDDLE_LEFT = "middle left"
    BOTTOM_RIGHT = "bottom right"
    BOTTOM_CENTER = "bottom center"
    BOTTOM_LEFT = "bottom left"


def overlay(subject_image, bg_img, position, bgColor):
    subject_image = subject_image.resize(
        tuple(int(ti / 4) for ti in subject_image.size)
    )
    subject_image_bg_removed = remove(
        subject_image, alpha_matting=True, bgcolor=bgColor
    )  # remove background
    ## in proportion to A4 size
    bg_img = bg_img.resize((842, 1191))

    if position == Position.TOP_RIGHT.value:
        x = bg_img.width - subject_image_bg_removed.width
        y = 0
    elif position == Position.TOP_CENTER.value:
        x = (bg_img.width - subject_image_bg_removed.width) // 2
        y = 0
    elif position == Position.TOP_LEFT.value:
        x = 0
        y = 0
    elif position == Position.MIDDLE_RIGHT.value:
        x = bg_img.width - subject_image_bg_removed.width - 30
        y = (bg_img.height - subject_image_bg_removed.height) // 2
    elif position == Position.CENTER.value:
        x = (bg_img.width - subject_image_bg_removed.width) // 2
        y = (bg_img.height - subject_image_bg_removed.height) // 2
    elif position == Position.MIDDLE_LEFT.value:
        x = 30
        y = (bg_img.height - subject_image_bg_removed.height) // 2
    elif position == Position.BOTTOM_RIGHT.value:
        print("came in middle right")
        x = bg_img.width - subject_image_bg_removed.width
        y = bg_img.height - subject_image_bg_removed.height
    elif position == Position.BOTTOM_CENTER.value:
        x = (bg_img.width - subject_image_bg_removed.width) // 2
        y = bg_img.height - subject_image_bg_removed.height
    elif position == Position.BOTTOM_LEFT.value:
        x = 0
        y = bg_img.height - subject_image_bg_removed.height

    bg_img.paste(subject_image_bg_removed, (x, y), subject_image_bg_removed)
    return bg_img
```

## Wrapper call to instantiate required variables and call brochure generation 


```
def call_brochure_maker(
    prompt, negative_prompt, logo_image, subject_image, banner_color, sampleResultCount
):
    IMAGE_MODEL_NAME = "imagegeneration@005"
    sampleCount = int(sampleResultCount)
    negativePrompt = "faces, people, animals, food, large objects"

    # Advanced option, try different the seed numbers any random integer number range: (0, 2147483647)
    seed = None
    sampleImageStyle = None

    images = imagen_generate_sdk(
        IMAGE_MODEL_NAME,
        prompt,
        negativePrompt,
        sampleCount,
        seed,
        False,
        sampleImageStyle,
    )
    # display_images(images)

    # Get the current timestamp, # Generate a random number # Concatenate timestamp and random number to create a unique number
    current_timestamp = int(time.time())
    random_number = random.randint(1000, 9999)
    unique_number = int(str(current_timestamp) + str(random_number))
    directory_path = "results/" + str(unique_number)
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)

    results = []
    for bg_image in images:
        final_image = make_brochure(subject_image, bg_image, logo_image, banner_color)
        results.append(final_image)
    return results
```

### TEST - Call the Wrapper to call background generation, add Banner, logo and create Brochure 


```
fg_path = "./S-DRV-4D.jpg"
logo_path = "./company_logo.jpeg"
banner_color = "teal"
subject_image = Image.open(fg_path)  # load image
logo_image = Image.open(logo_path)  # load image
sample_resultcount = 3
results = call_brochure_maker(
    "beautiful underwater image of sea in the background, subtle",
    "faces",
    logo_image,
    subject_image,
    banner_color,
    sample_resultcount,
)

for res in results:
    display_image(res)
    res.save("brochure.jpg")
```


    
![png](output_33_0.png)
    



    
![png](output_33_1.png)
    



    
![png](output_33_2.png)
    


### TEST - End to End Flow


```
fg_path = "./S-DRV-4D.jpg"
logo_path = "./company_logo.jpeg"
banner_color = "teal"
subject_image = Image.open(fg_path)  # load image
logo_image = Image.open(logo_path)  # load image
sample_resultcount = 3


temperature = "Cool"  ##  warm , cool ,neutral
colors = "white,blue"
is_blurred = "No"
patterns = "waves"
style = "Minimalistic"

# generate Prompt
prompt_for_imagen = generate_imagen_prompt_using_llm(
    temperature, colors, is_blurred, patterns, style
)

# Generate Brochure
results = call_brochure_maker(
    prompt_for_imagen,
    "faces",
    logo_image,
    subject_image,
    banner_color,
    sample_resultcount,
)

for res in results:
    res.show()
    res.save("brochure.jpg")
```

    /usr/bin/xdg-open: 882: www-browser: not found
    /usr/bin/xdg-open: 882: links2: not found
    /usr/bin/xdg-open: 882: elinks: not found
    /usr/bin/xdg-open: 882: links: not found
    /usr/bin/xdg-open: 882: lynx: not found
    /usr/bin/xdg-open: 882: w3m: not found
    xdg-open: no method available for opening '/var/tmp/tmp4alom2uh.PNG'
    /usr/bin/xdg-open: 882: www-browser: not found
    /usr/bin/xdg-open: 882: links2: not found
    /usr/bin/xdg-open: 882: elinks: not found
    /usr/bin/xdg-open: 882: links: not found
    /usr/bin/xdg-open: 882: lynx: not found
    /usr/bin/xdg-open: 882: w3m: not found
    xdg-open: no method available for opening '/var/tmp/tmp05dc5bfi.PNG'
    /usr/bin/xdg-open: 882: www-browser: not found
    /usr/bin/xdg-open: 882: links2: not found
    /usr/bin/xdg-open: 882: elinks: not found
    /usr/bin/xdg-open: 882: links: not found
    /usr/bin/xdg-open: 882: lynx: not found
    /usr/bin/xdg-open: 882: w3m: not found
    xdg-open: no method available for opening '/var/tmp/tmpjlqfn9yi.PNG'
    

### TEST - Call just the brochure maker with the background image of your choice to test 

# Sample UI App fro Demo


```
banner_color_options = []
for color in ImageColor.colormap:
    banner_color_options.append(color)


def image_process_call_brochure_maker(
    background_context,
    negative_prompt,
    banner_color,
    logo_image,
    product_image,
    sample_resultcount,
):
    subject_image = PIL_Image.fromarray(product_image)
    logo_image_pil = PIL_Image.fromarray(logo_image)

    images_list_results = call_brochure_maker(
        background_context,
        negative_prompt,
        logo_image_pil,
        subject_image,
        banner_color,
        sample_resultcount,
    )
    return images_list_results


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# Brochure Maker")
    with gr.Row():
        with gr.Column(min_width=100):
            gr.Markdown("## Prompt Generator")
            temperature = gr.Dropdown(
                choices=["Warm", "Cool", "Neutral"],
                label="Temperature",
                info="Color temperature or light appearance on a cool to warm scale",
            )
            colors = gr.Textbox(
                label="Colors you want to see eg; white, blue or green, yellow"
            )
            patterns = gr.Textbox(
                label="Any patterns you want to see eg: lines, geometric shapes, leaves, circles"
            )
            style = gr.Dropdown(
                choices=["Abstract", "Vintage", "Modern", "Minimalistic"], label="Style"
            )
            is_blurred = gr.Dropdown(choices=["Yes", "No"], label="Blurred ?")
            submit_button = gr.Button("Show Prompts")
            llm_image_prompts = gr.Textbox(label="Output Prompt(s)")
        with gr.Column(min_width=100):
            gr.Markdown("## Brochure Maker")
            with gr.Row():
                with gr.Column(min_width=100):
                    background_context = gr.Textbox(
                        label="Enter your prompt for image background"
                    )
                    negative_prompt = gr.Textbox(
                        label="Anything you want to avoid seeing in the image? eg: forest, plants, car"
                    )
                    banner_color = gr.Dropdown(
                        choices=banner_color_options,
                        label="Banner color",
                        info="Will appear on the top and bottom",
                    )
                    product_image = gr.Image(label="Product Image")
                    logo_image = gr.Image(label="Logo Image")
                    sample_resultcount = gr.Textbox(
                        label="How many images you want to see? 1-4"
                    )

            with gr.Row():
                with gr.Row():
                    with gr.Column(min_width=50):
                        generate_btn = gr.Button("Generate")
                    with gr.Column(min_width=50):
                        clear = gr.ClearButton()
    with gr.Row():
        with gr.Column():
            with gr.Column():
                with gr.Row():
                    bg_gallery = gr.Gallery()

    # clear.add([bg_gallery, background_context, negative_prompt, banner_color,logo_image, product_image])
    def on_click(
        background_context,
        negative_prompt,
        banner_color,
        logo_image,
        product_image,
        sample_resultcount,
    ):
        images_list = image_process_call_brochure_maker(
            background_context=background_context,
            negative_prompt=negative_prompt,
            banner_color=banner_color,
            logo_image=logo_image,
            product_image=product_image,
            sample_resultcount=sample_resultcount,
        )
        return gr.Gallery(images_list)

    generate_btn.click(
        fn=on_click,
        inputs=[
            background_context,
            negative_prompt,
            banner_color,
            logo_image,
            product_image,
            sample_resultcount,
        ],
        outputs=[bg_gallery],
    )

    submit_button.click(
        fn=generate_imagen_prompt_using_llm,
        inputs=[temperature, colors, is_blurred, patterns, style],
        outputs=[llm_image_prompts],
    )

demo.launch(debug=True, share=True)
```

    Running on local URL:  http://127.0.0.1:7860
    Running on public URL: https://36b63a2d9264a51776.gradio.live
    
    This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
    


<div><iframe src="https://36b63a2d9264a51776.gradio.live" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>


# Conclusion

In this notebook we have successfully used Google Cloud's Generative AI capabilities to generate background images and tie them together to generate a Product Brochure. Along with that we also created a sample UI to test the pipeline end to end.
