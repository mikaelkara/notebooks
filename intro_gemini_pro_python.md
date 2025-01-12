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

# Getting Started with the Gemini Pro Model

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_pro_python.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fgetting-started%2Fintro_gemini_pro_python.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_pro_python.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/getting-started/intro_gemini_pro_python.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Eric Dong](https://github.com/gericdong), [Polong Lin](https://github.com/polong-lin), [Wanheng Li](https://github.com/wanhengli) |

## Overview

### Gemini

Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases. The Gemini API gives you access to the Gemini Pro and Gemini Pro Vision models.

### Gemini API in Vertex AI

The Gemini API in Vertex AI provides a unified interface for interacting with Gemini models. There are two Gemini 1.0 Pro models available in the Gemini API:

- **Gemini 1.0 Pro model** (`gemini-1.0-pro`): Designed to handle natural language tasks, multi-turn text and code chat, and code generation.
- **Gemini 1.0 Pro Vision model** (`gemini-1.0-pro-vision`): Supports multimodal prompts. You can include text, images, and video in your prompt requests and get text or code responses.

You can interact with the Gemini API using the following methods:

- Use [Vertex AI Studio](https://cloud.google.com/generative-ai-studio) for quick testing and command generation
- Use cURL commands
- Use the Vertex AI SDK

This notebook focuses on using the **Vertex AI SDK for Python** to call the Gemini API in Vertex AI with the Gemini 1.0 Pro model.

For more information, see the [Generative AI on Vertex AI](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview) documentation.


### Objectives

In this tutorial, you will learn how to use the Gemini API in Vertex AI with the Vertex AI SDK for Python to interact with the Gemini 1.0 Pro (`gemini-1.0-pro`) model.

You will complete the following tasks:

- Install the Vertex AI SDK for Python
- Use the Gemini API in Vertex AI to interact with Gemini 1.0 Pro (`gemini-1.0-pro`) model:
    - Generate text from text prompts
    - Explore various features and configuration options


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

### Define Google Cloud project information and initialize Vertex AI

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
from vertexai.generative_models import GenerationConfig, GenerativeModel
```

## Use the Gemini 1.0 Pro model

The Gemini 1.0 Pro (`gemini-1.0-pro`) model is designed to handle natural language tasks, multi-turn text and code chat, and code generation.


### Load the Gemini 1.0 Pro model



```
model = GenerativeModel("gemini-1.0-pro")
```

### Generate text from text prompts

Send a text prompt to the model. The Gemini 1.0 Pro (`gemini-1.0-pro`) model provides a streaming response mechanism. With this approach, you don't need to wait for the complete response; you can start processing fragments as soon as they're accessible.



```
responses = model.generate_content("Why is the sky blue?", stream=True)

for response in responses:
    print(response.text, end="")
```

    The sky is blue because of a phenomenon known as Rayleigh scattering. Light from the sun is composed of a mixture of all colors, but the shorter wavelengths, such as blue and violet, are more readily scattered by molecules in the Earth's atmosphere. This scattering is caused by the interaction of the light waves with the molecules, which have a size that is comparable to the wavelength of the light.
    
    When light interacts with a molecule, the electrons in the molecule are excited and the molecule vibrates. The excited electrons then re-emit the light in all directions, but most of the light is emitted in the direction of the original wave. The amount of scattering depends on the wavelength of the light, and the shorter the wavelength, the more it is scattered. This means that blue and violet light are scattered more than other colors, which is why the sky appears blue. 
    
    At sunset and sunrise, the sunlight has to pass through more of the atmosphere to reach our eyes. This means that more of the blue and violet light is scattered away, and the light that reaches us is more concentrated in the longer wavelengths, such as red and orange. This is why the sky appears red or orange at these times.

#### Try your own prompts

- What are the biggest challenges facing the healthcare industry?
- What are the latest developments in the automotive industry?
- What are the biggest opportunities in retail industry?
- (Try your own prompts!)



```
prompt = """Create a numbered list of 10 items. Each item in the list should be a trend in the tech industry.

Each trend should be less than 5 words."""  # try your own prompt

responses = model.generate_content(prompt, stream=True)

for response in responses:
    print(response.text, end="")
```

    1. Artificial Intelligence
    2. Cloud Computing
    3. Cybersecurity
    4. 5G Networks
    5. IoT (Internet of Things)
    6. Blockchain
    7. Augmented Reality
    8. Machine Learning
    9. Quantum Computing
    10. Robotics

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

responses = model.generate_content(
    "Why is the sky blue?",
    generation_config=generation_config,
    stream=True,
)

for response in responses:
    print(response.text, end="")
```

    The sky appears blue because of a phenomenon called Rayleigh scattering.
    
    White light from the sun is composed of all colors of the visible spectrum. As this light passes through the Earth's atmosphere, it encounters molecules of nitrogen, oxygen, and other gases. These molecules are much smaller than the wavelength of visible light, so they scatter the light in all directions.
    
    However, the amount of scattering depends on the wavelength of the light. Shorter wavelengths, such as blue and violet, are scattered more strongly than longer wavelengths, such as red and orange. This is because the smaller molecules of the atmosphere are more effective at scattering shorter wavelengths.
    
    As a result, the scattered light that reaches our eyes is predominantly blue and violet. This is what gives the sky its characteristic blue appearance.
    
    At sunset and sunrise, the sunlight has to travel through more of the atmosphere to reach our eyes. This allows more of the blue and violet light to be scattered away, leaving the remaining light to appear red, orange, or yellow.

### Test chat prompts

The Gemini 1.0 Pro model supports natural multi-turn conversations and is ideal for text tasks that require back-and-forth interactions. The following examples show how the model responds during a multi-turn conversation.



```
chat = model.start_chat()

prompt = """My name is Ned. You are my personal assistant. My favorite movies are Lord of the Rings and Hobbit.

Suggest another movie I might like.
"""

responses = chat.send_message(prompt, stream=True)

for response in responses:
    print(response.text, end="")
```

    **Fantasy Adventures:**
    
    * **Game of Thrones** (TV series): A complex and gripping fantasy epic with political intrigue, warfare, and mythical creatures.
    * **The Princess Bride** (1987): A charming and whimsical adventure with a fairy-tale setting and memorable characters.
    * **The Chronicles of Narnia: The Lion, the Witch and the Wardrobe** (2005): A magical and heartwarming adaptation of C.S. Lewis's classic novel.
    * **How to Train Your Dragon** (2010): An animated adventure about a young Viking who befriends a dragon and fights alongside him.
    * **Eragon** (2006): A fantasy epic based on Christopher Paolini's novel, featuring a young farm boy who discovers his destiny as a dragon rider.
    
    **Epic Historical Dramas:**
    
    * **Braveheart** (1995): A rousing and historically accurate tale of Scottish resistance against English rule.
    * **Gladiator** (2000): A brutal and immersive depiction of ancient Rome, with stunning battle scenes and a powerful performance from Russell Crowe.
    * **Kingdom of Heaven** (2005): A visually stunning and emotionally resonant historical epic set during the Crusades.
    * **The Last Samurai** (2003): A captivating exploration of cultural clashes and the power of redemption, set in 19th-century Japan.
    * **Troy** (2004): A stirring and action-packed adaptation of Homer's epic poem, featuring a star-studded cast and breathtaking battles.

This follow-up prompt shows how the model responds based on the previous prompt:



```
prompt = "Are my favorite movies based on a book series?"

responses = chat.send_message(prompt, stream=True)

for response in responses:
    print(response.text, end="")
```

    Yes, both **The Lord of the Rings** and **The Hobbit** are based on book series by J.R.R. Tolkien.
    
    * **The Lord of the Rings** is a trilogy of epic fantasy novels published between 1954 and 1955. It tells the story of the quest of the Fellowship of the Ring to destroy the One Ring, an evil artifact created by the Dark Lord Sauron.
    * **The Hobbit** is a prequel to The Lord of the Rings, published in 1937. It follows the adventures of Bilbo Baggins, a hobbit who joins a group of dwarves on a quest to reclaim their lost kingdom from the dragon Smaug.
    
    Both book series have been adapted into highly successful film trilogies directed by Peter Jackson. The Lord of the Rings film trilogy was released between 2001 and 2003, and The Hobbit film trilogy was released between 2012 and 2014.

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
      text: "**Fantasy Adventures:**\n\n* **Game of Thrones** (TV series): A complex and gripping fantasy epic with political intrigue, warfare, and mythical creatures.\n* **The Princess Bride** (1987): A charming and whimsical adventure with a fairy-tale setting and memorable characters.\n* **The Chronicles of Narnia: The Lion, the Witch and the Wardrobe** (2005): A magical and heartwarming adaptation of C.S. Lewis\'s classic novel.\n* **How to Train Your Dragon** (2010): An animated adventure about a young Viking who befriends a dragon and fights alongside him.\n* **Eragon** (2006): A fantasy epic based on Christopher Paolini\'s novel, featuring a young farm boy who discovers his destiny as a dragon rider.\n\n**Epic Historical Dramas:**\n\n* **Braveheart** (1995): A rousing and historically accurate tale of Scottish resistance against English rule.\n* **Gladiator** (2000): A brutal and immersive depiction of ancient Rome, with stunning battle scenes and a powerful performance from Russell Crowe.\n* **Kingdom of Heaven** (2005): A visually stunning and emotionally resonant historical epic set during the Crusades.\n* **The Last Samurai** (2003): A captivating exploration of cultural clashes and the power of redemption, set in 19th-century Japan.\n* **Troy** (2004): A stirring and action-packed adaptation of Homer\'s epic poem, featuring a star-studded cast and breathtaking battles."
    }
    , role: "user"
    parts {
      text: "Are my favorite movies based on a book series?"
    }
    , role: "model"
    parts {
      text: "Yes, both **The Lord of the Rings** and **The Hobbit** are based on book series by J.R.R. Tolkien.\n\n* **The Lord of the Rings** is a trilogy of epic fantasy novels published between 1954 and 1955. It tells the story of the quest of the Fellowship of the Ring to destroy the One Ring, an evil artifact created by the Dark Lord Sauron.\n* **The Hobbit** is a prequel to The Lord of the Rings, published in 1937. It follows the adventures of Bilbo Baggins, a hobbit who joins a group of dwarves on a quest to reclaim their lost kingdom from the dragon Smaug.\n\nBoth book series have been adapted into highly successful film trilogies directed by Peter Jackson. The Lord of the Rings film trilogy was released between 2001 and 2003, and The Hobbit film trilogy was released between 2012 and 2014."
    }
    ]
    
