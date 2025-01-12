##### Copyright 2024 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Create a marketing campaign from a product sketch of a Jet Backpack

This notebook contains a code example of using the Gemini API to analyze a a product sketch (in this case, a drawing of a Jet Backpack), create a marketing campaign for it, and output taglines in JSON format.

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Market_a_Jet_Backpack.ipynb"><img src="../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

## Setup


```
!pip install -q "google-generativeai>=0.7.2"
```


```
from google.colab import userdata
import google.generativeai as genai
import PIL.Image
from IPython.display import display, Image, HTML
import ipywidgets as widgets

import json
from typing_extensions import TypedDict
```

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) quickstart for an example.


```
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
```

## Marketing Campaign
- Product Name
- Description
- Feature List / Descriptions
- H1
- H2



```
model = genai.GenerativeModel(model_name='gemini-1.5-flash')
```

## Analyze Product Sketch

First you will download a sample image to be used:


```
productSketchUrl = "https://storage.googleapis.com/generativeai-downloads/images/jetpack.jpg"
!curl -o jetpack.jpg {productSketchUrl}
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0100  349k  100  349k    0     0  2141k      0 --:--:-- --:--:-- --:--:-- 2155k
    

You can view the sample image to understand the prompts you are going to work with:


```
img = PIL.Image.open('jetpack.jpg')
display(Image('jetpack.jpg', width=300))
```


    
![jpeg](output_14_0.jpg)
    


Now define a prompt to analyze the sample image:


```
analyzePrompt = """This image contains a sketch of a potential product along with some notes.
Given the product sketch, describe the product as thoroughly as possible based on what you
see in the image, making sure to note all of the product features.

Return output in json format."""
```

- Set the model to return JSON by setting `response_mime_type="application/json"`.
- Describe the schema for the response using a `TypedDict`.


```
class Response(TypedDict):
  description: str
  features: list[str]
```


```
response = model.generate_content(
    [analyzePrompt, img],
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=Response))
```


```
productInfo = json.loads(response.text)

print(json.dumps(productInfo, indent=4))
```

    {
        "description": "The Jetpack Backpack is a backpack that looks like a normal backpack but has retractable boosters that allow the user to fly. It has a 15-minute battery life and can be charged using a USB-C port. It is lightweight and can fit a 18\" laptop.",
        "features": [
            "retractable boosters",
            "15-min battery life",
            "USB-C charging",
            "lightweight",
            "fits 18\" laptop",
            "steam-powered",
            "green/clean",
            "padded strap support"
        ]
    }
    

> Note: Here the model is just following text instructions for how the output json should be formatted. The API also supports a **strict JSON mode** where you specify a schema, and the API uses "Controlled Generation" (aka "Constrained Decoding") to ensure the model follows the schema, see the [JSON mode quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/JSON_mode.ipynb) for details.

## Generate marketing ideas

Now using the image you can use Gemini API to generate marketing names ideas:


```
namePrompt = """You are a marketing whiz and writer trying to come up with a name for the
product shown in the image. Come up with ten varied, interesting possible names."""

response = model.generate_content(
    [namePrompt, img],
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=list[str]))

names = json.loads(response.text)
# Create a Dropdown widget to choose a name from the
# returned possible names
dropdown = widgets.Dropdown(
    options=names,
    value=names[0],  # default value
    description='Name:',
    disabled=False,
)
display(dropdown)
```


    Dropdown(description='Name:', options=('Jetpack', 'FlytePack', 'The Booster', 'SkyBackpack', 'The Commute Jet'…


Finally you can work on generating a page for your product campaign:


```
name = dropdown.value
```


```
websiteCopyPrompt = f"""You're a marketing whiz and expert copywriter. You're writing
website copy for a product named {name}. Your first job is to come up with H1
H2 copy. These are brief, pithy sentences or phrases that are the first and second
things the customer sees when they land on the splash page. Here are some examples:
[{{
  "h1": "A feeling is canned",
  "h2": "drinks and powders to help you feel calm cool and collected\
   despite the stressful world around you"
}},
{{
  "h1": "Design. Publish. Done.",
  "h2": "Stop rebuilding your designs from scratch. In Framer, everything\
   you put on the canvas is ready to be published to the web."
}}]

Create the same json output for a product named "{name}" with description\
 "{productInfo['description']}".
Output ten different options as json in an array.
"""
```




    'You\'re a marketing whiz and expert copywriter. You\'re writing\nwebsite copy for a product named Jetpack. Your first job is to come up with H1\nH2 copy. These are brief, pithy sentences or phrases that are the first and second\nthings the customer sees when they land on the splash page. Here are some examples:\n[{\n  "h1": "A feeling is canned",\n  "h2": "drinks and powders to help you feel calm cool and collected   despite the stressful world around you"\n},\n{\n  "h1": "Design. Publish. Done.",\n  "h2": "Stop rebuilding your designs from scratch. In Framer, everything   you put on the canvas is ready to be published to the web."\n}]\n\nCreate the same json output for a product named "Jetpack" with description "The Jetpack Backpack is a backpack that looks like a normal backpack but has retractable boosters that allow the user to fly. It has a 15-minute battery life and can be charged using a USB-C port. It is lightweight and can fit a 18" laptop.".\nOutput ten different options as json in an array.\n'




```
class Headings(TypedDict):
  h1:str
  h2:str
```


```
copyResponse = model.generate_content(
    [websiteCopyPrompt, img],
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=list[Headings]))
```


```
copy = json.loads(copyResponse.text)
```


```
copy
```




    [{'h1': 'Fly High.  Fly Green.',
      'h2': "The Jetpack Backpack is a revolutionary new backpack that lets you take to the skies.  It's powered by steam, so it's eco-friendly and sustainable."},
     {'h1': 'The Future of Commuting is Here.',
      'h2': 'The Jetpack Backpack is a lightweight backpack with retractable boosters that make it easy to get around town.'},
     {'h1': 'Jetpack Backpack:  Your Everyday Escape.',
      'h2': 'With a 15-minute battery life and a USB-C charging port, the Jetpack Backpack is perfect for short trips or commutes.'},
     {'h1': 'Get Above the Traffic.',
      'h2': 'The Jetpack Backpack is the perfect way to avoid traffic jams and get to your destination quickly.'},
     {'h1': 'Go Anywhere, Do Anything.',
      'h2': 'The Jetpack Backpack is designed to look like a normal backpack, so you can take it anywhere without drawing attention.'},
     {'h1': 'The Jetpack Backpack:  Adventure Awaits.',
      'h2': 'The Jetpack Backpack is the perfect way to explore new places and see the world from a different perspective.'},
     {'h1': 'Experience the Thrill of Flight.',
      'h2': "The Jetpack Backpack is a safe and easy way to experience the thrill of flight.  It's perfect for beginners and experienced flyers alike."},
     {'h1': 'The Jetpack Backpack:  The Future is Here.',
      'h2': "The Jetpack Backpack is a revolutionary new product that's changing the way people travel.  It's the perfect combination of technology and sustainability."},
     {'h1': 'Get Ready to Take Off.',
      'h2': "The Jetpack Backpack is the perfect way to add a little excitement to your daily routine.  It's easy to use and safe for everyone."},
     {'h1': 'Break Free from the Ordinary.',
      'h2': "The Jetpack Backpack is the perfect way to stand out from the crowd.  It's a unique and stylish way to get around."}]




```
h1 = copy[2]['h1']
h2 = copy[2]['h2']
```


```
htmlPrompt = f"""Generate HTML and CSS for a splash page for a new product called {name}.
Output only HTML and CSS and do not link to any external resources.
Include the top level title: "{h1}" with the subtitle: "{h2}".

Return the HTML directly, do not wrap it in triple-back-ticks (```).
"""
```


```
response = model.generate_content([htmlPrompt])
print(response.text)
```

    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Jetpack Backpack</title>
      <style>
        body {
          font-family: sans-serif;
          margin: 0;
          display: flex;
          justify-content: center;
          align-items: center;
          min-height: 100vh;
          background-color: #f0f0f0;
        }
    
        .container {
          background-color: white;
          padding: 50px;
          border-radius: 10px;
          box-shadow: 0 4px 8px rgba(0,0,0,0.1);
          text-align: center;
        }
    
        h1 {
          font-size: 3em;
          margin-bottom: 10px;
        }
    
        h2 {
          font-size: 1.5em;
          margin-bottom: 20px;
          color: #555;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Jetpack Backpack: Your Everyday Escape</h1>
        <h2>With a 15-minute battery life and a USB-C charging port, the Jetpack Backpack is perfect for short trips or commutes.</h2>
      </div>
    </body>
    </html>
    
    


```
HTML(response.text)
```




<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Jetpack Backpack</title>
  <style>
    body {
      font-family: sans-serif;
      margin: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      background-color: #f0f0f0;
    }

    .container {
      background-color: white;
      padding: 50px;
      border-radius: 10px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.1);
      text-align: center;
    }

    h1 {
      font-size: 3em;
      margin-bottom: 10px;
    }

    h2 {
      font-size: 1.5em;
      margin-bottom: 20px;
      color: #555;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Jetpack Backpack: Your Everyday Escape</h1>
    <h2>With a 15-minute battery life and a USB-C charging port, the Jetpack Backpack is perfect for short trips or commutes.</h2>
  </div>
</body>
</html>



