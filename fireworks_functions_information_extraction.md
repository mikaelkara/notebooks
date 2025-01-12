[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SI6jz66k122vv641e8wDDI0Ujh4cwlUy?usp=sharing)

# Summarize Anything - Information Extraction via [Fireworks Function Calling](https://readme.fireworks.ai/docs/function-calling)

This is inspired by awesome colab notebook by [Deepset](https://colab.research.google.com/github/anakin87/notebooks/blob/main/information_extraction_via_llms.ipynb). Check out there OSS LLM Orchestration framework [haystack](https://haystack.deepset.ai/).

In this experiment, we will use function calling ability of [Fireworks Function Calling](https://readme.fireworks.ai/docs/function-calling) model to generate structured information from unstructured data.

🎯 Goal: create an application that, given a text (or URL) and a specific structure provided by the user, extracts information from the source.


The "**function calling**" capability first launched by [OpenAI](https://platform.openai.com/docs/guides/function-calling) unlocks this task: the user can describe a structure, by defining a fake function with all its typed and specific parameters. The LLM will prepare the data in this specific form and send it back to the user.

**Fireworks Function Calling**

Fireworks released a high quality function calling model which is capable of handling long tool context, multi turn conversations & interleaving tool invocations ith regular conversation. We are going to use this model today as our LLM to power our app.



>[Summarize Anything - Information Extraction via Fireworks Function Calling](#scrollTo=8Ksv005GbN2w)

>>[Introduction](#scrollTo=cp4hJ34JivkB)

>[Document Retrieval & Clean Up](#scrollTo=buM6rGqMwLZ4)

>>>[Let's learn about Capybara](#scrollTo=0kVJ8IfSI-Dx)

>>>[How about Yucatan Deer](#scrollTo=Tzz1LSS-JBk4)

>>>[Something more massive - African Elephant](#scrollTo=B0M4NEm9JMAw)

>>[Let's make example fun - News Summarization](#scrollTo=x7Y8_xmxDOKx)



# 🚀 Running This Notebook

This notebook is designed to be run in **Google Colab** for the best experience. If you prefer to run it locally, please follow the setup instructions below.

## 🛠️ Running Locally:

1. **Set up a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   .\venv\Scripts\activate  # On Windows
   ```

2. **Install the required libraries:**
   ```bash
   pip install openai torch requests beautifulsoup4 ipython
   ```

3. **Configure your API key:**
   Ensure you have a Fireworks API key and set it as an environment variable:
   ```bash
   export FW_API_KEY=your_api_key  # On macOS/Linux
   set FW_API_KEY=your_api_key  # On Windows
   ```

4. **Launch the Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

5. **Open the notebook file** and proceed with the cells as usual.

> **Note:** If you encounter any issues with dependencies, make sure your Python version is 3.8 or higher.

## Setup
Let's install the dependencies needed for the demo first and import any dependencies needed.


```python
!pip3 install openai bs4
```

    Requirement already satisfied: openai in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (1.47.0)
    Requirement already satisfied: bs4 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (0.0.2)
    Requirement already satisfied: anyio<5,>=3.5.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (4.6.0)
    Requirement already satisfied: distro<2,>=1.7.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (1.9.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (0.27.2)
    Requirement already satisfied: jiter<1,>=0.4.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (0.4.2)
    Requirement already satisfied: pydantic<3,>=1.9.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (2.9.2)
    Requirement already satisfied: sniffio in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (1.3.1)
    Requirement already satisfied: tqdm>4 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (4.66.5)
    Requirement already satisfied: typing-extensions<5,>=4.11 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (4.12.2)
    Requirement already satisfied: beautifulsoup4 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from bs4) (4.12.3)
    Requirement already satisfied: idna>=2.8 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from anyio<5,>=3.5.0->openai) (3.10)
    Requirement already satisfied: certifi in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpx<1,>=0.23.0->openai) (2024.8.30)
    Requirement already satisfied: httpcore==1.* in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpx<1,>=0.23.0->openai) (1.0.5)
    Requirement already satisfied: h11<0.15,>=0.13 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)
    Requirement already satisfied: annotated-types>=0.6.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic<3,>=1.9.0->openai) (0.7.0)
    Requirement already satisfied: pydantic-core==2.23.4 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic<3,>=1.9.0->openai) (2.23.4)
    Requirement already satisfied: soupsieve>1.2 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from beautifulsoup4->bs4) (2.6)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.2[0m[39;49m -> [0m[32;49m24.3.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip3 install --upgrade pip[0m
    


```python
import torch
import json
from typing import Dict
import os
import openai
from IPython.display import HTML, display
```

## Setup your API Key

In order to use the Fireworks AI function calling model, you must first obtain Fireworks API Keys. If you don't already have one, you can one by following the instructions [here](https://readme.fireworks.ai/docs/quickstart).


```python
model_name = "accounts/fireworks/models/firefunction-v2"
client = client = openai.OpenAI(
    base_url = "https://api.fireworks.ai/inference/v1",
    api_key = "YOUR_FIREWORKS_API_KEY",
)
```

## Introduction

The [documentation](https://readme.fireworks.ai/docs/function-calling) for FW function calling details the API we can use to specify the list of tools/functions available to the model. We will use the described API to test out the structured response usecase.

Before we can begin, let's give the function calling model a go with a simple toy example and examine it's output.


```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "uber.ride",
            "description": "Find a suitable ride for customers given the location, type of ride, and the amount of time the customer is willing to wait.",
            "parameters": {
                "type": "object",
                "properties": {
                    "loc": {
                        "type": "string",
                        "description": "Location of the starting place of the Uber ride."
                    },
                    "type": {
                        "type": "string",
                        "enum": ["plus", "comfort", "black"],
                        "description": "Type of Uber ride the user is ordering (e.g., plus, comfort, or black)."
                    },
                    "time": {
                        "type": "string",
                        "description": "The amount of time in minutes the customer is willing to wait."
                    }
                },
                "required": ["loc", "type", "time"]
            }
        }
    }
]

tool_choice = "auto"
user_prompt = 'Call me an Uber ride type "Plus" in Berkeley at zipcode 94704 in 10 minutes.'

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant with access to tools. Use them wisely and don't imagine parameter values.",
    },
    {
        "role": "user",
        "content": user_prompt,
    }
]

```


```python
chat_completion = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    tool_choice=tool_choice,
    temperature=0.1
)
```


```python
print(chat_completion.choices[0].message.model_dump_json(indent=4))
```

    {
        "content": null,
        "refusal": null,
        "role": "assistant",
        "function_call": null,
        "tool_calls": [
            {
                "id": "call_IU1yyD8YCD4PLpwdAdnHlM8D",
                "function": {
                    "arguments": "{\"loc\": \"Berkeley, 94704\", \"type\": \"plus\", \"time\": \"10\"}",
                    "name": "uber.ride"
                },
                "type": "function",
                "index": 0
            }
        ]
    }
    

The model outputs the function that should be called along with arguments under the `tool_calls` field. This field contains the arguments to be used for calling the function as JSON Schema and the `name` field contains the name of the function to be called.


The output demonstrates a sample input & output to function calling model. All good! ✅

## Document Retrieval & Clean Up

Before we can get started with extracting the right set of information. We need to first obtaint the document given a url & then clean it up. For cleaning up HTML, we will use [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/).


```python
import requests
from bs4 import BeautifulSoup
```


```python
url = "https://www.rainforest-alliance.org/species/capybara/"

# Page content from Website URL
page = requests.get(url)

# Function to remove tags
def remove_tags(html):

    # parse html content
    soup = BeautifulSoup(html, "html.parser")

    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()
    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)


# Print the extracted data
cleaned_content = remove_tags(page.content)
```

## Setup Information Extraction using Function Calling

After we have obtained clean data from a html page given a url, we are going to send this data to function calling model. Along with sending the cleaned html, we are also going to send it the schema in which we expect the model to produce output. This schema is sent under the tool specification of chat completion call.

For this notebook, we use the `info_tools` schema to extract information from species info pages of [Rain Forest Alliance](https://www.rainforest-alliance.org/). To make the task harder - we include another schema for extracting new information from news reports. There are several attributes about the animal we want the model to extract from the web page e.g. `weight`, `habitat`, `diet` etc. Additionally, we specify some attributes as `required` forcing the model to always output this information regardless of the input. Given, we would be supplying the model with species information pages, we expect this information to be always present.

**NOTE** We set the temperature to 0.0 to get reliable and consistent output across calls. In this particular example, we want the model to produce the right answer rather than creative answer.


```python
from typing import Dict, List, Any

def extract_data(tools: List[Dict[str, Any]], url: str) -> str:
  tool_choice = {
      "type": "function"
  }
  page = requests.get(url)
  cleaned_content = remove_tags(page.content)

  messages = [
      {
          "role": "system",
          "content": f"You are a helpful assistant with access to tools. Use them wisely and don't imageine parameter values."
      },
      {
          "role": "user",
          "content": f"Extract data from the following text. START TEXT {cleaned_content} END TEXT."
      }
  ]

  chat_completion = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    tool_choice=tool_choice,
    temperature=0.0
  )

  def val_to_color(val):
    """
    Helper function to return a color based on the type/value of a variable
    """
    if isinstance(val, list):
      return "#FFFEE0"
    if val is True:
      return "#90EE90"
    if val is False:
      return "#FFCCCB"
    return ""

  args = json.loads(chat_completion.choices[0].message.tool_calls[0].function.arguments)

  # Convert data to HTML format
  html_content = '<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">'
  for key, value in args.items():
      html_content += f'<p><span style="font-family: Cursive; font-size: 30px;">{key}:</span>'
      html_content += f'&emsp;<span style="background-color:{val_to_color(value)}; font-family: Cursive; font-size: 20px;">{value}</span></p>'
  html_content += '</div>'

  return {"html_visualization": html_content}
```

### Tool Schema

Below we specify 2 kinds of schema for information extraction. We want the model to decide which schema best fits the situation and extract information accordingly. First we will try extracting information for animal from given text and then we'll try news articles.


```python
info_tools = [
{
  "type": "function",
  "function": {
      "name": "extract_animal_info",
      "description": "Extract animal info from the text",
      "parameters": {
          "type": "object",
          "properties": {
              "about_animals": {
                  "description": "Is the article about animals?",
                  "type": "boolean",
              },
              "about_ai": {
                  "description": "Is the article about artificial intelligence?",
                  "type": "boolean",
              },
              "weight": {
                  "description": "the weight of the animal in lbs",
                  "type": "integer",
              },
              "habitat": {
                  "description": "List of places where the animal lives",
                  "type": "array",
                  "items": {"type": "string"},
              },
              "diet": {
                  "description": "What does the animal eat?",
                  "type": "array",
                  "items": {"type": "string"},
              },
              "predators": {
                  "description": "What are the animals that threaten them?",
                  "type": "array",
                  "items": {"type": "string"},
              },
          },
          "required": ["about_animals", "about_ai", "weight", "habitat", "diet", "predators"],
      }
  }
},
{
  "type": "function",
  "function": {
      "name": "extract_news_info",
      "description": "Extract news info from the text",
      "parameters": {
          "type": "object",
          "properties": {
              "about_ai": {
                  "description": "Is the article about artificial intelligence?",
                  "type": "boolean",
              },
              "company_name": {
                  "description": "The name of the company which is being referenced in document",
                  "type": "string",
              },
              "valuation": {
                  "description": "Valuation of the company which is being referenced in document",
                  "type": "string",
              },
              "investors": {
                  "description": "investors in the company being referenced in document",
                  "type": "array",
                  "items": {"type": "string"},
              },
              "competitors": {
                  "description": "competitors of the company being referenced in document",
                  "type": "array",
                  "items": {"type": "string"},
              },
          },
          "required": ["about_ai", "company_name", "valuation", "investors", "competitors"],
      }
  }
}
]
```

### Let's learn about Capybara

Given the schema, we expect the model to produce some basic information like `weight`, `habitat`, `diet` & `predators` for Capybara. You can visit the [webpage](https://www.rainforest-alliance.org/species/capybara/) to see the source of the truth.


```python
display(HTML(extract_data(info_tools, url="https://www.rainforest-alliance.org/species/capybara/")['html_visualization']))
```


<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9;"><p><span style="font-family: Cursive; font-size: 30px;">about_animals:</span>&emsp;<span style="background-color:#90EE90; font-family: Cursive; font-size: 20px;">True</span></p><p><span style="font-family: Cursive; font-size: 30px;">about_ai:</span>&emsp;<span style="background-color:#FFCCCB; font-family: Cursive; font-size: 20px;">False</span></p><p><span style="font-family: Cursive; font-size: 30px;">weight:</span>&emsp;<span style="background-color:; font-family: Cursive; font-size: 20px;">100</span></p><p><span style="font-family: Cursive; font-size: 30px;">habitat:</span>&emsp;<span style="background-color:#FFFEE0; font-family: Cursive; font-size: 20px;">['Panama', 'Colombia', 'Venezuela', 'Guyana', 'Peru', 'Brazil', 'Paraguay', 'Northeast Argentina', 'Uruguay']</span></p><p><span style="font-family: Cursive; font-size: 30px;">diet:</span>&emsp;<span style="background-color:#FFFEE0; font-family: Cursive; font-size: 20px;">['vegetation', 'grains', 'melons', 'reeds', 'squashes']</span></p><p><span style="font-family: Cursive; font-size: 30px;">predators:</span>&emsp;<span style="background-color:#FFFEE0; font-family: Cursive; font-size: 20px;">['jaguars', 'caimans', 'anacondas', 'ocelots', 'harpy eagles']</span></p></div>


You can see the model correctly identifies the correct weight - `100 lbs` for the Capybara even though the webpage mentions the weight in `kgs` too. It also identifies the correct habitat etc. for the animal.  

### How about Yucatan Deer


```python
display(HTML(extract_data(info_tools, url="https://www.rainforest-alliance.org/species/yucatan-deer/")['html_visualization']))
```


<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9;"><p><span style="font-family: Cursive; font-size: 30px;">about_animals:</span>&emsp;<span style="background-color:#90EE90; font-family: Cursive; font-size: 20px;">True</span></p><p><span style="font-family: Cursive; font-size: 30px;">about_ai:</span>&emsp;<span style="background-color:#FFCCCB; font-family: Cursive; font-size: 20px;">False</span></p><p><span style="font-family: Cursive; font-size: 30px;">weight:</span>&emsp;<span style="background-color:; font-family: Cursive; font-size: 20px;">70</span></p><p><span style="font-family: Cursive; font-size: 30px;">habitat:</span>&emsp;<span style="background-color:#FFFEE0; font-family: Cursive; font-size: 20px;">['Central America', 'Mexico', 'South America']</span></p><p><span style="font-family: Cursive; font-size: 30px;">diet:</span>&emsp;<span style="background-color:#FFFEE0; font-family: Cursive; font-size: 20px;">['grass', 'leaves', 'sprouts', 'lichens', 'mosses', 'tree bark', 'fruit']</span></p><p><span style="font-family: Cursive; font-size: 30px;">predators:</span>&emsp;<span style="background-color:#FFFEE0; font-family: Cursive; font-size: 20px;">['cougar', 'jaguar', 'ticks', 'horseflies', 'mosquitoes']</span></p></div>


### Something more massive - African Elephant


```python
display(HTML(extract_data(info_tools, url="https://www.rainforest-alliance.org/species/african-elephants/")['html_visualization']))
```


<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9;"><p><span style="font-family: Cursive; font-size: 30px;">about_animals:</span>&emsp;<span style="background-color:#90EE90; font-family: Cursive; font-size: 20px;">True</span></p><p><span style="font-family: Cursive; font-size: 30px;">about_ai:</span>&emsp;<span style="background-color:#FFCCCB; font-family: Cursive; font-size: 20px;">False</span></p><p><span style="font-family: Cursive; font-size: 30px;">weight:</span>&emsp;<span style="background-color:; font-family: Cursive; font-size: 20px;">5000</span></p><p><span style="font-family: Cursive; font-size: 30px;">habitat:</span>&emsp;<span style="background-color:#FFFEE0; font-family: Cursive; font-size: 20px;">['central and western Africa’s equatorial forests', 'grassy plains and bushlands of the continent']</span></p><p><span style="font-family: Cursive; font-size: 30px;">diet:</span>&emsp;<span style="background-color:#FFFEE0; font-family: Cursive; font-size: 20px;">['leaves and branches', 'grasses', 'fruit', 'bark']</span></p><p><span style="font-family: Cursive; font-size: 30px;">predators:</span>&emsp;<span style="background-color:#FFFEE0; font-family: Cursive; font-size: 20px;">['humans']</span></p></div>


## Let's make example fun - News Summarization

Now let's use a more fun example. In order for LLMs to leverage the world knowledge, they need to be able to organize unstructured sources like websites into more structured information. Let's take the example of a news article announcing the new funding round for the startup [Perplexity AI](https://www.perplexity.ai/). For our sample news summarization app, the user only specifies the small list of information that want from the article and then ask the LLM to generate the needed information for them.


```python
display(HTML(extract_data(info_tools, url="https://techcrunch.com/2024/01/04/ai-powered-search-engine-perplexity-ai-now-valued-at-520m-raises-70m")['html_visualization']))
```


<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9;"><p><span style="font-family: Cursive; font-size: 30px;">about_ai:</span>&emsp;<span style="background-color:#90EE90; font-family: Cursive; font-size: 20px;">True</span></p><p><span style="font-family: Cursive; font-size: 30px;">company_name:</span>&emsp;<span style="background-color:; font-family: Cursive; font-size: 20px;">Perplexity AI</span></p><p><span style="font-family: Cursive; font-size: 30px;">valuation:</span>&emsp;<span style="background-color:; font-family: Cursive; font-size: 20px;">520M</span></p><p><span style="font-family: Cursive; font-size: 30px;">investors:</span>&emsp;<span style="background-color:#FFFEE0; font-family: Cursive; font-size: 20px;">['IVP', 'NEA', 'Databricks Ventures', 'Elad Gil', 'Tobi Lutke', 'Nat Friedman', 'Guillermo Rauch', 'Nvidia', 'Jeff Bezos']</span></p><p><span style="font-family: Cursive; font-size: 30px;">competitors:</span>&emsp;<span style="background-color:#FFFEE0; font-family: Cursive; font-size: 20px;">['Google', 'Microsoft', 'You.com']</span></p></div>



