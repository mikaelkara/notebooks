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

# Working with Parallel Function Calls and Multiple Function Responses in Gemini

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/parallel_function_calling.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Ffunction-calling%2Fparallel_function_calling.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/function-calling/parallel_function_calling.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/parallel_function_calling.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Kristopher Overholt](https://github.com/koverholt) |

## Overview

Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases. The Gemini API gives you access to the Gemini Pro and Gemini Pro Vision models.

[Function Calling](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/function-calling) in Gemini lets you create a description of a function in your code, then pass that description to a language model in a request. The response from the model includes the name of a function that matches the description and the arguments to call it with.

In this tutorial, you'll learn how to work with parallel function calling within Gemini Function Calling, including:
    
- Handling parallel function calls for repeated functions
- Working with parallel function calls across multiple functions
- Extracting multiple function calls from a Gemini response
- Calling multiple functions and returning them to Gemini

### What is parallel function calling?

In previous versions of the Gemini Pro models (prior to May 2024), Gemini would return two or more chained function calls if the model determined that more than one function call was needed before returning a natural language summary. Here, a chained function call means that you get the first function call response, return the API data to Gemini, get a second function call response, return the API data to Gemini, and so on.

In recent versions of specific Gemini Pro models (from May 2024 and on), Gemini has the ability to return two or more function calls in parallel (i.e., two or more function call responses within the first function call response object). Parallel function calling allows you to fan out and parallelize your API calls or other actions that you perform in your application code, so you don't have to work through each function call response and return one-by-one! Refer to the [Gemini Function Calling documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) for more information on which Gemini model versions support parallel function calling.


<img src="https://storage.googleapis.com/github-repo/generative-ai/gemini/function-calling/parallel-function-calling-in-gemini.png">

## Getting Started


### Install Vertex AI SDK and other required packages


```
%pip install --upgrade --user --quiet google-cloud-aiplatform wikipedia
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

If you are running this notebook on Google Colab, run the cell below to authenticate your environment.


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

## Code Examples

### Import libraries


```
from typing import Any

from IPython.display import Markdown, display
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    Part,
    Tool,
)
import wikipedia
```

### Define helper function


```
# Helper function to extract one or more function calls from a Gemini Function Call response


def extract_function_calls(response: GenerationResponse) -> list[dict]:
    function_calls: list[dict] = []
    if response.candidates[0].function_calls:
        for function_call in response.candidates[0].function_calls:
            function_call_dict: dict[str, dict[str, Any]] = {function_call.name: {}}
            for key, value in function_call.args.items():
                function_call_dict[function_call.name][key] = value
            function_calls.append(function_call_dict)
    return function_calls
```

## Example: Parallel function calls on the same function

A great use case for parallel function calling is when you have a function that only accepts one parameter per API call and you need to make repeated calls to that function.

With Parallel Function Calling, rather than having to send N number of API requests to Gemini for N number function calls, instead you can send a single API request to Gemini, receive N number of Function Call Responses within a single response, make N number of external API calls in your code, then return all of the API responses to Gemini in bulk. And you can do all of this without any extra configuration in your function declarations, tools, or requests to Gemini.

In this example, you'll do exactly that and use Parallel Function Calling in Gemini to ask about multiple topics on [Wikipedia](https://www.wikipedia.org/). Let's get started!

### Write function declarations and wrap them in a tool

First, define your function declarations and tool using the Vertex AI Python SDK:


```
search_wikipedia = FunctionDeclaration(
    name="search_wikipedia",
    description="Search for articles on Wikipedia",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query to search for on Wikipedia",
            },
        },
    },
)

wikipedia_tool = Tool(
    function_declarations=[
        search_wikipedia,
    ],
)
```

### Initialize the Gemini model

Now you can initialize Gemini using a [model version that supports parallel function calling](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling):


```
model = GenerativeModel(
    "gemini-1.5-pro",
    generation_config=GenerationConfig(temperature=0),
    tools=[wikipedia_tool],
)
chat = model.start_chat()
```

### Send prompt to Gemini

Send a prompt to Gemini that includes a phrase that you expect to invoke two or more function calls.

In this case we'll ask about three different article topics to search for on Wikipedia:


```
prompt = "Search for articles related to solar panels, renewable energy, and battery storage and provide a summary of your findings"

response = chat.send_message(prompt)
```

### Extract function names and parameters

Use the helper function that we created earlier to extract the function names and function parameters for each Function Call that Gemini responded with:


```
function_calls = extract_function_calls(response)
function_calls
```




    [{'search_wikipedia': {'query': 'Solar panels'}},
     {'search_wikipedia': {'query': 'Renewable energy'}},
     {'search_wikipedia': {'query': 'Battery storage'}}]



Note that the helper function is just one way to extract fields from the structured Function Call response. You can modify the helper function or write your own custom code to extract and get the fields into your preferred format or data structure!

### Make external API calls

Next, you'll loop through the Function Calls and use the `wikipedia` Python package to make an API call for each search query:


```
api_response = []

# Loop over multiple function calls
for function_call in function_calls:
    print(function_call)

    # Make external API call
    result = wikipedia.summary(function_call["search_wikipedia"]["query"])

    # Collect all API responses
    api_response.append(result)
```

    {'search_wikipedia': {'query': 'Solar panels'}}
    {'search_wikipedia': {'query': 'Renewable energy'}}
    {'search_wikipedia': {'query': 'Battery storage'}}
    

### Get a natural language summary

Now you can return all of the API responses to Gemini so that it can generate a natural language summary:


```
# Return the API response to Gemini
response = chat.send_message(
    [
        Part.from_function_response(
            name="search_wikipedia",
            response={
                "content": api_response[0],
            },
        ),
        Part.from_function_response(
            name="search_wikipedia",
            response={
                "content": api_response[1],
            },
        ),
        Part.from_function_response(
            name="search_wikipedia",
            response={
                "content": api_response[2],
            },
        ),
    ],
)
display(Markdown(response.text))
```


Solar panels are devices that convert sunlight into electricity using photovoltaic cells. They are arranged in arrays or systems and connected to inverters to convert DC electricity to AC electricity. Solar panels are a renewable and clean energy source that can reduce greenhouse gas emissions and lower electricity bills. However, they depend on sunlight availability, require cleaning, and have high initial costs.

Renewable energy refers to energy derived from renewable resources like sunlight, wind, and water. These sources are naturally replenished and have minimal environmental impact. Renewable energy systems have become increasingly efficient and affordable, leading to significant growth in their adoption. They offer a cleaner and more sustainable alternative to fossil fuels, mitigating climate change and reducing air pollution.

Battery storage power stations utilize batteries to store electrical energy, providing a rapid response to grid fluctuations and enhancing grid stability. They are essential for integrating renewable energy sources like solar and wind power, which have intermittent generation patterns. Battery storage systems can store excess energy generated during peak periods and release it when demand exceeds supply, ensuring a reliable and continuous power supply.

In summary, solar panels, renewable energy, and battery storage are interconnected aspects of a sustainable energy future. Solar panels harness solar energy, a renewable resource, while battery storage addresses the intermittency challenges associated with renewable energy sources. The combination of these technologies paves the way for a cleaner, more resilient, and sustainable energy system. 



And you're done with no extra configuration necessary!

Note that Gemini will use the information in your `FunctionDeclarations` to determine if and when it should return parallel Function Call responses, or it will determine which Function Calls need to be made before others to get information that a subsequent Function Call depends on. So be sure to account for this behavior in your logic and application code!

## Example: Parallel function calls across multiple functions

Another good fit for parallel function calling is when you have multiple, independent functions that you want to be able to call in parallel, which reduces the number of consecutive Gemini API calls that you need to make and (ideally) reduces the overall response time to the end user who is waiting for a natural language response.

In this example, you'll use Parallel Function Calling in Gemini to ask about multiple aspects of topics and articles on [Wikipedia](https://www.wikipedia.org/).

### Write function declarations and wrap them in a tool

First, define your function declarations and tool using the Vertex AI Python SDK:


```
search_wikipedia = FunctionDeclaration(
    name="search_wikipedia",
    description="Search for articles on Wikipedia",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query to search for on Wikipedia",
            },
        },
    },
)

suggest_wikipedia = FunctionDeclaration(
    name="suggest_wikipedia",
    description="Get suggested titles from Wikipedia for a given term",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query to search for suggested titles on Wikipedia",
            },
        },
    },
)

summarize_wikipedia = FunctionDeclaration(
    name="summarize_wikipedia",
    description="Get article summaries from Wikipedia",
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Query to search for article summaries on Wikipedia",
            },
        },
    },
)

wikipedia_tool = Tool(
    function_declarations=[
        search_wikipedia,
        suggest_wikipedia,
        summarize_wikipedia,
    ],
)
```

### Initialize the Gemini model

Now you can initialize Gemini using a [model version that supports parallel function calling](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling):


```
model = GenerativeModel(
    "gemini-1.5-pro",
    generation_config=GenerationConfig(temperature=0),
    tools=[wikipedia_tool],
)
chat = model.start_chat()
```

### Send prompt to Gemini

Send a prompt to Gemini that includes a phrase that you expect to invoke two or more function calls.

In this case we'll ask about three types of details to look up for a given topic on Wikipedia:


```
prompt = "Show the search results, variations, and article summaries about Wikipedia articles related to the solar system"

response = chat.send_message(prompt)
```

### Extract function names and parameters

Use the helper function that we created earlier to extract the function names and function parameters for each Function Call that Gemini responded with:


```
function_calls = extract_function_calls(response)
function_calls
```




    [{'search_wikipedia': {'query': 'Solar System'}},
     {'suggest_wikipedia': {'query': 'Solar System'}},
     {'summarize_wikipedia': {'topic': 'Solar System'}}]



### Make external API calls

Next, you'll loop through the Function Calls and use the `wikipedia` Python package to make APIs calls and gather information from Wikipedia:


```
api_response: dict[str, Any] = {}  # type: ignore

# Loop over multiple function calls
for function_call in function_calls:
    print(function_call)
    for function_name, function_args in function_call.items():
        # Determine which external API call to make
        if function_name == "search_wikipedia":
            result = wikipedia.search(function_args["query"])
        if function_name == "suggest_wikipedia":
            result = wikipedia.suggest(function_args["query"])
        if function_name == "summarize_wikipedia":
            result = wikipedia.summary(function_args["topic"], auto_suggest=False)

        # Collect all API responses
        api_response[function_name] = result
```

    {'search_wikipedia': {'query': 'Solar System'}}
    {'suggest_wikipedia': {'query': 'Solar System'}}
    {'summarize_wikipedia': {'topic': 'Solar System'}}
    

### Get a natural language summary

Now you can return all of the API responses to Gemini so that it can generate a natural language summary:


```
# Return the API response to Gemini
response = chat.send_message(
    [
        Part.from_function_response(
            name="search_wikipedia",
            response={
                "content": api_response.get("search_wikipedia", ""),
            },
        ),
        Part.from_function_response(
            name="suggest_wikipedia",
            response={
                "content": api_response.get("suggest_wikipedia", ""),
            },
        ),
        Part.from_function_response(
            name="summarize_wikipedia",
            response={
                "content": api_response.get("summarize_wikipedia", ""),
            },
        ),
    ],
)

display(Markdown(response.text))
```


Here's some information about the solar system:

Search results: "Solar System", "Formation and evolution of the Solar System", "List of Solar System objects by size", "Photovoltaic system", "Solar System (disambiguation)", "Solar System belts", "Small Solar System body", "List of Solar System objects", "Passive solar building design", "List of natural satellites"

Suggested searches: "soler system"

Summary: The Solar System formed about 4.6 billion years ago. It contains the Sun, eight planets (Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune), at least nine dwarf planets, and countless smaller objects like asteroids and comets.  The Sun, a G-type main-sequence star, contains over 99.86% of the Solar System's mass.  The outer boundary of the Solar System is theorized to be the Oort cloud, extending up to 2,000,000 AU from the Sun. 



And you're done! You successfully made parallel function calls for a couple of different use cases. Feel free to adapt the code samples here for your own use cases and applications. Or try another notebook to continue exploring other functionality in the Gemini API.

Happy parallel function calling!
