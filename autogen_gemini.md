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

# Building a Weather Agent with AutoGen and Gemini

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/orchestration/autogen_gemini.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Forchestration%2Fautogen_gemini.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/orchestration/autogen_gemini.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/orchestration/autogen_gemini.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
| Author(s) | [Karl Weinmeister](https://github.com/kweinmeister/) |

## Overview

This notebook demonstrates how to build a weather agent using [Autogen](https://microsoft.github.io/autogen/) with the [Gemini 1.5 Flash](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models) model on Vertex AI. The agent can understand free-form location queries, retrieve location coordinates using the [Nominatim API](https://nominatim.org/release-docs/latest/api/Overview/), and fetch weather forecasts using the [Open-Meteo API](https://open-meteo.com/en/docs). This example showcases Autogen's ability to integrate external APIs and tools within a conversational AI framework.

By the end of this notebook, you will learn how to:

* Define custom functions using Autogen's function registration decorators.
* Integrate external APIs (Nominatim and Open-Meteo) within your agent's functions.
* Create and manage conversations between a user proxy agent and a specialized assistant agent.
* Leverage Gemini for natural language understanding and response generation.

## Steps performed in this notebook:

* Define an [`AssistantAgent`](https://microsoft.github.io/autogen/docs/reference/agentchat/assistant_agent/) for weather information and a [`UserProxyAgent`](https://microsoft.github.io/autogen/docs/reference/agentchat/user_proxy_agent/) to simulate user interaction.
* Register custom Python functions (`search_location` and `get_weather_forecast`) with the `AssistantAgent`, making them callable by the language model.
* Integrate with the Nominatim API to geocode location queries and the Open-Meteo API to fetch weather forecasts.
* Initiate a chat between the user proxy and the weather agent, providing a sample query like "What's the weather like in Paris?"

## Get started

### Install Vertex AI SDK and other required packages



```
%pip install --upgrade --user --quiet google-cloud-aiplatform pyautogen[gemini] dask[dataframe]==2024.7.1
```

### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.


```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```




    {'status': 'ok', 'restart': True}



<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.


```
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Set Google Cloud project information and configure Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
# Use the environment variable if the user doesn't provide Project ID.
import os

PROJECT_ID = "[your-project-id]"  # @param {type:"string", isTemplate: true}
if PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

MODEL = "google/gemini-1.5-flash-001"  # @param {type:"string", isTemplate: true}

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

# Pricing parameters for AutoGen using the OpenAI API.
# The agent will work without these, but will log warnings.
# For latest pricing, see:
# https://cloud.google.com/vertex-ai/generative-ai/pricing
INPUT_PRICE_1K_CHARS = 0.00001875
OUTPUT_PRICE_1K_CHARS = 0.000075
OUTPUT_PRICE_1K_TOKENS = OUTPUT_PRICE_1K_CHARS * 4  # Estimate
```

## LLM Configuration

Next, we will define the AutoGen [LLM configuration](https://microsoft.github.io/autogen/docs/topics/llm_configuration/) for AutoGen.

As [tool use](https://microsoft.github.io/autogen/docs/tutorial/tool-use) in AutoGen is currently limited to the OpenAI API, we'll use the OpenAI [interface in Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-vertex-using-openai-library).

You can find more details on configuring AutoGen for Gemini API in Vertex AI [here](https://microsoft.github.io/autogen/docs/topics/non-openai-models/cloud-gemini_vertexai).


```
import google.auth
import google.auth.transport.requests

scopes = ["https://www.googleapis.com/auth/cloud-platform"]
creds, _ = google.auth.default(scopes)
auth_req = google.auth.transport.requests.Request()
creds.refresh(auth_req)

config_list = [
    {
        "model": MODEL,
        "api_type": "openai",
        "base_url": f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi",
        "api_key": creds.token,
        "price": [INPUT_PRICE_1K_CHARS, OUTPUT_PRICE_1K_TOKENS],
    }
]
```

## Agent development

### Import libraries


```
import time
from typing import Annotated

from autogen import AssistantAgent, Cache, UserProxyAgent
import requests
```

### Define agents

An [`AssistantAgent`](https://microsoft.github.io/autogen/docs/reference/agentchat/assistant_agent/) in AutoGen is a specialized agent designed to perform specific tasks or provide information within a conversational AI framework. It leverages a large language model to generate responses and interact with other agents.

In this scenario, our `weather_agent` will be responsible for understanding user queries about the weather, retrieving relevant information from external APIs (like location coordinates and weather forecasts), and providing formatting responses to the user.


```
weather_agent = AssistantAgent(
    name="WeatherAgent",
    description="""A weather assistant that summarizes and provides helpful
    details, customized for the user's query.""",
    llm_config={
        "config_list": config_list,
    },
)
```

A [`UserProxyAgent`](https://microsoft.github.io/autogen/docs/reference/agentchat/user_proxy_agent/) is an agent that acts as a proxy for a human user within a conversational AI system.  It can receive user input, either directly from a human user or from a predefined script, and then forward it to the other agents in the conversation.

In this scenario, the UserProxyAgent will be responsible for simulating a user who is interested in learning about the weather. It will receive user queries, such as "What's the weather like in Paris?", and forward them to the `weather_agent`. The `UserProxyAgent` will also receive the response from the `weather_agent` and display it to the user.

This allows us to test and explore the capabilities of the `weather_agent` without requiring a human user to interact with the system directly.



```
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
    is_termination_msg=lambda x: x.get("content", "")
    and x.get("content", "").rstrip().endswith("TERMINATE"),
)
```

## Define tools

There will be two tools we use in this scenario:
* `search_location` helps us pinpoint coordinates based on the user's query
* `get_weather_forecast` accepts the coordinates and then retrieves the weather

We will [register each tool](https://microsoft.github.io/autogen/docs/tutorial/tool-use/#registering-tools) using a decorator, so each agent can use them.


```
@user_proxy.register_for_execution()
@weather_agent.register_for_llm(
    description="Performs a free-form location search using the Nominatim API."
)
def search_location(
    query: Annotated[
        str, "A natural language or structured query containing a location"
    ]
) -> tuple[float, float] | None:
    """Performs a free-form location search using the Nominatim API."""
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": "1",
        "email": "your_email@example.com",  # Replace with your email
    }
    headers = {
        "User-Agent": f"MyWeatherApp/1.0 ({params['email']})",
    }
    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        search_results = response.json()
        lat = float(search_results[0]["lat"])
        lon = float(search_results[0]["lon"])
        return lat, lon
    except requests.exceptions.RequestException as e:
        print(f"Error during Nominatim API request: {e}")
        return None
    finally:
        time.sleep(1)
```


```
@user_proxy.register_for_execution()
@weather_agent.register_for_llm(
    description="Retrieves the weather forecast for a given latitude and longitude."
)
def get_weather_forecast(
    latitude: Annotated[float, "Distance north or south of the equator"],
    longitude: Annotated[float, "Distance east or west of the prime meridian"],
) -> dict | None:
    """Retrieves the weather forecast using the Open Meteo API."""
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": str(latitude),
        "longitude": str(longitude),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "timezone": "auto",
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        weather_forecast = response.json()
        print(weather_forecast)
        return weather_forecast
    except requests.exceptions.RequestException as e:
        print(f"Error during Open Meteo API request: {e}")
        return None
```

## Initiate conversation

We are ready to test our agent! We will [initiate the chat](https://microsoft.github.io/autogen/docs/reference/agentchat/conversable_agent/#initiate_chat), where you can see each step. The agents will perform tasks and communicate with each other.

[Caching](https://microsoft.github.io/autogen/docs/topics/llm-caching]) is enabled to reduce cost in our testing scenario.



```
with Cache.disk() as cache:
    result = user_proxy.initiate_chat(
        weather_agent, message="What's the weather like in Paris?", cache=cache
    )
```

    UserProxy (to WeatherAgent):
    
    What's the weather like in Paris?
    
    --------------------------------------------------------------------------------
    WeatherAgent (to UserProxy):
    
    ***** Suggested tool call (search_location): search_location *****
    Arguments: 
    {"query":"Paris"}
    ******************************************************************
    
    --------------------------------------------------------------------------------
    
    >>>>>>>> EXECUTING FUNCTION search_location...
    UserProxy (to WeatherAgent):
    
    UserProxy (to WeatherAgent):
    
    ***** Response from calling tool (search_location) *****
    [48.8588897, 2.3200410217200766]
    ********************************************************
    
    --------------------------------------------------------------------------------
    WeatherAgent (to UserProxy):
    
    ***** Suggested tool call (get_weather_forecast): get_weather_forecast *****
    Arguments: 
    {"latitude":48.8588897,"longitude":2.320041021720077}
    ****************************************************************************
    
    --------------------------------------------------------------------------------
    
    >>>>>>>> EXECUTING FUNCTION get_weather_forecast...
    {'latitude': 48.86, 'longitude': 2.3199997, 'generationtime_ms': 0.12099742889404297, 'utc_offset_seconds': 7200, 'timezone': 'Europe/Paris', 'timezone_abbreviation': 'CEST', 'elevation': 38.0, 'daily_units': {'time': 'iso8601', 'temperature_2m_max': '°C', 'temperature_2m_min': '°C', 'precipitation_sum': 'mm', 'windspeed_10m_max': 'km/h'}, 'daily': {'time': ['2024-09-15', '2024-09-16', '2024-09-17', '2024-09-18', '2024-09-19', '2024-09-20', '2024-09-21'], 'temperature_2m_max': [20.0, 20.9, 20.4, 21.4, 20.9, 23.2, 24.2], 'temperature_2m_min': [5.6, 10.6, 11.4, 12.6, 13.1, 12.0, 11.5], 'precipitation_sum': [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0], 'windspeed_10m_max': [7.3, 13.5, 18.9, 15.8, 14.8, 9.1, 6.7]}}
    UserProxy (to WeatherAgent):
    
    UserProxy (to WeatherAgent):
    
    ***** Response from calling tool (get_weather_forecast) *****
    {"latitude": 48.86, "longitude": 2.3199997, "generationtime_ms": 0.12099742889404297, "utc_offset_seconds": 7200, "timezone": "Europe/Paris", "timezone_abbreviation": "CEST", "elevation": 38.0, "daily_units": {"time": "iso8601", "temperature_2m_max": "°C", "temperature_2m_min": "°C", "precipitation_sum": "mm", "windspeed_10m_max": "km/h"}, "daily": {"time": ["2024-09-15", "2024-09-16", "2024-09-17", "2024-09-18", "2024-09-19", "2024-09-20", "2024-09-21"], "temperature_2m_max": [20.0, 20.9, 20.4, 21.4, 20.9, 23.2, 24.2], "temperature_2m_min": [5.6, 10.6, 11.4, 12.6, 13.1, 12.0, 11.5], "precipitation_sum": [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0], "windspeed_10m_max": [7.3, 13.5, 18.9, 15.8, 14.8, 9.1, 6.7]}}
    *************************************************************
    
    --------------------------------------------------------------------------------
    WeatherAgent (to UserProxy):
    
    The weather in Paris is expected to be pleasant with a high of 20°C and a low of 5.6°C. There is no precipitation expected for the next 7 days. 
    TERMINATE
    
    
    --------------------------------------------------------------------------------
    

Let's extract the summary from the `result`. Congratulations on finishing the tutorial! 🎉


```
print(result.summary)
```

    The weather in Paris is expected to be pleasant with a high of 20°C and a low of 5.6°C. There is no precipitation expected for the next 7 days. 
    
    
    
