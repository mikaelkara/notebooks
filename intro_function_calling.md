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

# Intro to Function Calling with the Gemini API & Python SDK

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_function_calling.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Ffunction-calling%2Fintro_function_calling.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>      
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_function_calling.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/function-calling/intro_function_calling.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Kristopher Overholt](https://github.com/koverholt) |

## Overview

### Gemini

Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases.

### Calling functions from Gemini

[Function Calling](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/function-calling) in Gemini lets developers create a description of a function in their code, then pass that description to a language model in a request. The response from the model includes the name of a function that matches the description and the arguments to call it with.

### Why function calling?

Imagine asking someone to write down important information without giving them a form or any guidelines on the structure. You might get a beautifully crafted paragraph, but extracting specific details like names, dates, or numbers would be tedious! Similarly, trying to get consistent structured data from a generative text model without function calling can be frustrating. You're stuck explicitly prompting for things like JSON output, often with inconsistent and frustrating results.

This is where Gemini Function Calling comes in. Instead of hoping for the best in a freeform text response from a generative model, you can define clear functions with specific parameters and data types. These function declarations act as structured guidelines, guiding the Gemini model to structure its output in a predictable and usable way. No more parsing text responses for important information!

Think of it like teaching Gemini to speak the language of your applications. Need to retrieve information from a database? Define a `search_db` function with parameters for search terms. Want to integrate with a weather API? Create a `get_weather` function that takes a location as input. Function calling bridges the gap between human language and the structured data needed to interact with external systems.

### Objectives

In this tutorial, you will learn how to use the Gemini API in Vertex AI with the Vertex AI SDK for Python to make function calls via the Gemini 1.5 Pro (`gemini-1.5-pro`) model.

You will complete the following tasks:

- Install the Vertex AI SDK for Python
- Use the Gemini API in Vertex AI to interact with the Gemini 1.5 Pro (`gemini-1.5-pro`) model:
- Use Function Calling in a chat session to answer user's questions about products in the Google Store
- Use Function Calling to geocode addresses with a maps API
- Use Function Calling for entity extraction on raw logging data

### Costs

This tutorial uses billable components of Google Cloud:

- Vertex AI

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.


## Getting Started


### Install Vertex AI SDK for Python



```
%pip install --upgrade --user --quiet google-cloud-aiplatform
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
import requests
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)
```

### Chat example: Using Function Calling in a chat session to answer user's questions about the Google Store

In this example, you'll use Function Calling along with the chat modality in the Gemini model to help customers get information about products in the Google Store.

You'll start by defining three functions: one to get product information, another to get the location of the closest stores, and one more to place an order:


```
get_product_info = FunctionDeclaration(
    name="get_product_info",
    description="Get the stock amount and identifier for a given product",
    parameters={
        "type": "object",
        "properties": {
            "product_name": {"type": "string", "description": "Product name"}
        },
    },
)

get_store_location = FunctionDeclaration(
    name="get_store_location",
    description="Get the location of the closest store",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string", "description": "Location"}},
    },
)

place_order = FunctionDeclaration(
    name="place_order",
    description="Place an order",
    parameters={
        "type": "object",
        "properties": {
            "product": {"type": "string", "description": "Product name"},
            "address": {"type": "string", "description": "Shipping address"},
        },
    },
)
```

Note that function parameters are specified as a Python dictionary in accordance with the [OpenAPI JSON schema format](https://spec.openapis.org/oas/v3.0.3#schemawr).

Define a tool that allows the Gemini model to select from the set of 3 functions:


```
retail_tool = Tool(
    function_declarations=[
        get_product_info,
        get_store_location,
        place_order,
    ],
)
```

Now you can initialize the Gemini model with Function Calling in a multi-turn chat session.

You can specify the `tools` kwarg when initializing the model to avoid having to send this kwarg with every subsequent request:


```
model = GenerativeModel(
    "gemini-1.5-pro",
    generation_config=GenerationConfig(temperature=0),
    tools=[retail_tool],
)
chat = model.start_chat()
```

*Note: The temperature parameter controls the degree of randomness in this generation. Lower temperatures are good for functions that require deterministic parameter values, while higher temperatures are good for functions with parameters that accept more diverse or creative parameter values. A temperature of 0 is deterministic. In this case, responses for a given prompt are mostly deterministic, but a small amount of variation is still possible.*

We're ready to chat! Let's start the conversation by asking if a certain product is in stock:


```
prompt = """
Do you have the Pixel 8 Pro in stock?
"""

response = chat.send_message(prompt)
response.candidates[0].content.parts[0]
```




    function_call {
      name: "get_product_info"
      args {
        fields {
          key: "product_name"
          value {
            string_value: "Pixel 8 Pro"
          }
        }
      }
    }



The response from the Gemini API consists of a structured data object that contains the name and parameters of the function that Gemini selected out of the available functions.

Since this notebook focuses on the ability to extract function parameters and generate function calls, you'll use mock data to feed synthetic responses back to the Gemini model rather than sending a request to an API server (not to worry, we'll make an actual API call in a later example!):


```
# Here you can use your preferred method to make an API request and get a response.
# In this example, we'll use synthetic data to simulate a payload from an external API response.

api_response = {"sku": "GA04834-US", "in_stock": "yes"}
```

In reality, you would execute function calls against an external system or database using your desired client library or REST API.

Now, you can pass the response from the (mock) API request and generate a response for the end user:


```
response = chat.send_message(
    Part.from_function_response(
        name="get_product_info",
        response={
            "content": api_response,
        },
    ),
)
response.text
```




    'Yes, the Pixel 8 Pro is in stock. \n'



Next, the user might ask where they can buy a different phone from a nearby store:


```
prompt = """
What about the Pixel 8? Is there a store in
Mountain View, CA that I can visit to try one out?
"""

response = chat.send_message(prompt)
response.candidates[0].content.parts[0]
```




    function_call {
      name: "get_product_info"
      args {
        fields {
          key: "product_name"
          value {
            string_value: "Pixel 8"
          }
        }
      }
    }



Again, you get a response with structured data, and the Gemini model selected the `get_product_info` function. This happened since the user asked about the "Pixel 8" phone this time rather than the "Pixel 8 Pro" phone.

Now you can build another synthetic payload that would come from an external API:


```
# Here you can use your preferred method to make an API request and get a response.
# In this example, we'll use synthetic data to simulate a payload from an external API response.

api_response = {"sku": "GA08475-US", "in_stock": "yes"}
```

Again, you can pass the response from the (mock) API request back to the Gemini model:


```
response = chat.send_message(
    Part.from_function_response(
        name="get_product_info",
        response={
            "content": api_response,
        },
    ),
)
response.candidates[0].content.parts[0]
```




    function_call {
      name: "get_store_location"
      args {
        fields {
          key: "location"
          value {
            string_value: "Mountain View, CA"
          }
        }
      }
    }



Wait a minute! Why did the Gemini API respond with a second function call to `get_store_location` this time rather than a natural language summary? Look closely at the prompt that you used in this conversation turn a few cells up, and you'll notice that the user asked about a product -and- the location of a store.

In cases like this when two or more functions are defined (or when the model predicts multiple function calls to the same function), the Gemini model might sometimes return back-to-back or parallel function call responses within a single conversation turn.

This is expected behavior since the Gemini model predicts which functions it should call at runtime, what order it should call dependent functions in, and which function calls can be parallelized, so that the model can gather enough information to generate a natural language response.

Not to worry! You can repeat the same steps as before and build another synthetic payload that would come from an external API:


```
# Here you can use your preferred method to make an API request and get a response.
# In this example, we'll use synthetic data to simulate a payload from an external API response.

api_response = {"store": "2000 N Shoreline Blvd, Mountain View, CA 94043, US"}
```

And you can pass the response from the (mock) API request back to the Gemini model:


```
response = chat.send_message(
    Part.from_function_response(
        name="get_store_location",
        response={
            "content": api_response,
        },
    ),
)
response.text
```




    'Yes, the Pixel 8 is in stock. You can visit the store at 2000 N Shoreline Blvd, Mountain View, CA 94043, US to try it out. \n'



Nice work!

Within a single conversation turn, the Gemini model requested 2 function calls in a row before returning a natural language summary. In reality, you might follow this pattern if you need to make an API call to an inventory management system, and another call to a store location database, customer management system, or document repository.

Finally, the user might ask to order a phone and have it shipped to their address:


```
prompt = """
I'd like to order a Pixel 8 Pro and have it shipped to 1155 Borregas Ave, Sunnyvale, CA 94089.
"""

response = chat.send_message(prompt)
response.candidates[0].content.parts[0]
```




    function_call {
      name: "place_order"
      args {
        fields {
          key: "product"
          value {
            string_value: "Pixel 8 Pro"
          }
        }
        fields {
          key: "address"
          value {
            string_value: "1155 Borregas Ave, Sunnyvale, CA 94089"
          }
        }
      }
    }



Perfect! The Gemini model extracted the user's selected product and their address. Now you can call an API to place the order:


```
# This is where you would make an API request to return the status of their order.
# Use synthetic data to simulate a response payload from an external API.

api_response = {
    "payment_status": "paid",
    "order_number": 12345,
    "est_arrival": "2 days",
}
```

And send the payload from the external API call so that the Gemini API returns a natural language summary to the end user.


```
response = chat.send_message(
    Part.from_function_response(
        name="place_order",
        response={
            "content": api_response,
        },
    ),
)
response.text
```




    'Your order has been placed and will arrive in 2 days. Your order number is 12345. \n'



And you're done!

You were able to have a multi-turn conversation with the Gemini model using function calls, handling payloads, and generating natural language summaries that incorporated the information from the external systems.

### Address example: Using Function Calling to geocode addresses with a maps API

In this example, you'll use the text modality in the Gemini API to define a function that takes multiple parameters as inputs. You'll use the function call response to then make a live API call to convert an address to latitude and longitude coordinates.

Start by defining a function declaration and wrapping it in a tool:


```
get_location = FunctionDeclaration(
    name="get_location",
    description="Get latitude and longitude for a given location",
    parameters={
        "type": "object",
        "properties": {
            "poi": {"type": "string", "description": "Point of interest"},
            "street": {"type": "string", "description": "Street name"},
            "city": {"type": "string", "description": "City name"},
            "county": {"type": "string", "description": "County name"},
            "state": {"type": "string", "description": "State name"},
            "country": {"type": "string", "description": "Country name"},
            "postal_code": {"type": "string", "description": "Postal code"},
        },
    },
)

location_tool = Tool(
    function_declarations=[get_location],
)
```

In this example, you're asking the Gemini model to extract components of the address into specific fields within a structured data object. You can then map this data to specific input fields to use with your REST API or client library.

Send a prompt that includes an address, such as:


```
prompt = """
I want to get the coordinates for the following address:
1600 Amphitheatre Pkwy, Mountain View, CA 94043, US
"""

response = model.generate_content(
    prompt,
    generation_config=GenerationConfig(temperature=0),
    tools=[location_tool],
)
response.candidates[0].content.parts[0]
```




    function_call {
      name: "get_location"
      args {
        fields {
          key: "street"
          value {
            string_value: "1600 Amphitheatre Pkwy"
          }
        }
        fields {
          key: "state"
          value {
            string_value: "CA"
          }
        }
        fields {
          key: "postal_code"
          value {
            string_value: "94043"
          }
        }
        fields {
          key: "country"
          value {
            string_value: "US"
          }
        }
        fields {
          key: "city"
          value {
            string_value: "Mountain View"
          }
        }
      }
    }



Now you can reference the parameters from the function call and make a live API request:


```
x = response.candidates[0].content.parts[0].function_call.args

url = "https://nominatim.openstreetmap.org/search?"
for i in x:
    url += f'{i}="{x[i]}"&'
url += "format=json"

headers = {"User-Agent": "none"}
x = requests.get(url, headers=headers)
content = x.json()
content

# Note: if you get a JSONDecodeError when running this cell, try modifying the
# user agent string in the `headers=` line of code in this cell and re-run.
```




    [{'place_id': 377680635,
      'licence': 'Data © OpenStreetMap contributors, ODbL 1.0. http://osm.org/copyright',
      'osm_type': 'node',
      'osm_id': 2192620021,
      'lat': '37.4217636',
      'lon': '-122.084614',
      'class': 'office',
      'type': 'it',
      'place_rank': 30,
      'importance': 0.6949356759210291,
      'addresstype': 'office',
      'name': 'Google Headquarters',
      'display_name': 'Google Headquarters, 1600, Amphitheatre Parkway, Mountain View, Santa Clara County, California, 94043, United States',
      'boundingbox': ['37.4217136', '37.4218136', '-122.0846640', '-122.0845640']}]



Great work! You were able to define a function that the Gemini model used to extract the relevant parameters from the prompt. Then you made a live API call to obtain the coordinates of the specified location.

Here we used the [OpenStreetMap Nominatim API](https://nominatim.openstreetmap.org/ui/search.html) to geocode an address to keep the number of steps in this tutorial to a reasonable number. If you're working with large amounts of address or geolocation data, you can also use the [Google Maps Geocoding API](https://developers.google.com/maps/documentation/geocoding), or any mapping service with an API!

### Logging example: Using Function Calling for entity extraction only

In the previous examples, we made use of the entity extraction functionality within Gemini Function Calling so that we could pass the resulting parameters to a REST API or client library. However, you might want to only perform the entity extraction step with Gemini Function Calling and stop there without actually calling an API. You can think of this functionality as a convenient way to transform unstructured text data into structured fields.

In this example, you'll build a log extractor that takes raw log data and transforms it into structured data with details about error messages.

You'll start by specifying a function declaration that represents the schema of the Function Call:


```
extract_log_data = FunctionDeclaration(
    name="extract_log_data",
    description="Extract details from error messages in raw log data",
    parameters={
        "type": "object",
        "properties": {
            "locations": {
                "type": "array",
                "description": "Errors",
                "items": {
                    "description": "Details of the error",
                    "type": "object",
                    "properties": {
                        "error_message": {
                            "type": "string",
                            "description": "Full error message",
                        },
                        "error_code": {"type": "string", "description": "Error code"},
                        "error_type": {"type": "string", "description": "Error type"},
                    },
                },
            }
        },
    },
)
```

You can then define a tool for the generative model to call that includes the `extract_log_data`:

Define a tool for the Gemini model to use that includes the log extractor function:


```
extraction_tool = Tool(
    function_declarations=[extract_log_data],
)
```

You can then pass the sample log data to the Gemini model. The model will call the log extractor function, and the model output will be a Function Call response.


```
prompt = """
[15:43:28] ERROR: Could not process image upload: Unsupported file format. (Error Code: 308)
[15:44:10] INFO: Search index updated successfully.
[15:45:02] ERROR: Service dependency unavailable (payment gateway). Retrying... (Error Code: 5522)
[15:45:33] ERROR: Application crashed due to out-of-memory exception. (Error Code: 9001)
"""

response = model.generate_content(
    prompt,
    generation_config=GenerationConfig(temperature=0),
    tools=[extraction_tool],
)

response.candidates[0].content.parts[0].function_call
```




    name: "extract_log_data"
    args {
      fields {
        key: "locations"
        value {
          list_value {
            values {
              struct_value {
                fields {
                  key: "error_type"
                  value {
                    string_value: "ERROR"
                  }
                }
                fields {
                  key: "error_message"
                  value {
                    string_value: "Could not process image upload: Unsupported file format."
                  }
                }
                fields {
                  key: "error_code"
                  value {
                    string_value: "308"
                  }
                }
              }
            }
            values {
              struct_value {
                fields {
                  key: "error_type"
                  value {
                    string_value: "ERROR"
                  }
                }
                fields {
                  key: "error_message"
                  value {
                    string_value: "Service dependency unavailable (payment gateway). Retrying..."
                  }
                }
                fields {
                  key: "error_code"
                  value {
                    string_value: "5522"
                  }
                }
              }
            }
            values {
              struct_value {
                fields {
                  key: "error_type"
                  value {
                    string_value: "ERROR"
                  }
                }
                fields {
                  key: "error_message"
                  value {
                    string_value: "Application crashed due to out-of-memory exception."
                  }
                }
                fields {
                  key: "error_code"
                  value {
                    string_value: "9001"
                  }
                }
              }
            }
          }
        }
      }
    }



The response includes a structured data object that contains the details of the error messages that appear in the log.

## Conclusions

You have explored the function calling feature through the Vertex AI Python SDK.

The next step is to enhance your skills by exploring this [documenation page](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling).
