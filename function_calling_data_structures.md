```python
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

# Working with Data Structures and Schemas in Gemini Function Calling

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/function_calling_data_structures.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Ffunction-calling%2Ffunction_calling_data_structures.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/function-calling/function_calling_data_structures.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/function_calling_data_structures.ipynb">
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

In this tutorial, you'll learn how to work with various data structures within Gemini Function Calling, including:
    
- Single parameter
- Multiple parameters
- Lists of parameters
- Nested parameters and data structures

## Getting Started


### Install Vertex AI SDK and other required packages


```python
%pip install --upgrade --user --quiet google-cloud-aiplatform
```

### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.


```python
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the cell below to authenticate your environment.


```python
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```python
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

## Code Examples

### Import libraries


```python
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Tool,
)
```

### Initialize model



```python
model = GenerativeModel(
    "gemini-1.5-pro",
    generation_config=GenerationConfig(temperature=0),
)
```

### Example: Single parameter

Let's say that you want to extract a location from a prompt to help a user navigate to their desired destination.

You can build out a simple schema for a function that takes a single parameter as an input:


```python
get_destination = FunctionDeclaration(
    name="get_destination",
    description="Get directions to a destination",
    parameters={
        "type": "object",
        "properties": {
            "destination": {
                "type": "string",
                "description": "Destination that the user wants to go to",
            },
        },
    },
)

destination_tool = Tool(
    function_declarations=[get_destination],
)
```

Now you can send a prompt with a destination, and the model will return structured data with a single key/value pair:


```python
prompt = "I'd like to travel to Paris"

response = model.generate_content(
    prompt,
    tools=[destination_tool],
)

response.candidates[0].function_calls
```




    [name: "get_destination"
     args {
       fields {
         key: "destination"
         value {
           string_value: "Paris"
         }
       }
     }]



### Example: Multiple parameters

What if you want the function call to return more than one parameter?

You can build out a simple schema for a function that takes multiple parameters as an input:


```python
get_destination_params = FunctionDeclaration(
    name="get_destination_params",
    description="Get directions to a destination",
    parameters={
        "type": "object",
        "properties": {
            "destination": {
                "type": "string",
                "description": "Destination that the user wants to go to",
            },
            "mode_of_transportation": {
                "type": "string",
                "description": "Mode of transportation to use",
            },
            "departure_time": {
                "type": "string",
                "description": "Time that the user will leave for the destination",
            },
        },
    },
)

destination_tool = Tool(
    function_declarations=[get_destination_params],
)
```

Now you can send a prompt with a destination, and the model will return structured data with a single key/value pair:


```python
prompt = "I'd like to travel to Paris by train and leave at 9:00 am"

response = model.generate_content(
    prompt,
    tools=[destination_tool],
)

response.candidates[0].function_calls
```




    [name: "get_destination_params"
     args {
       fields {
         key: "mode_of_transportation"
         value {
           string_value: "train"
         }
       }
       fields {
         key: "destination"
         value {
           string_value: "Paris"
         }
       }
       fields {
         key: "departure_time"
         value {
           string_value: "9:00 am"
         }
       }
     }]



### Example: Lists of parameters

What if you want the function call to return an array or list of parameters within an object?

For example, you might want to call an API to get the geocoded coordinates of several different locations within a single prompt.

In that case, you can build out a schema for a function that takes an array as an input:


```python
get_multiple_location_coordinates = FunctionDeclaration(
    name="get_location_coordinates",
    description="Get coordinates of multiple locations",
    parameters={
        "type": "object",
        "properties": {
            "locations": {
                "type": "array",
                "description": "A list of locations",
                "items": {
                    "description": "Components of the location",
                    "type": "object",
                    "properties": {
                        "point_of_interest": {
                            "type": "string",
                            "description": "Name or type of point of interest",
                        },
                        "city": {"type": "string", "description": "City"},
                        "country": {"type": "string", "description": "Country"},
                    },
                    "required": [
                        "point_of_interest",
                        "city",
                        "country",
                    ],
                },
            }
        },
    },
)

geocoding_tool = Tool(
    function_declarations=[get_multiple_location_coordinates],
)
```

Now we'll send a prompt with a few different locations and points of interest:


```python
prompt = """
    I'd like to get the coordinates for
    the Eiffel tower in Paris,
    the statue of liberty in New York,
    and Port Douglas near the Great Barrier Reef.
"""

response = model.generate_content(
    prompt,
    tools=[geocoding_tool],
)

response.candidates[0].function_calls
```




    [name: "get_location_coordinates"
     args {
       fields {
         key: "locations"
         value {
           list_value {
             values {
               struct_value {
                 fields {
                   key: "point_of_interest"
                   value {
                     string_value: "Eiffel Tower"
                   }
                 }
                 fields {
                   key: "country"
                   value {
                     string_value: "France"
                   }
                 }
                 fields {
                   key: "city"
                   value {
                     string_value: "Paris"
                   }
                 }
               }
             }
             values {
               struct_value {
                 fields {
                   key: "point_of_interest"
                   value {
                     string_value: "Statue of Liberty"
                   }
                 }
                 fields {
                   key: "country"
                   value {
                     string_value: "USA"
                   }
                 }
                 fields {
                   key: "city"
                   value {
                     string_value: "New York"
                   }
                 }
               }
             }
             values {
               struct_value {
                 fields {
                   key: "point_of_interest"
                   value {
                     string_value: "Great Barrier Reef"
                   }
                 }
                 fields {
                   key: "country"
                   value {
                     string_value: "Australia"
                   }
                 }
                 fields {
                   key: "city"
                   value {
                     string_value: "Port Douglas"
                   }
                 }
               }
             }
           }
         }
       }
     }]



Note that the generative model populated values for all of the parameters for a given location since all three parameters are required.

### Example: Nested parameters and data structures

What if you want the function call to include nested parameters or other complex data structures?

You might want to send a command to create a product listing based on a few sentences that include the product details.

In that case, you can build out a schema for a function that takes nested data structures as an input:


```python
create_product_listing = FunctionDeclaration(
    name="create_product_listing",
    description="Create a product listing using the details provided by the user",
    parameters={
        "type": "object",
        "properties": {
            "product": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "brand": {"type": "string"},
                    "price": {"type": "number"},
                    "category": {"type": "string"},
                    "description": {"type": "string"},
                    "colors": {
                        "type": "object",
                        "properties": {
                            "color": {"type": "number"},
                        },
                    },
                },
            }
        },
    },
)

product_listing_tool = Tool(
    function_declarations=[create_product_listing],
)
```

Now we'll send a prompt with a few different locations and attributes:


```python
prompt = """Create a listing for noise-canceling headphones for $149.99.
These headphones create a distraction-free environment.
Available colors include black, white, and red."""

response = model.generate_content(
    prompt,
    tools=[product_listing_tool],
)

response.candidates[0].function_calls
```




    [name: "create_product_listing"
     args {
       fields {
         key: "product"
         value {
           list_value {
             values {
               struct_value {
                 fields {
                   key: "price"
                   value {
                     string_value: "$149.99"
                   }
                 }
                 fields {
                   key: "name"
                   value {
                     string_value: "Noise-Canceling Headphones"
                   }
                 }
                 fields {
                   key: "description"
                   value {
                     string_value: "Create a distraction-free environment."
                   }
                 }
                 fields {
                   key: "available_colors"
                   value {
                     list_value {
                       values {
                         string_value: "black"
                       }
                       values {
                         string_value: "white"
                       }
                       values {
                         string_value: "red"
                       }
                     }
                   }
                 }
               }
             }
           }
         }
       }
     }]



And you're done! You successfully generated various types of data structures, including a single parameter, multiple parameters, a list of parameters, and nested parameters. Try another notebook to continue exploring other functionality in the Gemini API!
