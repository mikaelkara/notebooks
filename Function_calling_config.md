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

# Gemini API: Function calling config

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Function_calling_config.ipynb"><img src="../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

Specifying a `function_calling_config` allows you to control how the Gemini API acts when `tools` have been specified. For example, you can choose to only allow free-text output (disabling function calling), force it to choose from a subset of the functions provided in `tools`, or let it act automatically.

This guide assumes you are already familiar with function calling. For an introduction, check out the [docs](https://ai.google.dev/docs/function_calling).


```
!pip install -U -q "google-generativeai>=0.7.2"
```

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](https://github.com/google-gemini/gemini-api-cookbook/blob/main/quickstarts/Authentication.ipynb) quickstart for an example.


```
from google.colab import userdata
import google.generativeai as genai

genai.configure(api_key=userdata.get("GOOGLE_API_KEY"))
```

## Set up a model with tools

This example uses 3 functions that control a simple hypothetical lighting system. Using these functions requires them to be called in a specific order. For example, you must turn the light system on before you can change color.

While you can pass these directly to the model and let it try to call them correctly, specifying the `function_calling_config` gives you precise control over the functions that are available to the model.


```
def enable_lights():
    """Turn on the lighting system."""
    print("LIGHTBOT: Lights enabled.")


def set_light_color(rgb_hex: str):
    """Set the light color. Lights must be enabled for this to work."""
    print(f"LIGHTBOT: Lights set to {rgb_hex}.")


def stop_lights():
    """Stop flashing lights."""
    print("LIGHTBOT: Lights turned off.")


light_controls = [enable_lights, set_light_color, stop_lights]
instruction = "You are a helpful lighting system bot. You can turn lights on and off, and you can set the color. Do not perform any other tasks."

model = genai.GenerativeModel(
    "models/gemini-1.5-pro", tools=light_controls, system_instruction=instruction
)

chat = model.start_chat()
```

Create a helper function for setting `function_calling_config` on `tool_config`.


```
from google.generativeai.types import content_types
from collections.abc import Iterable


def tool_config_from_mode(mode: str, fns: Iterable[str] = ()):
    """Create a tool config with the specified function calling mode."""
    return content_types.to_tool_config(
        {"function_calling_config": {"mode": mode, "allowed_function_names": fns}}
    )
```

## Text-only mode: `NONE`

If you have provided the model with tools, but do not want to use those tools for the current conversational turn, then specify `NONE` as the mode. `NONE` tells the model not to make any function calls, and will behave as though none have been provided.


```
tool_config = tool_config_from_mode("none")

response = chat.send_message(
    "Hello light-bot, what can you do?", tool_config=tool_config
)
print(response.text)
```

    I can turn lights on and off, and I can set the color of the lights.  What can I do for you? 
    
    

## Automatic mode: `AUTO`

To allow the model to decide whether to respond in text or call specific functions, you can specify `AUTO` as the mode.


```
tool_config = tool_config_from_mode("auto")

response = chat.send_message("Light this place up!", tool_config=tool_config)
print(response.parts[0])
chat.rewind();  # You are not actually calling the function, so remove this from the history.
```

    function_call {
      name: "enable_lights"
      args {
      }
    }
    
    

## Function-calling mode: `ANY`

Setting the mode to `ANY` will force the model to make a function call. By setting `allowed_function_names`, the model will only choose from those functions. If it is not set, all of the functions in `tools` are candidates for function calling.

In this example system, if the lights are already on, then the user can change color or turn the lights off.


```
available_fns = ["set_light_color", "stop_lights"]

tool_config = tool_config_from_mode("any", available_fns)

response = chat.send_message("Make this place PURPLE!", tool_config=tool_config)
print(response.parts[0])
```

    function_call {
      name: "set_light_color"
      args {
        fields {
          key: "rgb_hex"
          value {
            string_value: "800080"
          }
        }
      }
    }
    
    

## Automatic function calling

`tool_config` works when enabling automatic function calling too.


```
available_fns = ["enable_lights"]
tool_config = tool_config_from_mode("any", available_fns)

auto_chat = model.start_chat(enable_automatic_function_calling=True)
auto_chat.send_message("It's awful dark in here...", tool_config=tool_config)
```

    LIGHTBOT: Lights enabled.
    




    response:
    GenerateContentResponse(
        done=True,
        iterator=None,
        result=glm.GenerateContentResponse({
          "candidates": [
            {
              "content": {
                "parts": [
                  {
                    "text": "Let there be light! \ud83d\udca1 \n"
                  }
                ],
                "role": "model"
              },
              "finish_reason": 1,
              "index": 0,
              "safety_ratings": [
                {
                  "category": 9,
                  "probability": 1,
                  "blocked": false
                },
                {
                  "category": 8,
                  "probability": 1,
                  "blocked": false
                },
                {
                  "category": 7,
                  "probability": 1,
                  "blocked": false
                },
                {
                  "category": 10,
                  "probability": 1,
                  "blocked": false
                }
              ],
              "token_count": 0,
              "grounding_attributions": []
            }
          ]
        }),
    )



## Further reading

Check out the function calling [quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Function_calling.ipynb) for an introduction to function calling. You can find another fun function calling example [here](https://github.com/google-gemini/cookbook/blob/main/quickstarts/rest/Function_calling_REST.ipynb) using curl.

