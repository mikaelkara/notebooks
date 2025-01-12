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

# Gemini API: Function calling config with REST

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/rest/Function_calling_config_REST.ipynb"><img src="../../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

Specifying a `function_calling_config` allows you to control how the Gemini API acts when `tools` have been specified. For example, you can choose to only allow free-text output (disabling function calling), force it to choose from a subset of the functions provided in `tools`, or let it act automatically.

This guide assumes you are already familiar with function calling. For an introduction, check out the [Function calling with REST](./Function_calling_REST.ipynb) recipe.

This notebook provides quick code examples that show you how to get started with function calling using `curl`.

You can run this in Google Colab, or you can copy/paste the `curl` commands into your terminal.

To run this notebook, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you are running in a different environment, you can store your key in an environment variable. See [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) to learn more.


```
import os
from google.colab import userdata

os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

## Set up a model with tools

This example provides the model with some functions that control a hypothetical lighting system. Using these functions requires them to be called in a specific order. For example, you must turn the light system on before you can change the color.

While you can pass these directly to the model and let it try to call them correctly, specifying the `function_calling_config` gives you precise control over the functions that are available to the model.

Write the tools to `tools.json` so that you can reference it in later steps.


```
%%file tools.json
{
  "function_declarations": [
    {
      "name": "enable_lights",
      "description": "Turn on the lighting system.",
      "parameters": { "type": "object" }
    },
    {
      "name": "set_light_color",
      "description": "Set the light color. Lights must be enabled for this to work.",
      "parameters": {
        "type": "object",
        "properties": {
          "rgb_hex": {
            "type": "string",
            "description": "The light color as a 6-digit hex string, e.g. ff0000 for red."
          }
        },
        "required": [
          "rgb_hex"
        ]
      }
    },
    {
      "name": "stop_lights",
      "description": "Turn off the lighting system.",
      "parameters": { "type": "object" }
    }
  ]
}
```

    Overwriting tools.json
    

## Text-only mode: `NONE`

If you have provided the model with tools, but do not want to use those tools for the current conversational turn, then specify `NONE` as the mode. `NONE` tells the model not to make any function calls, and it will behave as though none have been provided.



```bash
%%bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key=$GOOGLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d @<(echo '
  {
    "system_instruction": {
      "parts": {
        "text": "You are a helpful lighting system bot. You can turn lights on and off, and you can set the color. Do not perform any other tasks."
      }
    },
    "tools": [' $(cat tools.json) '],

    "tool_config": {
      "function_calling_config": {"mode": "none"}
    },

    "contents": {
      "role": "user",
      "parts": {
        "text": "What can you do?"
      }
    }
  }
') 2>/dev/null |sed -n '/"content"/,/"finishReason"/p'
```

          "content": {
            "parts": [
              {
                "text": "As your lighting system, I can turn the lights on and off, and I can set the color of the lights. \n"
              }
            ],
            "role": "model"
          },
          "finishReason": "STOP",
    

## Automatic mode: `AUTO`

To allow the model to decide whether to respond in text or call specific functions, you can specify `AUTO` as the mode.


```bash
%%bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key=$GOOGLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d @<(echo '
  {
    "system_instruction": {
      "parts": {
        "text": "You are a helpful lighting system bot. You can turn lights on and off, and you can set the color. Do not perform any other tasks."
      }
    },
    "tools": [' $(cat tools.json) '],

    "tool_config": {
      "function_calling_config": {"mode": "auto"}
    },

    "contents": {
      "role": "user",
      "parts": {
        "text": "Light this place up!"
      }
    }
  }
') 2>/dev/null |sed -n '/"content"/,/"finishReason"/p'
```

          "content": {
            "parts": [
              {
                "functionCall": {
                  "name": "enable_lights",
                  "args": {}
                }
              }
            ],
            "role": "model"
          },
          "finishReason": "STOP",
    

## Function-calling mode: `ANY`

Setting the mode to `ANY` will force the model to make a function call. By setting `allowed_function_names`, the model will only choose from those functions. If it is not set, all of the functions in `tools` are candidates for function calling.

In this example system, if the lights are already on, then the user can change color or turn the lights off.



```bash
%%bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key=$GOOGLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d @<(echo '
  {
    "system_instruction": {
      "parts": {
        "text": "You are a helpful lighting system bot. You can turn lights on and off, and you can set the color. Do not perform any other tasks."
      }
    },
    "tools": [' $(cat tools.json) '],

    "tool_config": {
      "function_calling_config": {
        "mode": "any",
        "allowed_function_names": ["set_light_color", "stop_lights"]
      }
    },

    "contents": {
      "role": "user",
      "parts": {
        "text": "Make this place PURPLE!"
      }
    }
  }
') 2>/dev/null |sed -n '/"content"/,/"finishReason"/p'
```

          "content": {
            "parts": [
              {
                "functionCall": {
                  "name": "set_light_color",
                  "args": {
                    "rgb_hex": "9400d3"
                  }
                }
              }
            ],
            "role": "model"
          },
          "finishReason": "STOP",
    

## Further reading

Check out the [function calling recipe](./Function_calling_REST.ipynb) for more on function calling.
