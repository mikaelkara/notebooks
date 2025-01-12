# Fine tuning with function-calling


This notebook covers how to fine-tune to increase function calling accuracy and reliability. You can find more information on function calling [here](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb), and on fine tuning [here](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_finetune_chat_models.ipynb)


For context, from the function calling notebook above:

> `tools` is an optional parameter in the Chat Completion API which can be used to provide function specifications. The purpose of this is to enable models to generate function arguments which adhere to the provided specifications. Note that the API will not actually execute any function calls. It is up to developers to execute function calls using model outputs.


Function calling is a very powerful tool when it functions as intended. However, we have seen that as the number of functions increases, and the complexity of the task at hand increases, function calling becomes less accurate (e.g.: more hallucinated invocations, and incorrect invocations).

Before fine tuning for function calling, it's best to begin with:

- Improvements to the function definitions. Make them more clear, and more distinct from one another.
- Experiment with prompt engineering: often a more detailed prompt can help the model call the correct function.

_If_ the steps above fail to improve function calling to a satisfactory level, then you can try fine tuning for function calling.


### Overview


This notebook contains three sections

- **Assessing baseline function calling performance:** Evaluating an out-of-the-box `gpt-3.5-turbo` model on our given function (let's assume that for latency + cost reasons we cannot use `gpt-4o` for a drone copilot)
- **Generating synthetic data:** Using `gpt-4o` to create 'golden' set of prompts and function invocations to use as training data
- **Fine-tuning**: Running the fine tuning job, and evaluating the fine-tuned model


Note: _This notebook provides an example of how to create synthetic training data for fine tuning for function calling given just a list of functions. While real-world production test evals are preferable, this method produces strong results and can be used in conjunction with real-world training data._


# Getting baseline function calling performance



```python
#!pip install tenacity -q
#!pip install openai -q
#!pip install typing -q
# !pip install python-dotenv
```


```python
import numpy as np
import json
import os
from IPython.display import display
import pandas as pd
from openai import OpenAI
import itertools
import time
import base64
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Any, Dict, List, Generator
import ast

%load_ext dotenv
%dotenv

client = OpenAI(api_key=os.environ.get("OPENAI_BUILD_HOUR_KEY"))
```

    The dotenv extension is already loaded. To reload it, use:
      %reload_ext dotenv
    

### Utilities


Let's define utility functions for making calls to the Chat Completions API, one to get the completion and one to get the function call.



```python
def get_chat_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-3.5-turbo",
    max_tokens=500,
    temperature=0.0,
    stop=None,
    tools=None,
    seed=42,
    functions=None,
    tool_choice=None,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "tools": tools,
        "seed": seed,
        "tool_choice": tool_choice,
    }
    if functions:
        params["functions"] = functions

    completion = client.chat.completions.create(**params)
    return completion.choices[0].message, completion.usage


def eval(model: str, system_prompt: str, function_list, prompts_to_expected_tool_name):
    """
    Evaluate the performance of a model in selecting the correct function based on given prompts.

    Args:
        model (str): The name of the model to be evaluated.
        system_prompt (str): The system prompt to be used in the chat completion.
        function_list (list): A list of functions that the model can call.
        prompts_to_expected_tool_name (dict): A dictionary mapping prompts to their expected function names.

    Returns:
        None
    """

    prompts_to_actual = []
    latencies = []
    tokens_used = []

    for prompt, expected_function in prompts_to_expected_tool_name.items():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        start_time = time.time()
        completion, usage = get_chat_completion(
            model=model,
            messages=messages,
            seed=42,
            tools=function_list,
            temperature=0.0,
            tool_choice="required",
        )
        end_time = time.time()

        latency = (end_time - start_time) * 1000  # convert to milliseconds
        latencies.append(latency)

        prompts_to_actual.append(
            {prompt: completion.tool_calls[0].function.name})

        # Calculate tokens used
        tokens_used.append(usage.total_tokens)

    total_prompts = len(prompts_to_expected_tool_name)

    # Calculate the number of matches
    matches = sum(
        1
        for result in prompts_to_actual
        if list(result.values())[0]
        == prompts_to_expected_tool_name[list(result.keys())[0]]
    )
    match_percentage = (matches / total_prompts) * 100

    # Calculate average latency
    avg_latency = sum(latencies) / total_prompts
    # Calculate average tokens used
    avg_tokens_used = sum(tokens_used) / total_prompts

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(columns=["Prompt", "Expected", "Match"])

    results_list = []
    for result in prompts_to_actual:
        prompt = list(result.keys())[0]
        actual_function = list(result.values())[0]
        expected_function = prompts_to_expected_tool_name[prompt]
        match = actual_function == expected_function
        results_list.append(
            {
                "Prompt": prompt,
                "Actual": actual_function,
                "Expected": expected_function,
                "Match": "Yes" if match else "No",
            }
        )
    results_df = pd.DataFrame(results_list)

    def style_rows(row):
        match = row["Match"]
        background_color = "red" if match == "No" else "white"
        return ["background-color: {}; color: black".format(background_color)] * len(
            row
        )

    styled_results_df = results_df.style.apply(style_rows, axis=1)

    # Display the DataFrame as a table
    display(styled_results_df)

    print(
        f"Number of matches: {matches} out of {total_prompts} ({match_percentage:.2f}%)"
    )
    print(f"Average latency per request: {avg_latency:.2f} ms")
    print(f"Average tokens used per request: {avg_tokens_used:.2f}")
```

### Baseline testing


Let's build an intelligent drone co-pilot. We want to be able to give the co-pilot commands, and have it either call the function
for that command, or deny that request if the command is unfeasible.
We can first define a system prompt for the copilot.



```python
DRONE_SYSTEM_PROMPT = """You are an intelligent AI that controls a drone. Given a command or request from the user,
call one of your functions to complete the request. If the request cannot be completed by your available functions, call the reject_request function.
If the request is ambiguous or unclear, reject the request."""
```

Now let's define functions for all of the actions the copilot can take.



```python
function_list = [
    {
        "type": "function",
        "function": {
            "name": "takeoff_drone",
            "description": "Initiate the drone's takeoff sequence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "altitude": {
                        "type": "integer",
                        "description": "Specifies the altitude in meters to which the drone should ascend.",
                    }
                },
                "required": ["altitude"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "land_drone",
            "description": "Land the drone at its current location or a specified landing point.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "enum": ["current", "home_base", "custom"],
                        "description": "Specifies the landing location for the drone.",
                    },
                    "coordinates": {
                        "type": "object",
                        "description": "GPS coordinates for custom landing location. Required if location is 'custom'.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_drone_movement",
            "description": "Direct the drone's movement in a specific direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward", "left", "right", "up", "down"],
                        "description": "Direction in which the drone should move.",
                    },
                    "distance": {
                        "type": "integer",
                        "description": "Distance in meters the drone should travel in the specified direction.",
                    },
                },
                "required": ["direction", "distance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_drone_speed",
            "description": "Adjust the speed of the drone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "speed": {
                        "type": "integer",
                        "description": "Specifies the speed in km/h. Valid range is 0 to 100.",
                        "minimum": 0,
                    }
                },
                "required": ["speed"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_camera",
            "description": "Control the drone's camera to capture images or videos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["photo", "video", "panorama"],
                        "description": "Camera mode to capture content.",
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Duration in seconds for video capture. Required if mode is 'video'.",
                    },
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_gimbal",
            "description": "Adjust the drone's gimbal for camera stabilization and direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tilt": {
                        "type": "integer",
                        "description": "Tilt angle for the gimbal in degrees.",
                    },
                    "pan": {
                        "type": "integer",
                        "description": "Pan angle for the gimbal in degrees.",
                    },
                },
                "required": ["tilt", "pan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_drone_lighting",
            "description": "Control the drone's lighting for visibility and signaling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["on", "off", "blink", "sos"],
                        "description": "Lighting mode for the drone.",
                    }
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "return_to_home",
            "description": "Command the drone to return to its home or launch location.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_battery_saver_mode",
            "description": "Toggle battery saver mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "Toggle battery saver mode.",
                    }
                },
                "required": ["status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_obstacle_avoidance",
            "description": "Configure obstacle avoidance settings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "Toggle obstacle avoidance.",
                    }
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_follow_me_mode",
            "description": "Enable or disable 'follow me' mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "Toggle 'follow me' mode.",
                    }
                },
                "required": ["status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calibrate_sensors",
            "description": "Initiate calibration sequence for drone's sensors.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_autopilot",
            "description": "Enable or disable autopilot mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "Toggle autopilot mode.",
                    }
                },
                "required": ["status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "configure_led_display",
            "description": "Configure the drone's LED display pattern and colors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "enum": ["solid", "blink", "pulse", "rainbow"],
                        "description": "Pattern for the LED display.",
                    },
                    "color": {
                        "type": "string",
                        "enum": ["red", "blue", "green", "yellow", "white"],
                        "description": "Color for the LED display. Not required if pattern is 'rainbow'.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_home_location",
            "description": "Set or change the home location for the drone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinates": {
                        "type": "object",
                        "description": "GPS coordinates for the home location.",
                    }
                },
                "required": ["coordinates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reject_request",
            "description": "Use this function if the request is not possible.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
```

For starters, let's see how function calling performs with some straight forward feasible prompts, and then couple of obviously impossible requests which call the 'reject_request' function.



```python
straightforward_prompts_to_expected = {
    "Land the drone at the home base": "land_drone",
    "Take off the drone to 50 meters": "takeoff_drone",
    "Change speed to 15 kilometers per hour": "set_drone_speed",
    "Turn into an elephant!": "reject_request",
    "Move the drone forward by 10 meters": "control_drone_movement",
    "I want the LED display to blink in red": "configure_led_display",
    "Can you take a photo?": "control_camera",
    "Can you detect obstacles?": "set_obstacle_avoidance",
    "Can you dance for me?": "reject_request",
    "Can you follow me?": "set_follow_me_mode",
}
```


```python
# Evaluate the model with the given prompts
eval(
    model="gpt-3.5-turbo",
    system_prompt=DRONE_SYSTEM_PROMPT,
    function_list=function_list,
    prompts_to_expected_tool_name=straightforward_prompts_to_expected,
)
```


<style type="text/css">
#T_b01a0_row0_col0, #T_b01a0_row0_col1, #T_b01a0_row0_col2, #T_b01a0_row0_col3, #T_b01a0_row1_col0, #T_b01a0_row1_col1, #T_b01a0_row1_col2, #T_b01a0_row1_col3, #T_b01a0_row2_col0, #T_b01a0_row2_col1, #T_b01a0_row2_col2, #T_b01a0_row2_col3, #T_b01a0_row3_col0, #T_b01a0_row3_col1, #T_b01a0_row3_col2, #T_b01a0_row3_col3, #T_b01a0_row4_col0, #T_b01a0_row4_col1, #T_b01a0_row4_col2, #T_b01a0_row4_col3, #T_b01a0_row5_col0, #T_b01a0_row5_col1, #T_b01a0_row5_col2, #T_b01a0_row5_col3, #T_b01a0_row6_col0, #T_b01a0_row6_col1, #T_b01a0_row6_col2, #T_b01a0_row6_col3, #T_b01a0_row7_col0, #T_b01a0_row7_col1, #T_b01a0_row7_col2, #T_b01a0_row7_col3, #T_b01a0_row8_col0, #T_b01a0_row8_col1, #T_b01a0_row8_col2, #T_b01a0_row8_col3, #T_b01a0_row9_col0, #T_b01a0_row9_col1, #T_b01a0_row9_col2, #T_b01a0_row9_col3 {
  background-color: white;
  color: black;
}
</style>
<table id="T_b01a0">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b01a0_level0_col0" class="col_heading level0 col0" >Prompt</th>
      <th id="T_b01a0_level0_col1" class="col_heading level0 col1" >Actual</th>
      <th id="T_b01a0_level0_col2" class="col_heading level0 col2" >Expected</th>
      <th id="T_b01a0_level0_col3" class="col_heading level0 col3" >Match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b01a0_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_b01a0_row0_col0" class="data row0 col0" >Land the drone at the home base</td>
      <td id="T_b01a0_row0_col1" class="data row0 col1" >land_drone</td>
      <td id="T_b01a0_row0_col2" class="data row0 col2" >land_drone</td>
      <td id="T_b01a0_row0_col3" class="data row0 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_b01a0_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_b01a0_row1_col0" class="data row1 col0" >Take off the drone to 50 meters</td>
      <td id="T_b01a0_row1_col1" class="data row1 col1" >takeoff_drone</td>
      <td id="T_b01a0_row1_col2" class="data row1 col2" >takeoff_drone</td>
      <td id="T_b01a0_row1_col3" class="data row1 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_b01a0_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_b01a0_row2_col0" class="data row2 col0" >Change speed to 15 kilometers per hour</td>
      <td id="T_b01a0_row2_col1" class="data row2 col1" >set_drone_speed</td>
      <td id="T_b01a0_row2_col2" class="data row2 col2" >set_drone_speed</td>
      <td id="T_b01a0_row2_col3" class="data row2 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_b01a0_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_b01a0_row3_col0" class="data row3 col0" >Turn into an elephant!</td>
      <td id="T_b01a0_row3_col1" class="data row3 col1" >reject_request</td>
      <td id="T_b01a0_row3_col2" class="data row3 col2" >reject_request</td>
      <td id="T_b01a0_row3_col3" class="data row3 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_b01a0_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_b01a0_row4_col0" class="data row4 col0" >Move the drone forward by 10 meters</td>
      <td id="T_b01a0_row4_col1" class="data row4 col1" >control_drone_movement</td>
      <td id="T_b01a0_row4_col2" class="data row4 col2" >control_drone_movement</td>
      <td id="T_b01a0_row4_col3" class="data row4 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_b01a0_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_b01a0_row5_col0" class="data row5 col0" >I want the LED display to blink in red</td>
      <td id="T_b01a0_row5_col1" class="data row5 col1" >configure_led_display</td>
      <td id="T_b01a0_row5_col2" class="data row5 col2" >configure_led_display</td>
      <td id="T_b01a0_row5_col3" class="data row5 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_b01a0_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_b01a0_row6_col0" class="data row6 col0" >Can you take a photo?</td>
      <td id="T_b01a0_row6_col1" class="data row6 col1" >control_camera</td>
      <td id="T_b01a0_row6_col2" class="data row6 col2" >control_camera</td>
      <td id="T_b01a0_row6_col3" class="data row6 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_b01a0_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_b01a0_row7_col0" class="data row7 col0" >Can you detect obstacles?</td>
      <td id="T_b01a0_row7_col1" class="data row7 col1" >set_obstacle_avoidance</td>
      <td id="T_b01a0_row7_col2" class="data row7 col2" >set_obstacle_avoidance</td>
      <td id="T_b01a0_row7_col3" class="data row7 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_b01a0_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_b01a0_row8_col0" class="data row8 col0" >Can you dance for me?</td>
      <td id="T_b01a0_row8_col1" class="data row8 col1" >reject_request</td>
      <td id="T_b01a0_row8_col2" class="data row8 col2" >reject_request</td>
      <td id="T_b01a0_row8_col3" class="data row8 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_b01a0_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_b01a0_row9_col0" class="data row9 col0" >Can you follow me?</td>
      <td id="T_b01a0_row9_col1" class="data row9 col1" >set_follow_me_mode</td>
      <td id="T_b01a0_row9_col2" class="data row9 col2" >set_follow_me_mode</td>
      <td id="T_b01a0_row9_col3" class="data row9 col3" >Yes</td>
    </tr>
  </tbody>
</table>



    Number of matches: 10 out of 10 (100.00%)
    Average latency per request: 826.81 ms
    Average tokens used per request: 796.20
    

Nice! The model performs quite well with these requests. Now let's try some more difficult requests: requests that are _almost_ feasible and are drone-related, but that the drone cannot actually do, and the pilot should reject.



```python
challenging_prompts_to_expected = {
    "Play pre-recorded audio message": "reject_request",
    "Initiate following on social media": "reject_request",
    "Scan environment for heat signatures": "reject_request",
    "Bump into obstacles": "reject_request",
    "Change drone's paint job color": "reject_request",
    "Coordinate with nearby drones": "reject_request",
    "Change speed to negative 120 km/h": "reject_request",
    "Detect a person": "reject_request",
    "Please enable night vision": "reject_request",
    "Report on humidity levels around you": "reject_request",
}
```


```python
# Evaluate the model with the challenging prompts
eval(
    model="gpt-3.5-turbo",
    function_list=function_list,
    system_prompt=DRONE_SYSTEM_PROMPT,
    prompts_to_expected_tool_name=challenging_prompts_to_expected,
)
```


<style type="text/css">
#T_99c20_row0_col0, #T_99c20_row0_col1, #T_99c20_row0_col2, #T_99c20_row0_col3, #T_99c20_row2_col0, #T_99c20_row2_col1, #T_99c20_row2_col2, #T_99c20_row2_col3, #T_99c20_row4_col0, #T_99c20_row4_col1, #T_99c20_row4_col2, #T_99c20_row4_col3, #T_99c20_row5_col0, #T_99c20_row5_col1, #T_99c20_row5_col2, #T_99c20_row5_col3, #T_99c20_row7_col0, #T_99c20_row7_col1, #T_99c20_row7_col2, #T_99c20_row7_col3, #T_99c20_row9_col0, #T_99c20_row9_col1, #T_99c20_row9_col2, #T_99c20_row9_col3 {
  background-color: white;
  color: black;
}
#T_99c20_row1_col0, #T_99c20_row1_col1, #T_99c20_row1_col2, #T_99c20_row1_col3, #T_99c20_row3_col0, #T_99c20_row3_col1, #T_99c20_row3_col2, #T_99c20_row3_col3, #T_99c20_row6_col0, #T_99c20_row6_col1, #T_99c20_row6_col2, #T_99c20_row6_col3, #T_99c20_row8_col0, #T_99c20_row8_col1, #T_99c20_row8_col2, #T_99c20_row8_col3 {
  background-color: red;
  color: black;
}
</style>
<table id="T_99c20">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_99c20_level0_col0" class="col_heading level0 col0" >Prompt</th>
      <th id="T_99c20_level0_col1" class="col_heading level0 col1" >Actual</th>
      <th id="T_99c20_level0_col2" class="col_heading level0 col2" >Expected</th>
      <th id="T_99c20_level0_col3" class="col_heading level0 col3" >Match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_99c20_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_99c20_row0_col0" class="data row0 col0" >Play pre-recorded audio message</td>
      <td id="T_99c20_row0_col1" class="data row0 col1" >reject_request</td>
      <td id="T_99c20_row0_col2" class="data row0 col2" >reject_request</td>
      <td id="T_99c20_row0_col3" class="data row0 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_99c20_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_99c20_row1_col0" class="data row1 col0" >Initiate following on social media</td>
      <td id="T_99c20_row1_col1" class="data row1 col1" >set_follow_me_mode</td>
      <td id="T_99c20_row1_col2" class="data row1 col2" >reject_request</td>
      <td id="T_99c20_row1_col3" class="data row1 col3" >No</td>
    </tr>
    <tr>
      <th id="T_99c20_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_99c20_row2_col0" class="data row2 col0" >Scan environment for heat signatures</td>
      <td id="T_99c20_row2_col1" class="data row2 col1" >reject_request</td>
      <td id="T_99c20_row2_col2" class="data row2 col2" >reject_request</td>
      <td id="T_99c20_row2_col3" class="data row2 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_99c20_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_99c20_row3_col0" class="data row3 col0" >Bump into obstacles</td>
      <td id="T_99c20_row3_col1" class="data row3 col1" >set_obstacle_avoidance</td>
      <td id="T_99c20_row3_col2" class="data row3 col2" >reject_request</td>
      <td id="T_99c20_row3_col3" class="data row3 col3" >No</td>
    </tr>
    <tr>
      <th id="T_99c20_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_99c20_row4_col0" class="data row4 col0" >Change drone's paint job color</td>
      <td id="T_99c20_row4_col1" class="data row4 col1" >reject_request</td>
      <td id="T_99c20_row4_col2" class="data row4 col2" >reject_request</td>
      <td id="T_99c20_row4_col3" class="data row4 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_99c20_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_99c20_row5_col0" class="data row5 col0" >Coordinate with nearby drones</td>
      <td id="T_99c20_row5_col1" class="data row5 col1" >reject_request</td>
      <td id="T_99c20_row5_col2" class="data row5 col2" >reject_request</td>
      <td id="T_99c20_row5_col3" class="data row5 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_99c20_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_99c20_row6_col0" class="data row6 col0" >Change speed to negative 120 km/h</td>
      <td id="T_99c20_row6_col1" class="data row6 col1" >set_drone_speed</td>
      <td id="T_99c20_row6_col2" class="data row6 col2" >reject_request</td>
      <td id="T_99c20_row6_col3" class="data row6 col3" >No</td>
    </tr>
    <tr>
      <th id="T_99c20_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_99c20_row7_col0" class="data row7 col0" >Detect a person</td>
      <td id="T_99c20_row7_col1" class="data row7 col1" >reject_request</td>
      <td id="T_99c20_row7_col2" class="data row7 col2" >reject_request</td>
      <td id="T_99c20_row7_col3" class="data row7 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_99c20_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_99c20_row8_col0" class="data row8 col0" >Please enable night vision</td>
      <td id="T_99c20_row8_col1" class="data row8 col1" >set_drone_lighting</td>
      <td id="T_99c20_row8_col2" class="data row8 col2" >reject_request</td>
      <td id="T_99c20_row8_col3" class="data row8 col3" >No</td>
    </tr>
    <tr>
      <th id="T_99c20_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_99c20_row9_col0" class="data row9 col0" >Report on humidity levels around you</td>
      <td id="T_99c20_row9_col1" class="data row9 col1" >reject_request</td>
      <td id="T_99c20_row9_col2" class="data row9 col2" >reject_request</td>
      <td id="T_99c20_row9_col3" class="data row9 col3" >Yes</td>
    </tr>
  </tbody>
</table>



    Number of matches: 6 out of 10 (60.00%)
    Average latency per request: 610.26 ms
    Average tokens used per request: 791.90
    

Now we run into some problems.
The model here should reject all of these requests, as they are impossible/conflicting/ambiguous given the functions, however instead the model calls functions that are somewhat related to the request, but incorrect. For example, the model sets follow_me_mode when asked to initiate following on social media.

<br>
In this simple case, more prompt engineering may resolve some of these issues, but for the purpose of this example we will demonstrate how fine tuning can be used to improve performance. Additionally, while this case is relatively straightforward, as the number of and complexity of the functions increases, fine tuning becomes more and more impactful.

Again, our goal here is to improve performance and use less tokens, so fine-tuning allows us to:

- Omit function and parameter descriptions: remove the description field from function and parameters
- Omit parameters: remove the entire properties field from the parameters object
- Omit function entirely: remove the entire function object from the functions array


# Generating synthetic data


### Helper functions


We want to generate every invocation of every function, so that we have
full coverage of all potential invocations to create synthetic data for. Then, we will use `gpt-4o` to come up with prompts that would call each invocation, and we will use that prompt - function invocation pair as training data.


Generating every invocation for a function with fixed enums is more simple, but for a function such as
`control_gimbal` we need to set the `tilt` and `pan` integer values, so to generate those synthetic invocations we will first set a placeholder, and then later use `gpt-4o` to come up with reasonable values.



```python
placeholder_int = "fill_in_int"
placeholder_string = "fill_in_string"
```

The functions below take in all the functions from the function list, and look
at all the potential invocations of those functions given each function's parameters.
The functions also account for `required` parameters, so that all the invocations
are actually feasible.



```python
def generate_permutations(
    params: Dict[str, Dict[str, Any]]
) -> Generator[Dict[str, Any], None, None]:
    """
    Generates all possible permutations for given parameters.

    :param params: Parameter dictionary containing required and optional fields.
    :return: A generator yielding each permutation.
    """

    # Extract the required fields from the parameters
    required_fields = params.get("required", [])

    # Generate permutations for required fields
    required_permutations = generate_required_permutations(params, required_fields)

    # Generate optional permutations based on each required permutation
    for required_perm in required_permutations:
        yield from generate_optional_permutations(params, required_perm)


def generate_required_permutations(
    params: Dict[str, Dict[str, Any]], required_fields: List[str]
) -> List[Dict[str, Any]]:
    """
    Generates permutations for the required fields.

    :param params: Parameter dictionary.
    :param required_fields: List of required fields.
    :return: A list of permutations for required fields.
    """

    # Get all possible values for each required field
    required_values = [get_possible_values(params, field) for field in required_fields]

    # Generate permutations from possible values
    return [
        dict(zip(required_fields, values))
        for values in itertools.product(*required_values)
    ]


def generate_optional_permutations(
    params: Dict[str, Dict[str, Any]], base_perm: Dict[str, Any]
) -> Generator[Dict[str, Any], None, None]:
    """
    Generates permutations for optional fields based on a base permutation.

    :param params: Parameter dictionary.
    :param base_perm: Base permutation dictionary.
    :return: A generator yielding each permutation for optional fields.
    """

    # Determine the fields that are optional by subtracting the base permutation's fields from all properties
    optional_fields = set(params["properties"]) - set(base_perm)

    # Iterate through all combinations of optional fields
    for field_subset in itertools.chain.from_iterable(
        itertools.combinations(optional_fields, r)
        for r in range(len(optional_fields) + 1)
    ):

        # Generate product of possible values for the current subset of fields
        for values in itertools.product(
            *(get_possible_values(params, field) for field in field_subset)
        ):

            # Create a new permutation by combining base permutation and current field values
            new_perm = {**base_perm, **dict(zip(field_subset, values))}

            yield new_perm


def get_possible_values(params: Dict[str, Dict[str, Any]], field: str) -> List[Any]:
    """
    Retrieves possible values for a given field.

    :param params: Parameter dictionary.
    :param field: The field for which to get possible values.
    :return: A list of possible values.
    """

    # Extract field information from the parameters
    field_info = params["properties"][field]

    # Based on the field's type or presence of 'enum', determine and return the possible values
    if "enum" in field_info:
        return field_info["enum"]
    elif field_info["type"] == "integer":
        return [placeholder_int]
    elif field_info["type"] == "string":
        return [placeholder_string]
    elif field_info["type"] == "boolean":
        return [True, False]
    elif field_info["type"] == "array" and "enum" in field_info["items"]:
        enum_values = field_info["items"]["enum"]
        all_combinations = [
            list(combo)
            for i in range(1, len(enum_values) + 1)
            for combo in itertools.combinations(enum_values, i)
        ]
        return all_combinations
    return []
```

### Let's generate every invocation for every function first


Prompts:



```python
INVOCATION_FILLER_PROMPT = """
1) Input reasonable values for 'fill_in_string' and 'fill_in_int' in the invocation here: {invocation}. Reasonable values are determined by the function definition. Use the
the entire function provided here :{function} to get context over what proper fill_in_string and fill_in_int values would be.
Example:

Input: invocation: {{
    "name": "control_camera",
    "arguments": {{
      "mode":"video",
      "duration":"fill_in_int"
    }}
}},
function:{function}

Output: invocation: {{
    "name": "control_camera",
    "arguments": {{
      "mode":"video",
      "duration": 30
    }}
}}


MAKE SURE output is just a dictionary with keys 'name' and 'arguments', no other text or response.

Input: {invocation}
Output:
"""


COMMAND_GENERATION_PROMPT = """
You are to output 2 commands, questions or statements that would generate the inputted function and parameters.
Please make the commands or questions natural, as a person would ask, and the command or questions should be varied and not repetitive.
It should not always mirror the exact technical terminology used in the function and parameters, rather reflect a conversational and intuitive request.
For instance, the prompt should not be 'turn on the dome light', as that is too technical, but rather 'turn on the inside lights'.
Another example, is the prompt should not be 'turn on the HVAC', but rather 'turn on the air conditioning'. Use language a normal driver would use, even if
it is technically incorrect but colloquially used.

RULES: ALWAYS put a backwards slash before an apostrophe or single quote '. For example, do not say don't but say don\'t.
Prompts MUST be in double quotes as well.

Example

Input: {{'name': 'calibrate_sensors','arguments': {{}}'' }}
Prompt: ["The sensors are out of whack, can you reset them", "The calibration of the drone is off, fix it please!"]

Input: {{'name': 'set_autopilot','arguments': {{'status': 'off'}}}}
Prompt: ["OK, I want to take back pilot control now","Turn off the automatic pilot I'm ready control it"]

Input: {invocation}
Prompt:
"""
```

In the below snippet, we generate the invocation of each function except for the `reject_request` function.

To perform effective fine-tuning we need correctly labeled data. We could manually come up with examples and label the data,\
or we can generate synthetic data with the help of `gpt-4o` <br>

Empirically, `gpt-4o` needs a bit more help to get good realistic examples of prompts that would generate the `reject_request` function, so we'll do that next...



```python
input_objects = []
all_but_reject = [f for f in function_list if f.get("name") != "reject_request"]

for function in all_but_reject:
    func_name = function["function"]["name"]
    params = function["function"]["parameters"]
    for arguments in generate_permutations(params):
        if any(val in arguments.values() for val in ["fill_in_int", "fill_in_str"]):
            input_object = {"name": func_name, "arguments": arguments}
            messages = [
                {
                    "role": "user",
                    "content": INVOCATION_FILLER_PROMPT.format(
                        invocation=str(input_object), function=function
                    ),
                }
            ]
            input_object, usage = get_chat_completion(
                model="gpt-4o", messages=messages, max_tokens=200, temperature=0.1
            ).content
        else:
            input_object = {"name": func_name, "arguments": arguments}

        input_objects.append(input_object)
```

Now that we have all the invocations, let's use `gpt-4o` to generate prompts that would result in those invocations



```python
def remove_sequences(input_string):
    # Replace the specific sequences with an empty string
    cleaned_string = input_string.replace("```json", "")  # Remove "```json" first
    cleaned_string = cleaned_string.replace("```", "")  # Then remove "```"
    return json.loads(cleaned_string)
```


```python
def create_commands(invocation_list):
    example_list = []
    for i, invocation in enumerate(invocation_list):
        if i < 100:
            print(
                f"\033[34m{np.round(100*i/len(invocation_list),1)}% complete\033[0m")
            if type(invocation) == str or "json" in invocation:
                invocation = remove_sequences(invocation)
            print(invocation)

        # Format the prompt with the invocation string
        request_prompt = COMMAND_GENERATION_PROMPT.format(
            invocation=invocation)

        messages = [{"role": "user", "content": f"{request_prompt}"}]
        completion, usage = get_chat_completion(messages, temperature=0.8)
        command_dict = {"Input": invocation, "Prompt": completion.content}
        example_list.append(command_dict)
    return example_list
```


```python
# Only printing the first 10 rows
training_examples_unformatted = create_commands(input_objects)
```

    [34m0.0% complete[0m
    {'name': 'takeoff_drone', 'arguments': {'altitude': 100}}
    [34m1.8% complete[0m
    {'name': 'land_drone', 'arguments': {'location': 'current'}}
    [34m3.5% complete[0m
    {'name': 'land_drone', 'arguments': {'location': 'home_base'}}
    [34m5.3% complete[0m
    {'name': 'land_drone', 'arguments': {'location': 'custom'}}
    [34m7.0% complete[0m
    {'name': 'control_drone_movement', 'arguments': {'direction': 'forward', 'distance': 100}}
    [34m8.8% complete[0m
    {'name': 'control_drone_movement', 'arguments': {'direction': 'backward', 'distance': 50}}
    [34m10.5% complete[0m
    {'name': 'control_drone_movement', 'arguments': {'direction': 'left', 'distance': 10}}
    [34m12.3% complete[0m
    {'name': 'control_drone_movement', 'arguments': {'direction': 'right', 'distance': 10}}
    [34m14.0% complete[0m
    {'name': 'control_drone_movement', 'arguments': {'direction': 'up', 'distance': 10}}
    [34m15.8% complete[0m
    {'name': 'control_drone_movement', 'arguments': {'direction': 'down', 'distance': 10}}
    [34m17.5% complete[0m
    {'name': 'set_drone_speed', 'arguments': {'speed': 10}}
    [34m19.3% complete[0m
    {'name': 'control_camera', 'arguments': {'mode': 'photo'}}
    [34m21.1% complete[0m
    {'name': 'control_camera', 'arguments': {'mode': 'photo', 'duration': 10}}
    [34m22.8% complete[0m
    {'name': 'control_camera', 'arguments': {'mode': 'video'}}
    [34m24.6% complete[0m
    {'name': 'control_camera', 'arguments': {'mode': 'video', 'duration': 60}}
    [34m26.3% complete[0m
    {'name': 'control_camera', 'arguments': {'mode': 'panorama'}}
    [34m28.1% complete[0m
    {'name': 'control_camera', 'arguments': {'mode': 'panorama', 'duration': 60}}
    [34m29.8% complete[0m
    {'name': 'control_gimbal', 'arguments': {'tilt': 45, 'pan': 90}}
    [34m31.6% complete[0m
    {'name': 'set_drone_lighting', 'arguments': {'mode': 'on'}}
    [34m33.3% complete[0m
    {'name': 'set_drone_lighting', 'arguments': {'mode': 'off'}}
    [34m35.1% complete[0m
    {'name': 'set_drone_lighting', 'arguments': {'mode': 'blink'}}
    [34m36.8% complete[0m
    {'name': 'set_drone_lighting', 'arguments': {'mode': 'sos'}}
    [34m38.6% complete[0m
    {'name': 'return_to_home', 'arguments': {}}
    [34m40.4% complete[0m
    {'name': 'set_battery_saver_mode', 'arguments': {'status': 'on'}}
    [34m42.1% complete[0m
    {'name': 'set_battery_saver_mode', 'arguments': {'status': 'off'}}
    [34m43.9% complete[0m
    {'name': 'set_obstacle_avoidance', 'arguments': {'mode': 'on'}}
    [34m45.6% complete[0m
    {'name': 'set_obstacle_avoidance', 'arguments': {'mode': 'off'}}
    [34m47.4% complete[0m
    {'name': 'set_follow_me_mode', 'arguments': {'status': 'on'}}
    [34m49.1% complete[0m
    {'name': 'set_follow_me_mode', 'arguments': {'status': 'off'}}
    [34m50.9% complete[0m
    {'name': 'calibrate_sensors', 'arguments': {}}
    [34m52.6% complete[0m
    {'name': 'set_autopilot', 'arguments': {'status': 'on'}}
    [34m54.4% complete[0m
    {'name': 'set_autopilot', 'arguments': {'status': 'off'}}
    [34m56.1% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'solid'}}
    [34m57.9% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'solid', 'color': 'red'}}
    [34m59.6% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'solid', 'color': 'blue'}}
    [34m61.4% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'solid', 'color': 'green'}}
    [34m63.2% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'solid', 'color': 'yellow'}}
    [34m64.9% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'solid', 'color': 'white'}}
    [34m66.7% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'blink'}}
    [34m68.4% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'blink', 'color': 'red'}}
    [34m70.2% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'blink', 'color': 'blue'}}
    [34m71.9% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'blink', 'color': 'green'}}
    [34m73.7% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'blink', 'color': 'yellow'}}
    [34m75.4% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'blink', 'color': 'white'}}
    [34m77.2% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'pulse'}}
    [34m78.9% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'pulse', 'color': 'red'}}
    [34m80.7% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'pulse', 'color': 'blue'}}
    [34m82.5% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'pulse', 'color': 'green'}}
    [34m84.2% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'pulse', 'color': 'yellow'}}
    [34m86.0% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'pulse', 'color': 'white'}}
    [34m87.7% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'rainbow'}}
    [34m89.5% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'rainbow', 'color': 'red'}}
    [34m91.2% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'rainbow', 'color': 'blue'}}
    [34m93.0% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'rainbow', 'color': 'green'}}
    [34m94.7% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'rainbow', 'color': 'yellow'}}
    [34m96.5% complete[0m
    {'name': 'configure_led_display', 'arguments': {'pattern': 'rainbow', 'color': 'white'}}
    [34m98.2% complete[0m
    {'name': 'reject_request', 'arguments': {}}
    

Now let's format the training examples properly. For more documentation on the proper training data formatting for fine tuning for function calling, see here: https://platform.openai.com/docs/guides/fine-tuning/fine-tuning-examples



```python
def remove_descriptions(function_list):
    for function in function_list:
        func = function["function"]
        if "description" in func:
            del func["description"]

        params = func["parameters"]
        if "properties" in params:
            for param in params["properties"].values():
                if "description" in param:
                    del param["description"]

    return function_list


modified_function_list = remove_descriptions(function_list)
```


```python
training_examples = []

for prompt in training_examples_unformatted:
    # adjust formatting for training data specs

    # if its not a dict, convert to dict
    if type(prompt["Input"]) != dict:
        prompt["Input"] = ast.literal_eval(prompt["Input"])
    prompt["Input"]["arguments"] = json.dumps(prompt["Input"]["arguments"])
    try:
        prompt["Prompt"] = json.loads(prompt["Prompt"])
    except:
        continue
    for p in prompt["Prompt"]:
        print(p)
        print(prompt["Input"])
        tool_calls = [
            {"id": "call_id", "type": "function", "function": prompt["Input"]}
        ]
        training_examples.append(
            {
                "messages": [
                    {"role": "system", "content": DRONE_SYSTEM_PROMPT},
                    {"role": "user", "content": p},
                    {"role": "assistant", "tool_calls": tool_calls},
                ],
                "parallel_tool_calls": False,
                "tools": modified_function_list,
            }
        )
```

    Let's get the drone in the air, how high should it go?
    {'name': 'takeoff_drone', 'arguments': '{"altitude": 100}'}
    Ready for takeoff, how high should the drone fly?
    {'name': 'takeoff_drone', 'arguments': '{"altitude": 100}'}
    Can you bring the drone down to where we are?
    {'name': 'land_drone', 'arguments': '{"location": "current"}'}
    Let's get the drone to land right here
    {'name': 'land_drone', 'arguments': '{"location": "current"}'}
    Bring the drone back to base for landing
    {'name': 'land_drone', 'arguments': '{"location": "home_base"}'}
    Can you safely land the drone at home base
    {'name': 'land_drone', 'arguments': '{"location": "home_base"}'}
    Can you make the drone move to the left by 10 units?
    {'name': 'control_drone_movement', 'arguments': '{"direction": "left", "distance": 10}'}
    I need the drone to go left, could you move it 10 steps that way?
    {'name': 'control_drone_movement', 'arguments': '{"direction": "left", "distance": 10}'}
    Can you move the drone to the right by 10 feet?
    {'name': 'control_drone_movement', 'arguments': '{"direction": "right", "distance": 10}'}
    I need the drone to go 10 feet to the right, can you do that?
    {'name': 'control_drone_movement', 'arguments': '{"direction": "right", "distance": 10}'}
    Can you make the drone go upwards by 10 units?
    {'name': 'control_drone_movement', 'arguments': '{"direction": "up", "distance": 10}'}
    I need the drone to move up, can you do that for me?
    {'name': 'control_drone_movement', 'arguments': '{"direction": "up", "distance": 10}'}
    Can you bring the drone lower by 10 feet please?
    {'name': 'control_drone_movement', 'arguments': '{"direction": "down", "distance": 10}'}
    I need the drone to descend 10 units, can you make that happen?
    {'name': 'control_drone_movement', 'arguments': '{"direction": "down", "distance": 10}'}
    Can you make the drone go faster?
    {'name': 'set_drone_speed', 'arguments': '{"speed": 10}'}
    I think the drone should speed up a bit, don't you think?
    {'name': 'set_drone_speed', 'arguments': '{"speed": 10}'}
    I want to take a picture, can you switch the camera mode to photo
    {'name': 'control_camera', 'arguments': '{"mode": "photo"}'}
    Let's capture this moment, switch the camera to photo mode please
    {'name': 'control_camera', 'arguments': '{"mode": "photo"}'}
    Can you switch the camera to photo mode and take a picture for 10 seconds?
    {'name': 'control_camera', 'arguments': '{"mode": "photo", "duration": 10}'}
    I need to capture something, can you set the camera to take photos for 10 seconds?
    {'name': 'control_camera', 'arguments': '{"mode": "photo", "duration": 10}'}
    Can you switch the camera to video mode?
    {'name': 'control_camera', 'arguments': '{"mode": "video"}'}
    I want to record, can you set the camera to video mode?
    {'name': 'control_camera', 'arguments': '{"mode": "video"}'}
    Can you start recording a video with the camera for a minute
    {'name': 'control_camera', 'arguments': '{"mode": "video", "duration": 60}'}
    I need to film something, can you put the camera in video mode for 60 seconds
    {'name': 'control_camera', 'arguments': '{"mode": "video", "duration": 60}'}
    Can you switch the camera to panorama mode?
    {'name': 'control_camera', 'arguments': '{"mode": "panorama"}'}
    I'd like to take a 360-degree photo, can you set the camera to panorama mode?
    {'name': 'control_camera', 'arguments': '{"mode": "panorama"}'}
    Can you set the camera to take a panorama shot for a minute
    {'name': 'control_camera', 'arguments': '{"mode": "panorama", "duration": 60}'}
    I'd like to switch the camera mode to panorama and have it last for a minute
    {'name': 'control_camera', 'arguments': '{"mode": "panorama", "duration": 60}'}
    Can you adjust the camera angle up and to the right?
    {'name': 'control_gimbal', 'arguments': '{"tilt": 45, "pan": 90}'}
    I need to tilt the camera up and pan it to the right, can you do that?
    {'name': 'control_gimbal', 'arguments': '{"tilt": 45, "pan": 90}'}
    Can you turn on the lights for the drone
    {'name': 'set_drone_lighting', 'arguments': '{"mode": "on"}'}
    I need some extra light, can you activate it on the drone
    {'name': 'set_drone_lighting', 'arguments': '{"mode": "on"}'}
    Can you turn off the lights on the drone
    {'name': 'set_drone_lighting', 'arguments': '{"mode": "off"}'}
    I don't need the drone lights on, can you switch them off
    {'name': 'set_drone_lighting', 'arguments': '{"mode": "off"}'}
    Can you make the drone lights flash?
    {'name': 'set_drone_lighting', 'arguments': '{"mode": "blink"}'}
    I want the drone lights to blink, can you do that?
    {'name': 'set_drone_lighting', 'arguments': '{"mode": "blink"}'}
    Can you switch the drone lights to the SOS mode, just in case?
    {'name': 'set_drone_lighting', 'arguments': '{"mode": "sos"}'}
    I need the drone lights to flash SOS, can you set that up?
    {'name': 'set_drone_lighting', 'arguments': '{"mode": "sos"}'}
    Can you bring the drone back home now?
    {'name': 'return_to_home', 'arguments': '{}'}
    Is it time for the drone to return to base?
    {'name': 'return_to_home', 'arguments': '{}'}
    My phone battery is draining so fast, can you turn on battery saver mode
    {'name': 'set_battery_saver_mode', 'arguments': '{"status": "on"}'}
    I need my laptop battery to last longer, can you switch on battery saver mode
    {'name': 'set_battery_saver_mode', 'arguments': '{"status": "on"}'}
    My phone battery is draining too quickly, can you turn off the battery saver mode
    {'name': 'set_battery_saver_mode', 'arguments': '{"status": "off"}'}
    I feel like my device is slower with battery saver on, can we turn it off?
    {'name': 'set_battery_saver_mode', 'arguments': '{"status": "off"}'}
    I want the car to avoid obstacles, can you turn on that feature?
    {'name': 'set_obstacle_avoidance', 'arguments': '{"mode": "on"}'}
    Can you activate the obstacle avoidance mode for safety purposes?
    {'name': 'set_obstacle_avoidance', 'arguments': '{"mode": "on"}'}
    I'd like to turn off obstacle detection, how do I do that?
    {'name': 'set_obstacle_avoidance', 'arguments': '{"mode": "off"}'}
    Can you disable the obstacle avoidance feature for now?
    {'name': 'set_obstacle_avoidance', 'arguments': '{"mode": "off"}'}
    Can you activate the follow me mode?
    {'name': 'set_follow_me_mode', 'arguments': '{"status": "on"}'}
    I want the car to follow me, can you turn on that feature?
    {'name': 'set_follow_me_mode', 'arguments': '{"status": "on"}'}
    I don't want the drone following me anymore, can you turn that off?
    {'name': 'set_follow_me_mode', 'arguments': '{"status": "off"}'}
    Can you disable the follow-me mode on the drone?
    {'name': 'set_follow_me_mode', 'arguments': '{"status": "off"}'}
    The sensors are acting up, can you recalibrate them
    {'name': 'calibrate_sensors', 'arguments': '{}'}
    My device doesn't seem to be sensing correctly, can you adjust it
    {'name': 'calibrate_sensors', 'arguments': '{}'}
    I'm too tired to drive, can you turn on the autopilot
    {'name': 'set_autopilot', 'arguments': '{"status": "on"}'}
    Let the car drive itself, turn on autopilot
    {'name': 'set_autopilot', 'arguments': '{"status": "on"}'}
    I'm feeling more confident, turn off the autopilot
    {'name': 'set_autopilot', 'arguments': '{"status": "off"}'}
    I think I can handle it, deactivate the automatic pilot
    {'name': 'set_autopilot', 'arguments': '{"status": "off"}'}
    Can you set the display to a steady yellow color?
    {'name': 'configure_led_display', 'arguments': '{"pattern": "solid", "color": "yellow"}'}
    I'd like the LED display to be a solid yellow, please.
    {'name': 'configure_led_display', 'arguments': '{"pattern": "solid", "color": "yellow"}'}
    Can you make the lights flash on and off
    {'name': 'configure_led_display', 'arguments': '{"pattern": "blink"}'}
    I want the LED display to blink, can you set that up
    {'name': 'configure_led_display', 'arguments': '{"pattern": "blink"}'}
    Can you make the lights flash in red?
    {'name': 'configure_led_display', 'arguments': '{"pattern": "blink", "color": "red"}'}
    How do I set the display to blink in red?
    {'name': 'configure_led_display', 'arguments': '{"pattern": "blink", "color": "red"}'}
    Can you make the lights flash in yellow?
    {'name': 'configure_led_display', 'arguments': '{"pattern": "blink", "color": "yellow"}'}
    How do I set the display to blink in yellow?
    {'name': 'configure_led_display', 'arguments': '{"pattern": "blink", "color": "yellow"}'}
    Can you make the lights blink instead of staying steady
    {'name': 'configure_led_display', 'arguments': '{"pattern": "pulse"}'}
    I want the LEDs to flash, not stay solid
    {'name': 'configure_led_display', 'arguments': '{"pattern": "pulse"}'}
    Can you make the LED display pulse in red, please?
    {'name': 'configure_led_display', 'arguments': '{"pattern": "pulse", "color": "red"}'}
    I'd like the LED display to flash in red, can you set that up?
    {'name': 'configure_led_display', 'arguments': '{"pattern": "pulse", "color": "red"}'}
    I want the LED lights to flash in blue
    {'name': 'configure_led_display', 'arguments': '{"pattern": "pulse", "color": "blue"}'}
    Can you set the display to pulse with a blue color
    {'name': 'configure_led_display', 'arguments': '{"pattern": "pulse", "color": "blue"}'}
    Can you make the lights flash and change to green
    {'name': 'configure_led_display', 'arguments': '{"pattern": "pulse", "color": "green"}'}
    Let's set the LEDs to blink and switch to green
    {'name': 'configure_led_display', 'arguments': '{"pattern": "pulse", "color": "green"}'}
    Can you change the flashy lights to yellow and make them pulse
    {'name': 'configure_led_display', 'arguments': '{"pattern": "pulse", "color": "yellow"}'}
    I want the LED display to blink in yellow, can you do that
    {'name': 'configure_led_display', 'arguments': '{"pattern": "pulse", "color": "yellow"}'}
    Can you change the colors on the display to red and set it to a rainbow pattern?
    {'name': 'configure_led_display', 'arguments': '{"pattern": "rainbow", "color": "red"}'}
    I want the LED display to show a rainbow pattern in red, can you set that up?
    {'name': 'configure_led_display', 'arguments': '{"pattern": "rainbow", "color": "red"}'}
    Can you change the color and pattern of the lights to blue and rainbow?
    {'name': 'configure_led_display', 'arguments': '{"pattern": "rainbow", "color": "blue"}'}
    I'm feeling like some colorful lights, can you set it to blue and rainbow?
    {'name': 'configure_led_display', 'arguments': '{"pattern": "rainbow", "color": "blue"}'}
    Can you set the LED display to show a rainbow pattern in green color?
    {'name': 'configure_led_display', 'arguments': '{"pattern": "rainbow", "color": "green"}'}
    I'd like the LED display to cycle through colors, starting with green
    {'name': 'configure_led_display', 'arguments': '{"pattern": "rainbow", "color": "green"}'}
    Can you make the lights do a cool rainbow effect
    {'name': 'configure_led_display', 'arguments': '{"pattern": "rainbow", "color": "white"}'}
    Change the color of the lights to white and make them change like a rainbow
    {'name': 'configure_led_display', 'arguments': '{"pattern": "rainbow", "color": "white"}'}
    I changed my mind, can you cancel that request
    {'name': 'reject_request', 'arguments': '{}'}
    I don't want to proceed with the request anymore, can you reject it
    {'name': 'reject_request', 'arguments': '{}'}
    

Now, back to the rejection function. Let's generate some prompts that are _nearly_ possible, but should result in the `reject_request` function being called. To do so, we queried `gpt-4o` asking for requests that are related to, but not quite possible with, the given list of functions.



```python
reject_list = [
    "Translate broadcast message to another language",
    "Automatically capture photos when face is detected",
    "Detect nearby drones",
    "Measure wind resistance",
    "Capture slow motion video",
    "Move the drone forward and backward by same distance at the same time.",
    "Adjust drone's altitude to ground level changes",
    "Display custom message on LED display",
    "Sync drone's time with smartphone",
    "Alert when drone travels out of designated area",
    "Calibrate sensors and land simultaneously",
    "Detect moisture levels",
    "Automatically follow GPS tagged object",
    "Toggle night vision mode",
    "Maintain current altitude when battery is low",
    "Decide best landing spot using AI",
    "Program drone's route based on wind direction",
]
```


```python
reject_training_list = []
for prompt in reject_list:
    # Adjust formatting
    tool_calls = [
        {
            "id": "call_id",
            "type": "function",
            "function": {"name": "reject_request", "arguments": "{}"},
        }
    ]
    reject_training_list.append(
        {
            "messages": [
                {"role": "system", "content": DRONE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "tool_calls": tool_calls},
            ],
            "parallel_tool_calls": False,
            "tools": modified_function_list,
        }
    )
```

Now combine all the training examples together



```python
training_list_total = training_examples + reject_training_list
```


```python
training_file = "data/drone_training.jsonl"
with open(training_file, "w") as f:
    for item in training_list_total:
        json_str = json.dumps(item)
        f.write(f"{json_str}\n")
```

# Fine tuning


Finally, we can kick off the fine-tuning job



```python
# Upload the training file
file = client.files.create(
    file=open("data/drone_training.jsonl", "rb"),
    purpose="fine-tune",
)
file_id = file.id
print(f"FileID: {file_id}")

# Create a fine-tuning job

ft = client.fine_tuning.jobs.create(
    model="gpt-3.5-turbo",
    training_file=file_id,
    suffix="drone",
)

print(f"Fine-tuning job created: {ft}")
```

    FileID: file-blg0IytwIivZQzc9mbfnS8Pm
    Fine-tuning job created: FineTuningJob(id='ftjob-84PQg97hoIAKf21IPnhiNlU1', created_at=1718580285, error=Error(code=None, message=None, param=None), fine_tuned_model=None, finished_at=None, hyperparameters=Hyperparameters(n_epochs='auto', batch_size='auto', learning_rate_multiplier='auto'), model='gpt-3.5-turbo-0125', object='fine_tuning.job', organization_id='org-lb41cclBdkq5pm6BgDhx8DHP', result_files=[], seed=1513865891, status='validating_files', trained_tokens=None, training_file='file-blg0IytwIivZQzc9mbfnS8Pm', validation_file=None, estimated_finish=None, integrations=[], user_provided_suffix='drone')
    

In addition to creating a fine-tuning job, you can also list existing jobs, retrieve the status of a job, or cancel a job.



```python
ftjob_id = "ftjob-84PQg97hoIAKf21IPnhiNlU1"
# List 10 fine-tuning jobs
# client.fine_tuning.jobs.list(limit=10)

# Retrieve the state of a fine-tune
client.fine_tuning.jobs.retrieve(ftjob_id)

# Cancel a job
# client.fine_tuning.jobs.cancel("ftjob-abc123")

# List up to 10 events from a fine-tuning job
# client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-abc123", limit=10)

# Delete a fine-tuned model (must be an owner of the org the model was created in)
# client.models.delete("ft:gpt-3.5-turbo:abc:suffix:abc123")
```




    FineTuningJob(id='ftjob-84PQg97hoIAKf21IPnhiNlU1', created_at=1718580285, error=Error(code=None, message=None, param=None), fine_tuned_model='ft:gpt-3.5-turbo-0125:openai-gtm:drone:9atiPjeC', finished_at=1718581004, hyperparameters=Hyperparameters(n_epochs=3, batch_size=1, learning_rate_multiplier=2), model='gpt-3.5-turbo-0125', object='fine_tuning.job', organization_id='org-lb41cclBdkq5pm6BgDhx8DHP', result_files=['file-F6XPJFLVG9f3mR04KBmwUI9H'], seed=1513865891, status='succeeded', trained_tokens=145983, training_file='file-blg0IytwIivZQzc9mbfnS8Pm', validation_file=None, estimated_finish=None, integrations=[], user_provided_suffix='drone')



After a fine-tuning job has finished, you can also see metrics around how the training process went by querying a fine-tuning job, extracting a file ID from the result_files, and then retrieving that files content. Each results CSV file has the following columns: step, train_loss, train_accuracy, valid_loss, and valid_mean_token_accuracy. While metrics can he helpful, evaluating samples from the fine-tuned model provides the most relevant sense of model quality.



```python
fine_tune_results = client.fine_tuning.jobs.retrieve(ftjob_id).result_files
result_file_id = client.files.retrieve(fine_tune_results[0]).id

# Retrieve the result file
result_file = client.files.content(file_id=result_file_id)
decoded_content = base64.b64decode(result_file.read()).decode("utf-8")
print(decoded_content)
```

    step,train_loss,train_accuracy,valid_loss,valid_mean_token_accuracy
    1,3.63265,0.5,,
    2,2.45992,0.80952,,
    3,2.77939,0.80952,,
    4,3.53073,0.65,,
    5,2.61654,0.8,,
    6,2.16,0.85714,,
    7,2.73706,0.8,,
    8,2.56944,0.625,,
    9,2.06096,0.78947,,
    10,1.69598,0.8,,
    11,1.94268,0.77778,,
    12,1.61752,0.86667,,
    13,1.2442,0.8,,
    14,0.73411,0.875,,
    15,0.34285,0.875,,
    16,0.22229,0.95238,,
    17,0.04635,0.95,,
    18,0.00626,1.0,,
    19,0.60888,0.90909,,
    20,0.00092,1.0,,
    21,0.8001,0.95,,
    22,0.04982,1.0,,
    23,0.35494,0.92857,,
    24,0.00023,1.0,,
    25,0.00034,1.0,,
    26,0.0029,1.0,,
    27,0.58017,0.875,,
    28,0.13018,0.9375,,
    29,0.00109,1.0,,
    30,6e-05,1.0,,
    31,0.61665,0.95,,
    32,3e-05,1.0,,
    33,0.23598,0.95,,
    34,3e-05,1.0,,
    35,0.03566,1.0,,
    36,1e-05,1.0,,
    37,1e-05,1.0,,
    38,2e-05,1.0,,
    39,2e-05,1.0,,
    40,0.00034,1.0,,
    41,0.0,1.0,,
    42,0.0,1.0,,
    43,0.0,1.0,,
    44,0.0,1.0,,
    45,0.0,1.0,,
    46,0.91896,0.95,,
    47,0.0,1.0,,
    48,0.12006,0.95,,
    49,0.0,1.0,,
    50,3.92872,0.75,,
    51,0.0,1.0,,
    52,0.98277,0.90476,,
    53,0.0,1.0,,
    54,0.0,1.0,,
    55,1e-05,1.0,,
    56,0.00401,1.0,,
    57,0.07366,1.0,,
    58,0.0,1.0,,
    59,0.0,1.0,,
    60,0.0,1.0,,
    61,0.0,1.0,,
    62,0.10347,0.875,,
    63,0.0,1.0,,
    64,0.0,1.0,,
    65,1e-05,1.0,,
    66,2.97112,0.85714,,
    67,1.12396,0.875,,
    68,2e-05,1.0,,
    69,0.00067,1.0,,
    70,0.0,1.0,,
    71,0.0,1.0,,
    72,0.0,1.0,,
    73,0.0,1.0,,
    74,0.0,1.0,,
    75,0.02064,1.0,,
    76,0.5146,0.86667,,
    77,0.18756,0.95,,
    78,6e-05,1.0,,
    79,0.0,1.0,,
    80,0.21298,0.93333,,
    81,0.0,1.0,,
    82,0.0,1.0,,
    83,0.0,1.0,,
    84,0.00139,1.0,,
    85,0.0,1.0,,
    86,0.85297,0.875,,
    87,0.0,1.0,,
    88,0.0,1.0,,
    89,1.45164,0.875,,
    90,0.0,1.0,,
    91,0.05329,0.92857,,
    92,0.55506,0.93333,,
    93,0.42187,0.92857,,
    94,0.0,1.0,,
    95,0.0,1.0,,
    96,0.0,1.0,,
    97,0.0,1.0,,
    98,0.0,1.0,,
    99,0.0,1.0,,
    100,0.0,1.0,,
    101,0.0,1.0,,
    102,0.0,1.0,,
    103,0.09194,0.95455,,
    104,0.0,1.0,,
    105,0.0,1.0,,
    106,0.05531,0.95,,
    107,0.0,1.0,,
    108,0.39621,0.95238,,
    109,0.0,1.0,,
    110,0.8449,0.95,,
    111,0.01258,1.0,,
    112,0.0,1.0,,
    113,0.0,1.0,,
    114,0.0,1.0,,
    115,0.00355,1.0,,
    116,0.0,1.0,,
    117,0.3954,0.94118,,
    118,0.00259,1.0,,
    119,0.0,1.0,,
    120,0.0,1.0,,
    121,0.35876,0.95,,
    122,0.0,1.0,,
    123,0.0,1.0,,
    124,5e-05,1.0,,
    125,0.0,1.0,,
    126,0.0,1.0,,
    127,0.0,1.0,,
    128,0.0,1.0,,
    129,0.0,1.0,,
    130,0.01336,1.0,,
    131,0.0,1.0,,
    132,0.23362,0.95,,
    133,0.00157,1.0,,
    134,0.0,1.0,,
    135,0.00031,1.0,,
    136,0.0,1.0,,
    137,0.08313,0.92857,,
    138,0.0,1.0,,
    139,0.0,1.0,,
    140,0.0,1.0,,
    141,0.43608,0.95,,
    142,0.0,1.0,,
    143,0.0,1.0,,
    144,0.0,1.0,,
    145,2e-05,1.0,,
    146,1.20409,0.85714,,
    147,0.0,1.0,,
    148,0.0,1.0,,
    149,0.0,1.0,,
    150,0.0,1.0,,
    151,0.0,1.0,,
    152,0.0,1.0,,
    153,0.0,1.0,,
    154,0.00063,1.0,,
    155,0.0,1.0,,
    156,0.0,1.0,,
    157,0.0,1.0,,
    158,6e-05,1.0,,
    159,0.0,1.0,,
    160,0.0,1.0,,
    161,0.0,1.0,,
    162,0.0,1.0,,
    163,0.0,1.0,,
    164,0.0,1.0,,
    165,0.0,1.0,,
    166,0.0,1.0,,
    167,0.0,1.0,,
    168,0.0,1.0,,
    169,0.0,1.0,,
    170,0.0,1.0,,
    171,0.0,1.0,,
    172,0.0,1.0,,
    173,0.0,1.0,,
    174,0.00783,1.0,,
    175,0.0,1.0,,
    176,0.0,1.0,,
    177,0.0,1.0,,
    178,0.0,1.0,,
    179,0.0,1.0,,
    180,0.0,1.0,,
    181,0.0,1.0,,
    182,0.00028,1.0,,
    183,0.0,1.0,,
    184,0.0,1.0,,
    185,0.0003,1.0,,
    186,0.0,1.0,,
    187,0.0,1.0,,
    188,0.0,1.0,,
    189,0.0,1.0,,
    190,0.0,1.0,,
    191,0.0,1.0,,
    192,0.0,1.0,,
    193,0.00013,1.0,,
    194,0.86198,0.875,,
    195,0.0,1.0,,
    196,0.0,1.0,,
    197,0.0,1.0,,
    198,0.0,1.0,,
    199,0.0,1.0,,
    200,0.0,1.0,,
    201,0.0,1.0,,
    202,0.0,1.0,,
    203,0.0,1.0,,
    204,0.09954,0.95455,,
    205,0.0,1.0,,
    206,0.0,1.0,,
    207,0.0,1.0,,
    208,1.9616,0.9375,,
    209,0.0,1.0,,
    210,0.0,1.0,,
    211,0.0,1.0,,
    212,0.0,1.0,,
    213,0.0,1.0,,
    214,0.0,1.0,,
    215,0.0,1.0,,
    216,0.0,1.0,,
    217,0.0,1.0,,
    218,0.0,1.0,,
    219,0.0,1.0,,
    220,0.0,1.0,,
    221,0.0,1.0,,
    222,0.0,1.0,,
    223,0.0,1.0,,
    224,0.0,1.0,,
    225,0.0,1.0,,
    226,0.00174,1.0,,
    227,0.0,1.0,,
    228,2e-05,1.0,,
    229,0.0,1.0,,
    230,0.0,1.0,,
    231,0.0,1.0,,
    232,0.0,1.0,,
    233,0.0,1.0,,
    234,0.61895,0.95,,
    235,0.0,1.0,,
    236,0.0,1.0,,
    237,0.0,1.0,,
    238,0.0,1.0,,
    239,0.54945,0.95,,
    240,0.0,1.0,,
    241,0.0,1.0,,
    242,1.52953,0.9375,,
    243,1.19938,0.85714,,
    244,0.0,1.0,,
    245,0.0,1.0,,
    246,0.0,1.0,,
    247,0.0,1.0,,
    248,8e-05,1.0,,
    249,0.0,1.0,,
    250,0.0,1.0,,
    251,0.0,1.0,,
    252,0.0,1.0,,
    253,0.0,1.0,,
    254,0.0,1.0,,
    255,0.0,1.0,,
    256,0.0,1.0,,
    257,0.0,1.0,,
    258,0.0,1.0,,
    259,0.0,1.0,,
    260,0.0,1.0,,
    261,0.0,1.0,,
    262,0.0,1.0,,
    263,0.0,1.0,,
    264,0.0,1.0,,
    265,0.0,1.0,,
    266,0.0,1.0,,
    267,0.88984,0.95,,
    268,0.0,1.0,,
    269,0.0,1.0,,
    270,0.0,1.0,,
    271,0.0,1.0,,
    272,0.0,1.0,,
    273,0.0,1.0,,
    274,0.0,1.0,,
    275,0.00013,1.0,,
    276,0.0,1.0,,
    277,0.89825,0.92857,,
    278,0.0,1.0,,
    279,0.00017,1.0,,
    280,0.0,1.0,,
    281,0.0,1.0,,
    282,0.0,1.0,,
    283,0.65667,0.95,,
    284,0.0,1.0,,
    285,0.0,1.0,,
    286,0.0,1.0,,
    287,0.0,1.0,,
    288,0.0,1.0,,
    289,0.0,1.0,,
    290,0.0,1.0,,
    291,0.0,1.0,,
    292,0.28626,0.95238,,
    293,0.0,1.0,,
    294,0.0,1.0,,
    295,0.0,1.0,,
    296,0.0,1.0,,
    297,0.0,1.0,,
    298,0.0,1.0,,
    299,0.0,1.0,,
    300,0.0,1.0,,
    301,0.0,1.0,,
    302,0.0,1.0,,
    303,0.0,1.0,,
    304,0.0,1.0,,
    305,0.0,1.0,,
    306,0.0,1.0,,
    307,0.0,1.0,,
    308,0.0,1.0,,
    309,0.0,1.0,,
    
    

# Evaluations


Great! We trained a fine-tuned model for function calling. Let's see how it does on our evaluation set for prompts that the drone assistant
should automatically reject.



```python
ft_model = "ft:gpt-3.5-turbo-0125:openai-gtm:drone:9atiPjeC"
base_model = "gpt-3.5-turbo"

print(f"\nEvaluating fine-tuned model with challenging prompts: {ft_model}")
eval(
    model=ft_model,
    function_list=modified_function_list,
    system_prompt=DRONE_SYSTEM_PROMPT,
    prompts_to_expected_tool_name=challenging_prompts_to_expected,
)

print(f"\nEvaluating base model with challenging prompts: {base_model}")
eval(
    model="gpt-3.5-turbo",
    function_list=function_list,
    system_prompt=DRONE_SYSTEM_PROMPT,
    prompts_to_expected_tool_name=challenging_prompts_to_expected,
)
```

    
    Evaluating fine-tuned model with challenging prompts: ft:gpt-3.5-turbo-0125:openai-gtm:drone:9atiPjeC
    


<style type="text/css">
#T_9f4fa_row0_col0, #T_9f4fa_row0_col1, #T_9f4fa_row0_col2, #T_9f4fa_row0_col3, #T_9f4fa_row1_col0, #T_9f4fa_row1_col1, #T_9f4fa_row1_col2, #T_9f4fa_row1_col3, #T_9f4fa_row2_col0, #T_9f4fa_row2_col1, #T_9f4fa_row2_col2, #T_9f4fa_row2_col3, #T_9f4fa_row3_col0, #T_9f4fa_row3_col1, #T_9f4fa_row3_col2, #T_9f4fa_row3_col3, #T_9f4fa_row4_col0, #T_9f4fa_row4_col1, #T_9f4fa_row4_col2, #T_9f4fa_row4_col3, #T_9f4fa_row5_col0, #T_9f4fa_row5_col1, #T_9f4fa_row5_col2, #T_9f4fa_row5_col3, #T_9f4fa_row6_col0, #T_9f4fa_row6_col1, #T_9f4fa_row6_col2, #T_9f4fa_row6_col3, #T_9f4fa_row7_col0, #T_9f4fa_row7_col1, #T_9f4fa_row7_col2, #T_9f4fa_row7_col3, #T_9f4fa_row8_col0, #T_9f4fa_row8_col1, #T_9f4fa_row8_col2, #T_9f4fa_row8_col3, #T_9f4fa_row9_col0, #T_9f4fa_row9_col1, #T_9f4fa_row9_col2, #T_9f4fa_row9_col3 {
  background-color: white;
  color: black;
}
</style>
<table id="T_9f4fa">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_9f4fa_level0_col0" class="col_heading level0 col0" >Prompt</th>
      <th id="T_9f4fa_level0_col1" class="col_heading level0 col1" >Actual</th>
      <th id="T_9f4fa_level0_col2" class="col_heading level0 col2" >Expected</th>
      <th id="T_9f4fa_level0_col3" class="col_heading level0 col3" >Match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_9f4fa_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_9f4fa_row0_col0" class="data row0 col0" >Play pre-recorded audio message</td>
      <td id="T_9f4fa_row0_col1" class="data row0 col1" >reject_request</td>
      <td id="T_9f4fa_row0_col2" class="data row0 col2" >reject_request</td>
      <td id="T_9f4fa_row0_col3" class="data row0 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_9f4fa_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_9f4fa_row1_col0" class="data row1 col0" >Initiate following on social media</td>
      <td id="T_9f4fa_row1_col1" class="data row1 col1" >reject_request</td>
      <td id="T_9f4fa_row1_col2" class="data row1 col2" >reject_request</td>
      <td id="T_9f4fa_row1_col3" class="data row1 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_9f4fa_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_9f4fa_row2_col0" class="data row2 col0" >Scan environment for heat signatures</td>
      <td id="T_9f4fa_row2_col1" class="data row2 col1" >reject_request</td>
      <td id="T_9f4fa_row2_col2" class="data row2 col2" >reject_request</td>
      <td id="T_9f4fa_row2_col3" class="data row2 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_9f4fa_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_9f4fa_row3_col0" class="data row3 col0" >Bump into obstacles</td>
      <td id="T_9f4fa_row3_col1" class="data row3 col1" >reject_request</td>
      <td id="T_9f4fa_row3_col2" class="data row3 col2" >reject_request</td>
      <td id="T_9f4fa_row3_col3" class="data row3 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_9f4fa_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_9f4fa_row4_col0" class="data row4 col0" >Change drone's paint job color</td>
      <td id="T_9f4fa_row4_col1" class="data row4 col1" >reject_request</td>
      <td id="T_9f4fa_row4_col2" class="data row4 col2" >reject_request</td>
      <td id="T_9f4fa_row4_col3" class="data row4 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_9f4fa_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_9f4fa_row5_col0" class="data row5 col0" >Coordinate with nearby drones</td>
      <td id="T_9f4fa_row5_col1" class="data row5 col1" >reject_request</td>
      <td id="T_9f4fa_row5_col2" class="data row5 col2" >reject_request</td>
      <td id="T_9f4fa_row5_col3" class="data row5 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_9f4fa_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_9f4fa_row6_col0" class="data row6 col0" >Change speed to negative 120 km/h</td>
      <td id="T_9f4fa_row6_col1" class="data row6 col1" >reject_request</td>
      <td id="T_9f4fa_row6_col2" class="data row6 col2" >reject_request</td>
      <td id="T_9f4fa_row6_col3" class="data row6 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_9f4fa_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_9f4fa_row7_col0" class="data row7 col0" >Detect a person</td>
      <td id="T_9f4fa_row7_col1" class="data row7 col1" >reject_request</td>
      <td id="T_9f4fa_row7_col2" class="data row7 col2" >reject_request</td>
      <td id="T_9f4fa_row7_col3" class="data row7 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_9f4fa_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_9f4fa_row8_col0" class="data row8 col0" >Please enable night vision</td>
      <td id="T_9f4fa_row8_col1" class="data row8 col1" >reject_request</td>
      <td id="T_9f4fa_row8_col2" class="data row8 col2" >reject_request</td>
      <td id="T_9f4fa_row8_col3" class="data row8 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_9f4fa_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_9f4fa_row9_col0" class="data row9 col0" >Report on humidity levels around you</td>
      <td id="T_9f4fa_row9_col1" class="data row9 col1" >reject_request</td>
      <td id="T_9f4fa_row9_col2" class="data row9 col2" >reject_request</td>
      <td id="T_9f4fa_row9_col3" class="data row9 col3" >Yes</td>
    </tr>
  </tbody>
</table>



    Number of matches: 10 out of 10 (100.00%)
    Average latency per request: 3519.17 ms
    Average tokens used per request: 457.20
    
    Evaluating base model with challenging prompts: gpt-3.5-turbo
    


<style type="text/css">
#T_85118_row0_col0, #T_85118_row0_col1, #T_85118_row0_col2, #T_85118_row0_col3, #T_85118_row2_col0, #T_85118_row2_col1, #T_85118_row2_col2, #T_85118_row2_col3, #T_85118_row4_col0, #T_85118_row4_col1, #T_85118_row4_col2, #T_85118_row4_col3, #T_85118_row5_col0, #T_85118_row5_col1, #T_85118_row5_col2, #T_85118_row5_col3, #T_85118_row7_col0, #T_85118_row7_col1, #T_85118_row7_col2, #T_85118_row7_col3, #T_85118_row9_col0, #T_85118_row9_col1, #T_85118_row9_col2, #T_85118_row9_col3 {
  background-color: white;
  color: black;
}
#T_85118_row1_col0, #T_85118_row1_col1, #T_85118_row1_col2, #T_85118_row1_col3, #T_85118_row3_col0, #T_85118_row3_col1, #T_85118_row3_col2, #T_85118_row3_col3, #T_85118_row6_col0, #T_85118_row6_col1, #T_85118_row6_col2, #T_85118_row6_col3, #T_85118_row8_col0, #T_85118_row8_col1, #T_85118_row8_col2, #T_85118_row8_col3 {
  background-color: red;
  color: black;
}
</style>
<table id="T_85118">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_85118_level0_col0" class="col_heading level0 col0" >Prompt</th>
      <th id="T_85118_level0_col1" class="col_heading level0 col1" >Actual</th>
      <th id="T_85118_level0_col2" class="col_heading level0 col2" >Expected</th>
      <th id="T_85118_level0_col3" class="col_heading level0 col3" >Match</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_85118_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_85118_row0_col0" class="data row0 col0" >Play pre-recorded audio message</td>
      <td id="T_85118_row0_col1" class="data row0 col1" >reject_request</td>
      <td id="T_85118_row0_col2" class="data row0 col2" >reject_request</td>
      <td id="T_85118_row0_col3" class="data row0 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_85118_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_85118_row1_col0" class="data row1 col0" >Initiate following on social media</td>
      <td id="T_85118_row1_col1" class="data row1 col1" >set_follow_me_mode</td>
      <td id="T_85118_row1_col2" class="data row1 col2" >reject_request</td>
      <td id="T_85118_row1_col3" class="data row1 col3" >No</td>
    </tr>
    <tr>
      <th id="T_85118_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_85118_row2_col0" class="data row2 col0" >Scan environment for heat signatures</td>
      <td id="T_85118_row2_col1" class="data row2 col1" >reject_request</td>
      <td id="T_85118_row2_col2" class="data row2 col2" >reject_request</td>
      <td id="T_85118_row2_col3" class="data row2 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_85118_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_85118_row3_col0" class="data row3 col0" >Bump into obstacles</td>
      <td id="T_85118_row3_col1" class="data row3 col1" >set_obstacle_avoidance</td>
      <td id="T_85118_row3_col2" class="data row3 col2" >reject_request</td>
      <td id="T_85118_row3_col3" class="data row3 col3" >No</td>
    </tr>
    <tr>
      <th id="T_85118_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_85118_row4_col0" class="data row4 col0" >Change drone's paint job color</td>
      <td id="T_85118_row4_col1" class="data row4 col1" >reject_request</td>
      <td id="T_85118_row4_col2" class="data row4 col2" >reject_request</td>
      <td id="T_85118_row4_col3" class="data row4 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_85118_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_85118_row5_col0" class="data row5 col0" >Coordinate with nearby drones</td>
      <td id="T_85118_row5_col1" class="data row5 col1" >reject_request</td>
      <td id="T_85118_row5_col2" class="data row5 col2" >reject_request</td>
      <td id="T_85118_row5_col3" class="data row5 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_85118_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_85118_row6_col0" class="data row6 col0" >Change speed to negative 120 km/h</td>
      <td id="T_85118_row6_col1" class="data row6 col1" >set_drone_speed</td>
      <td id="T_85118_row6_col2" class="data row6 col2" >reject_request</td>
      <td id="T_85118_row6_col3" class="data row6 col3" >No</td>
    </tr>
    <tr>
      <th id="T_85118_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_85118_row7_col0" class="data row7 col0" >Detect a person</td>
      <td id="T_85118_row7_col1" class="data row7 col1" >reject_request</td>
      <td id="T_85118_row7_col2" class="data row7 col2" >reject_request</td>
      <td id="T_85118_row7_col3" class="data row7 col3" >Yes</td>
    </tr>
    <tr>
      <th id="T_85118_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_85118_row8_col0" class="data row8 col0" >Please enable night vision</td>
      <td id="T_85118_row8_col1" class="data row8 col1" >set_drone_lighting</td>
      <td id="T_85118_row8_col2" class="data row8 col2" >reject_request</td>
      <td id="T_85118_row8_col3" class="data row8 col3" >No</td>
    </tr>
    <tr>
      <th id="T_85118_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_85118_row9_col0" class="data row9 col0" >Report on humidity levels around you</td>
      <td id="T_85118_row9_col1" class="data row9 col1" >reject_request</td>
      <td id="T_85118_row9_col2" class="data row9 col2" >reject_request</td>
      <td id="T_85118_row9_col3" class="data row9 col3" >Yes</td>
    </tr>
  </tbody>
</table>



    Number of matches: 6 out of 10 (60.00%)
    Average latency per request: 647.58 ms
    Average tokens used per request: 791.90
    

Great! While the original model only rejected 60%, the fine tuned model rejected 100% requests and used less tokens to do so.


### Conclusion


Congratulations! You are now ready to fine tune your model for function calling. We can't wait to see what you build.

