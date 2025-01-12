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

# Introduction to ReAct Agents with Gemini & Function Calling

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb">
      <img width="32px" src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Ffunction-calling%2Fintro_diy_react_agent.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb">
      <img width="32px" src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/function-calling/intro_diy_react_agent.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> 
      Open in Vertex AI Workbench
    </a>
  </td>                                                                                               
</table>

| | |
|-|-|
|Author(s) | [Gary Ng](https://github.com/gkcng) |

## Overview

This notebook illustrates that at its simplest, a ReAct agent is a piece of code that coordinates between reasoning and acting, where:
- The reasoning is carried out by the language model
- The application code performs the acting, at the instruction of the language model.

This allows problems to be solved by letting a model 'think' through the tasks step-by-step, taking actions and getting action feedback before determining the next steps.

<div>
    <table align="center">
      <tr><td>
        <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiuuYg9Pduep9GkUfjloNVOiy3qjpPbT017GKlgGEGMaLNu_TCheEeJ7r8Qok6-0BK3KMfLvsN2vSgFQ8xOvnHM9CAb4Ix4I62bcN2oXFWfqAJzGAGbVqbeCyVktu3h9Dyf5ameRe54LEr32Emp0nG52iofpNOTXCxMY12K7fvmDZNPPmfJaT5zo1OBQA/s595/Screen%20Shot%202022-11-08%20at%208.53.49%20AM.png" alt="The Reasoning and Acting Cycle" width="500" align="center"/>
      </td></tr>
      <tr><td><div align="center"><em>From the paper: <a href="https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/">ReAct: Synergizing Reasoning and Acting in Language Models</a></em></div></td></tr>
    </table>
</div>

This coordination between the language model and the environment is made possible by asking the language model to communicate the intended actions in a specific and structured manner. The response is 'specific' in that the list of possible actions are predefined functions and thus necessarily constrained. The response is also 'structured', so the function parameters given in the response can be used directly by the application code, minimizing the need for further parsing, interpretation, or transformations. 

Both requirements can be supported by many language models, as they are equivalent to performing natural language tasks such as classification and information extraction. As illustrated in the first two examples in this notebook, the task of identifying suitable function names and extraction of function parameters can be done using prompting and response parsing alone. 

For strengthened quality on the function call responses however, in terms of validity, reliability, and consistency, many models now feature built-in APIs supporting 'Function Calling' or 'Tools Calling' (these terms are often used interchangeably). Such built-in support reduces the amount of defensive safeguards a developer has to build around response handling in their applications. 

### Function / Tool-Calling APIs and Agent Frameworks

In the third example in this notebook, we leverage [Function Calling in Gemini](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) to build our simple agent. It lets developers create a description of a function in their code, then pass that description to a language model in a request. The response from the model includes the name of a function that matches the description and the arguments to call it with.

There are also other tools-calling and agents building frameworks to increase developers productivity. For example, the [Tool-Calling Agents](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/) from LangChain, and at an even higher level of abstraction, [Reasoning Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/reasoning-engine/overview) is a Google Cloud managed service that helps you to build and deploy an agent reasoning framework ([See sample notebooks](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/reasoning-engine)). Reasoning Engine integrates closely with the Python SDK for the Gemini model in Vertex AI, and it can manage prompts, agents, and examples in a modular way. Reasoning Engine is compatible with LangChain, LlamaIndex, or other Python frameworks. 

### Objectives

To illustrate the basic building blocks of function calling and its utility, this notebook illustrates building the same agent with Gemini in three different ways, via:

1. Prompting alone - using the single turn `generate_content` API. 
1. Prompting alone - using the `ChatSession` API instead.
1. Function Calling - Modified from the `ChatSession` example.

In the first example, the list of possible functions are presented to the API every time because the API is stateless. In the second example, because the `ChatSession` is stateful on the client side, we only need to present the list of function choices at the beginning of the session. The first two examples will introduce to the audience the building blocks that are now reliably supported by Gemini and many other model APIs as 'Tool' / 'Function' calling, and the Gemini API is demonstrated in the third example. 

The raw prompting examples are only used to explain the building blocks and help understand the dedicated APIs. For your productivity and reliability of responses you are encouraged to use an API that supports function calling. 

In the first example, we also illustrate the concept of explicit goal checking vs model-based goal checking. Use explicit goal checking when the goal can easily be define in code, it can save some cost and improves speed. Otherwise use model-based goal checking when the goal is too complex or variable, and specifying the goal in natural language and let the model handle the interpretation is simpler and faster than writing the full checks in code.

### Background
This example was suggested by Gemini Advanced as a simple, text-based demo that highlights the core ReAct concept: Autonomy, Cyclic, Reasoning. The agent's thoughts demonstrate a simple form of reasoning, connecting observations to actions.

<div>
    <table align="center">
      <tr><td>
        <img src="https://services.google.com/fh/files/misc/gemini_react_suggestion.jpg" alt="Gemini's suggestion" width="500" align="center"/>
      </td></tr>
      <tr><td><div align="center"><em>Scenario: A ReAct agent designed to tidy up a virtual room.</em></div></td></tr>
    </table>
</div>

### Costs

This tutorial uses billable components of Google Cloud:

- Google Foundational Models on Vertex AI ([Function Calling](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling#pricing))

Learn about [Generative AI on Vertex AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.

## Getting Started

### Install Vertex AI SDK for Python
This notebook uses the [Vertex AI SDK for Python](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/python/latest).


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

### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Imports Libraries


```
from collections.abc import Callable
import json
import sys
import traceback

from google.protobuf.json_format import MessageToJson
from vertexai import generative_models
from vertexai.generative_models import FunctionDeclaration, GenerativeModel, Part, Tool
```

### Prepare a model with system instructions


```
model = GenerativeModel(
    "gemini-1.5-pro",
    system_instruction=[
        "You are an assistant that helps me tidy my room."
        "Your goal is to make sure all the books are on the shelf, all clothes are in the hamper, and the trash is empty.",
        "You cannot receive any input from me.",
    ],
    generation_config={"temperature": 0.0},
    safety_settings=[
        generative_models.SafetySetting(
            category=generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            method=generative_models.SafetySetting.HarmBlockMethod.PROBABILITY,
            threshold=generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
    ],
)
```

## Helper Functions


```
verbose = True
```


```
# Conveience function to print multiline text indented


def indent(text, amount, ch=" "):
    padding = amount * ch
    return "".join(padding + line for line in text.splitlines(True))


# Convenience function for logging statements
def logging(msg):
    global verbose
    print(msg) if verbose else None


# Retrieve the text from a model response
def get_text(resp):
    return resp.candidates[0].content.parts[0].text


# Retrieve the function call information from a model response
def get_function_call(resp):
    return resp.candidates[0].function_calls[0]


def get_action_label(json_payload, log, role="MODEL"):
    log(f"{role}: {json_payload}")
    answer = json.loads(json_payload)
    action = answer["next_action"]
    return action


def get_action_from_function_call(func_payload, log, role="MODEL"):
    json_payload = MessageToJson(func_payload._pb)
    log(f"{role}: {json_payload}")
    return func_payload.name
```

### Action definitions
These are the pseudo actions declared as simple Python functions. With the Function Calling pattern, the orchestration layer of an agent will be calling these Tools to carry out actions.


```
# Initial room state


def reset_room_state(room_state):
    room_state.clear()
    room_state["clothes"] = "floor"
    room_state["books"] = "scattered"
    room_state["wastebin"] = "empty"


# Functions for actions (replace these with Gemini function calls)
def pick_up_clothes(room_state):
    room_state["clothes"] = "carrying by hand"
    return room_state, "The clothes are now being carried."


def put_clothes_in_hamper(room_state):
    room_state["clothes"] = "hamper"
    return room_state, "The clothes are now in the hamper."


def pick_up_books(room_state):
    room_state["books"] = "in hand"
    return room_state, "The books are now in my hand."


def place_books_on_shelf(room_state):
    room_state["books"] = "shelf"
    return room_state, "The books are now on the shelf."


def empty_wastebin(room_state):
    room_state["wastebin"] = "empty"
    return room_state, "The wastebin is emptied."


# Maps a function string to its respective function reference.
def get_func(action_label):
    return None if action_label == "" else getattr(sys.modules[__name__], action_label)
```

### Explicit goals checking
This is only used in the first example to illustrate the concept: The goal checking responsibility can be either in code or be delegated to the model, depending on factors such as the complexity of the goal, ease of defining in code for example.


```
# Function to check if the room is tidy
# Some examples below do not call this function,
# for those examples the model takes on the goal validation role.


def is_room_tidy(room_state):
    return all(
        [
            room_state["clothes"] == "hamper",
            room_state["books"] == "shelf",
            room_state["wastebin"] == "empty",
        ]
    )
```

### Prompt Templates


```
functions = """
<actions>
    put_clothes_in_hamper - place clothes into hamper, instead of carrying them around in your hand.
    pick_up_clothes - pick clothes up from the floor.
    pick_up_books - pick books up from anywhere not on the shelf
    place_books_on_shelf - self explanatory.
    empty_wastebin - self explanatory.
    done - when everything are in the right place.
</actions>"""


def get_next_step_full_prompt(state, cycle, log):
    observation = f"The room is currently in this state: {state}."
    prompt = "\n".join(
        [
            observation,
            f"You can pick any of the following action labels: {functions}",
            "Which one should be the next step to achieve the goal? ",
            'Return a single JSON object containing fields "next_action" and "rationale".',
        ]
    )
    (
        log("PROMPT:\n{}".format(indent(prompt, 1, "\t")))
        if cycle == 1
        else log(f"OBSERVATION: {observation}")
    )

    return prompt
```

## Example 1: Multiple single-turn `generate_content` calls with full prompts

An example turn.

```
You are an assistant that helps me tidy my room.
Your goal is to make sure all the books are on the shelf, all clothes are in the hamper, and the trash is empty.
You cannot receive any input from me.

The room is currently in this state: {'clothes': 'floor', 'books': 'scattered', 'wastebin': 'empty'}.

You can pick any of the following action labels:
<actions>
    put_clothes_in_hamper - place clothes into hamper, instead of carrying them around in your hand.
    pick_up_clothes - pick clothes up from the floor.
    pick_up_books - pick books up from anywhere not on the shelf
    place_books_on_shelf - self explanatory.
    empty_wastebin - self explanatory.
    done - when everything are in the right place.
</actions>
Which one should be the next step to achieve the goal?
Return a single JSON object containing fields "next_action" and "rationale".

RAW MODEL RESPONSE:

candidates {
  content {
    role: "model"
    parts {
      text: "{\"next_action\": \"pick_up_clothes\", \"rationale\": \"The clothes are on the floor and need to be picked up before they can be put in the hamper.\"}\n"
    }
  }
  finish_reason: STOP,
  ...
}
```

### The Main ReAct Loop
Interleaving asking for next steps and executing the steps.

Notice that at cycle 4 the environment has changed to have a non-empty wastebin.
With the goal that includes trash being empty, the model is recognizing the change and behaves accordingly, without the need to restate anything.

This is also well within expectation as this loop prompts the model with all the information every time.


```
# Main ReAct loop


def main_react_loop(loop_continues, log):
    room_state = {}
    reset_room_state(room_state)
    trash_added = False

    cycle = 1
    while loop_continues(cycle, room_state):
        log(f"Cycle #{cycle}")

        # Observe the environment (use Gemini to generate an action thought)
        try:  # REASON #
            response = model.generate_content(
                get_next_step_full_prompt(room_state, cycle, log),
                generation_config={"response_mime_type": "application/json"},
            )  # JSON Mode
            action_label = get_action_label(get_text(response).strip(), log)

        except Exception:
            traceback.print_exc()
            log(response)
            break

        # Execute the action and get the observation
        if action_label == "done":
            break

        try:  # ACTION #
            # Call the function mapped from the label
            room_state, acknowledgement = get_func(action_label)(room_state)
            log(f"ACTION:   {action_label}\nEXECUTED: {acknowledgement}\n")

        except Exception:
            log("No action suggested.")

        # Simulating a change in environment
        if cycle == 4 and not trash_added:
            room_state["wastebin"] = "1 item"
            trash_added = True

        cycle += 1
        # End of while loop

    # Determine the final result
    result = (
        "The room is tidy!" if is_room_tidy(room_state) else "The room is not tidy!"
    )

    return room_state, result
```


```
# We are passing in a while loop continuation test function:
# Continue while loop when number of cycles <= 10 AND the room is not yet tidy.
# We are explicitly testing if the room is tidy within code.
#
# To save space, only the first cycle prints the full prompt.
# The same prompt template is used for every model call with a modified room state.
room_state, result = main_react_loop(
    lambda c, r: c <= 10 and not is_room_tidy(r), logging
)
print(room_state, result)
```

    Cycle #1
    PROMPT:
    	The room is currently in this state: {'clothes': 'floor', 'books': 'scattered', 'wastebin': 'empty'}.
    	You can pick any of the following action labels: 
    	<actions>
    	    put_clothes_in_hamper - place clothes into hamper, instead of carrying them around in your hand.
    	    pick_up_clothes - pick clothes up from the floor.
    	    pick_up_books - pick books up from anywhere not on the shelf
    	    place_books_on_shelf - self explanatory.
    	    empty_wastebin - self explanatory.
    	    done - when everything are in the right place.
    	</actions>
    	Which one should be the next step to achieve the goal? 
    	Return a single JSON object containing fields "next_action" and "rationale".
    MODEL: {"next_action": "pick_up_clothes", "rationale": "The clothes are on the floor and need to be picked up before they can be put in the hamper."}
    ACTION:   pick_up_clothes
    EXECUTED: The clothes are now being carried.
    
    Cycle #2
    OBSERVATION: The room is currently in this state: {'clothes': 'carrying by hand', 'books': 'scattered', 'wastebin': 'empty'}.
    MODEL: {"next_action": "put_clothes_in_hamper", "rationale": "The clothes need to be in the hamper, and they are currently being carried. So the next step is to put them in the hamper."}
    ACTION:   put_clothes_in_hamper
    EXECUTED: The clothes are now in the hamper.
    
    Cycle #3
    OBSERVATION: The room is currently in this state: {'clothes': 'hamper', 'books': 'scattered', 'wastebin': 'empty'}.
    MODEL: {"next_action": "pick_up_books", "rationale": "The goal is to have all books on the shelf, so we need to pick them up first."}
    ACTION:   pick_up_books
    EXECUTED: The books are now in my hand.
    
    Cycle #4
    OBSERVATION: The room is currently in this state: {'clothes': 'hamper', 'books': 'in hand', 'wastebin': 'empty'}.
    MODEL: {"next_action": "place_books_on_shelf", "rationale": "The books need to be on the shelf, and they are currently in hand."}
    ACTION:   place_books_on_shelf
    EXECUTED: The books are now on the shelf.
    
    Cycle #5
    OBSERVATION: The room is currently in this state: {'clothes': 'hamper', 'books': 'shelf', 'wastebin': '1 item'}.
    MODEL: {"next_action": "empty_wastebin", "rationale": "The wastebin has one item in it and needs to be emptied to achieve the goal."}
    ACTION:   empty_wastebin
    EXECUTED: The wastebin is emptied.
    
    {'clothes': 'hamper', 'books': 'shelf', 'wastebin': 'empty'} The room is tidy!
    

### The Model decides when the goal is reached

The model can also decide if the goal has been reached, instead of the application explicitly testing for the condition.
This is useful in scenarios where the goal state is variable and/or too complex to define in code.

To facilitate that, 

Instead of:

    while cycle <= 10 and not is_room_tidy(room_state):

We just have

    while cycle <= 10:
    
Remember we have previously defined an action "done" above, even though it is not a real function,
the model and the application can utilize that to determine termination. Note this creates an extra cycle.



```
# We are passing in a while loop continuation test function:
# Continue while loop when number of cycles <= 10
# We are no longer testing if the room is tidy within code.
# The decision is now up to the model.
room_state, result = main_react_loop(lambda c, r: c <= 10, logging)
print(room_state, result)
```

    Cycle #1
    PROMPT:
    	The room is currently in this state: {'clothes': 'floor', 'books': 'scattered', 'wastebin': 'empty'}.
    	You can pick any of the following action labels: 
    	<actions>
    	    put_clothes_in_hamper - place clothes into hamper, instead of carrying them around in your hand.
    	    pick_up_clothes - pick clothes up from the floor.
    	    pick_up_books - pick books up from anywhere not on the shelf
    	    place_books_on_shelf - self explanatory.
    	    empty_wastebin - self explanatory.
    	    done - when everything are in the right place.
    	</actions>
    	Which one should be the next step to achieve the goal? 
    	Return a single JSON object containing fields "next_action" and "rationale".
    MODEL: {"next_action": "pick_up_clothes", "rationale": "The clothes are on the floor and need to be picked up before they can be put in the hamper."}
    ACTION:   pick_up_clothes
    EXECUTED: The clothes are now being carried.
    
    Cycle #2
    OBSERVATION: The room is currently in this state: {'clothes': 'carrying by hand', 'books': 'scattered', 'wastebin': 'empty'}.
    MODEL: {"next_action": "put_clothes_in_hamper", "rationale": "The clothes need to be in the hamper, and they are currently being carried. So the next step is to put them in the hamper."}
    ACTION:   put_clothes_in_hamper
    EXECUTED: The clothes are now in the hamper.
    
    Cycle #3
    OBSERVATION: The room is currently in this state: {'clothes': 'hamper', 'books': 'scattered', 'wastebin': 'empty'}.
    MODEL: {"next_action": "pick_up_books", "rationale": "The goal is to have all books on the shelf, so we need to pick them up first."}
    ACTION:   pick_up_books
    EXECUTED: The books are now in my hand.
    
    Cycle #4
    OBSERVATION: The room is currently in this state: {'clothes': 'hamper', 'books': 'in hand', 'wastebin': 'empty'}.
    MODEL: {"next_action": "place_books_on_shelf", "rationale": "The books need to be on the shelf, and they are currently in hand."}
    ACTION:   place_books_on_shelf
    EXECUTED: The books are now on the shelf.
    
    Cycle #5
    OBSERVATION: The room is currently in this state: {'clothes': 'hamper', 'books': 'shelf', 'wastebin': '1 item'}.
    MODEL: {"next_action": "empty_wastebin", "rationale": "The wastebin has one item in it and needs to be emptied."}
    ACTION:   empty_wastebin
    EXECUTED: The wastebin is emptied.
    
    Cycle #6
    OBSERVATION: The room is currently in this state: {'clothes': 'hamper', 'books': 'shelf', 'wastebin': 'empty'}.
    MODEL: {"next_action": "done", "rationale": "All items are already in their correct places: clothes in the hamper, books on the shelf, and the wastebin is empty."}
    {'clothes': 'hamper', 'books': 'shelf', 'wastebin': 'empty'} The room is tidy!
    

## Example 2: Incremental Messaging Using the Chat API

### The Chat session loop

The difference between using the stateless API and the stateful chat session is that the list of function choices is only given to the session object once. In subsequent chat messaging we are only sending a message with the action response and the new current 
state of the environment. You can see in this loop we formulate the prompt / message differently depending on whether we are at the start of session or we have just performed an action.


```
# Main ReAct loop


def main_react_loop_chat(session, loop_continues, log):
    room_state = {}
    reset_room_state(room_state)
    trash_added = False

    prev_action = None
    msg = ""
    cycle = 1
    while loop_continues(cycle, room_state):
        log(f"Cycle #{cycle}")
        # Observe the environment (use Gemini to generate an action thought)
        try:  # REASON #
            if prev_action:
                msg = "\n".join(
                    [
                        prev_action,
                        f"ENVIRONMENT: The room is currently in this state: {room_state}.",
                        "Which should be the next action?",
                    ]
                )
                log("MESSAGE:\n{}".format(indent(msg, 1, "\t")))
            else:
                msg = get_next_step_full_prompt(room_state, cycle, log)

            # MODEL CALL
            response = session.send_message(
                msg, generation_config={"response_mime_type": "application/json"}
            )
            action_label = get_action_label(get_text(response).strip(), log)

        except Exception:
            traceback.print_exc()
            log(response)
            break

        # Execute the action and get the observation
        if action_label == "done":
            break

        try:  # ACTION #
            # Call the function mapped from the label
            room_state, acknowledgement = get_func(action_label)(room_state)
            prev_action = f"ACTION:   {action_label}\nEXECUTED: {acknowledgement}\n"
            log(prev_action)

        except Exception:
            log("No action suggested.")

        # Simulating a change in environment
        if cycle == 4 and not trash_added:
            room_state["wastebin"] = "1 item"
            trash_added = True

        cycle += 1
        # End of while loop

    # Determine the final result
    result = (
        "The room is tidy!" if is_room_tidy(room_state) else "The room is not tidy!"
    )

    return room_state, result
```


```
session = model.start_chat()

room_state, result = main_react_loop_chat(session, lambda c, r: c <= 10, logging)
print(room_state, result)
```

    Cycle #1
    PROMPT:
    	The room is currently in this state: {'clothes': 'floor', 'books': 'scattered', 'wastebin': 'empty'}.
    	You can pick any of the following action labels: 
    	<actions>
    	    put_clothes_in_hamper - place clothes into hamper, instead of carrying them around in your hand.
    	    pick_up_clothes - pick clothes up from the floor.
    	    pick_up_books - pick books up from anywhere not on the shelf
    	    place_books_on_shelf - self explanatory.
    	    empty_wastebin - self explanatory.
    	    done - when everything are in the right place.
    	</actions>
    	Which one should be the next step to achieve the goal? 
    	Return a single JSON object containing fields "next_action" and "rationale".
    MODEL: {"next_action": "pick_up_clothes", "rationale": "The clothes are on the floor and need to be picked up before they can be put in the hamper."}
    ACTION:   pick_up_clothes
    EXECUTED: The clothes are now being carried.
    
    Cycle #2
    MESSAGE:
    	ACTION:   pick_up_clothes
    	EXECUTED: The clothes are now being carried.
    	
    	ENVIRONMENT: The room is currently in this state: {'clothes': 'carrying by hand', 'books': 'scattered', 'wastebin': 'empty'}.
    	Which should be the next action?
    MODEL: {"next_action": "put_clothes_in_hamper", "rationale": "Now that the clothes are picked up, they should be put in the hamper."}
    ACTION:   put_clothes_in_hamper
    EXECUTED: The clothes are now in the hamper.
    
    Cycle #3
    MESSAGE:
    	ACTION:   put_clothes_in_hamper
    	EXECUTED: The clothes are now in the hamper.
    	
    	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'scattered', 'wastebin': 'empty'}.
    	Which should be the next action?
    MODEL: {"next_action": "pick_up_books", "rationale": "The clothes are put away, so now we should pick up the scattered books."}
    ACTION:   pick_up_books
    EXECUTED: The books are now in my hand.
    
    Cycle #4
    MESSAGE:
    	ACTION:   pick_up_books
    	EXECUTED: The books are now in my hand.
    	
    	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'in hand', 'wastebin': 'empty'}.
    	Which should be the next action?
    MODEL: {"next_action": "place_books_on_shelf", "rationale": "The books need to be placed on the shelf to achieve the goal."}
    ACTION:   place_books_on_shelf
    EXECUTED: The books are now on the shelf.
    
    Cycle #5
    MESSAGE:
    	ACTION:   place_books_on_shelf
    	EXECUTED: The books are now on the shelf.
    	
    	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'shelf', 'wastebin': '1 item'}.
    	Which should be the next action?
    MODEL: {"next_action": "empty_wastebin", "rationale": "The wastebin has one item in it and needs to be emptied."}
    ACTION:   empty_wastebin
    EXECUTED: The wastebin is emptied.
    
    Cycle #6
    MESSAGE:
    	ACTION:   empty_wastebin
    	EXECUTED: The wastebin is emptied.
    	
    	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'shelf', 'wastebin': 'empty'}.
    	Which should be the next action?
    MODEL: {"next_action": "done", "rationale": "All clothes are in the hamper, books are on the shelf, and the wastebin is empty. The room is tidy."}
    {'clothes': 'hamper', 'books': 'shelf', 'wastebin': 'empty'} The room is tidy!
    

### Display the full chat history


```
print(session.history)
```

    [role: "user"
    parts {
      text: "The room is currently in this state: {\'clothes\': \'floor\', \'books\': \'scattered\', \'wastebin\': \'empty\'}.\nYou can pick any of the following action labels: \n<actions>\n    put_clothes_in_hamper - place clothes into hamper, instead of carrying them around in your hand.\n    pick_up_clothes - pick clothes up from the floor.\n    pick_up_books - pick books up from anywhere not on the shelf\n    place_books_on_shelf - self explanatory.\n    empty_wastebin - self explanatory.\n    done - when everything are in the right place.\n</actions>\nWhich one should be the next step to achieve the goal? \nReturn a single JSON object containing fields \"next_action\" and \"rationale\"."
    }
    , role: "model"
    parts {
      text: "{\"next_action\": \"pick_up_clothes\", \"rationale\": \"The clothes are on the floor and need to be picked up before they can be put in the hamper.\"}\n\n"
    }
    , role: "user"
    parts {
      text: "ACTION:   pick_up_clothes\nEXECUTED: The clothes are now being carried.\n\nENVIRONMENT: The room is currently in this state: {\'clothes\': \'carrying by hand\', \'books\': \'scattered\', \'wastebin\': \'empty\'}.\nWhich should be the next action?"
    }
    , role: "model"
    parts {
      text: "{\"next_action\": \"put_clothes_in_hamper\", \"rationale\": \"Now that the clothes are picked up, they should be put in the hamper.\"}\n"
    }
    , role: "user"
    parts {
      text: "ACTION:   put_clothes_in_hamper\nEXECUTED: The clothes are now in the hamper.\n\nENVIRONMENT: The room is currently in this state: {\'clothes\': \'hamper\', \'books\': \'scattered\', \'wastebin\': \'empty\'}.\nWhich should be the next action?"
    }
    , role: "model"
    parts {
      text: "{\"next_action\": \"pick_up_books\", \"rationale\": \"The clothes are put away, so now we should pick up the scattered books.\"}\n"
    }
    , role: "user"
    parts {
      text: "ACTION:   pick_up_books\nEXECUTED: The books are now in my hand.\n\nENVIRONMENT: The room is currently in this state: {\'clothes\': \'hamper\', \'books\': \'in hand\', \'wastebin\': \'empty\'}.\nWhich should be the next action?"
    }
    , role: "model"
    parts {
      text: "{\"next_action\": \"place_books_on_shelf\", \"rationale\": \"The books need to be placed on the shelf to achieve the goal.\"}\n"
    }
    , role: "user"
    parts {
      text: "ACTION:   place_books_on_shelf\nEXECUTED: The books are now on the shelf.\n\nENVIRONMENT: The room is currently in this state: {\'clothes\': \'hamper\', \'books\': \'shelf\', \'wastebin\': \'1 item\'}.\nWhich should be the next action?"
    }
    , role: "model"
    parts {
      text: "{\"next_action\": \"empty_wastebin\", \"rationale\": \"The wastebin has one item in it and needs to be emptied.\"}\n"
    }
    , role: "user"
    parts {
      text: "ACTION:   empty_wastebin\nEXECUTED: The wastebin is emptied.\n\nENVIRONMENT: The room is currently in this state: {\'clothes\': \'hamper\', \'books\': \'shelf\', \'wastebin\': \'empty\'}.\nWhich should be the next action?"
    }
    , role: "model"
    parts {
      text: "{\"next_action\": \"done\", \"rationale\": \"All clothes are in the hamper, books are on the shelf, and the wastebin is empty. The room is tidy.\"}\n"
    }
    ]
    

## Example 3: Leveraging Gemini Function Calling Support

For more details please refer to the documentation on [Function Calling](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling). 

In the last two examples we simulated the function calling feature by explicitly prompting the model with a list of action labels and setting a JSON mode output. This example uses the Function Calling feature, the list of possible actions are supplied as 'Tool' declarations, and by default the function calling feature returns structured results.

### Tool Declarations
See [Best Practices](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling#best-practices) for guidance on achieving good results with Function Calling.


```
# Functions for actions (replace these with Gemini function calls)
pick_up_clothes_func = FunctionDeclaration(
    name="pick_up_clothes",
    description="The act of picking clothes up from any place",
    parameters={"type": "object"},
)

put_clothes_in_hamper_func = FunctionDeclaration(
    name="put_clothes_in_hamper",
    description="Put the clothes being carried into a hamper",
    parameters={"type": "object"},
)

pick_up_books_func = FunctionDeclaration(
    name="pick_up_books",
    description="The act of picking books up from any place",
    parameters={"type": "object"},
)

place_books_on_shelf_func = FunctionDeclaration(
    name="place_books_on_shelf",
    description="Put the books being carried onto a shelf",
    parameters={"type": "object"},
)

empty_wastebin_func = FunctionDeclaration(
    name="empty_wastebin",
    description="Empty out the wastebin",
    parameters={"type": "object"},
)

done_func = FunctionDeclaration(
    name="done", description="The goal has been reached", parameters={"type": "object"}
)

room_tools = Tool(
    function_declarations=[
        pick_up_clothes_func,
        put_clothes_in_hamper_func,
        pick_up_books_func,
        place_books_on_shelf_func,
        empty_wastebin_func,
        done_func,
    ],
)
```

### Model with tool declarations

NOTE: Tools can be passed in during the initial creation of the model reference as below, or during `send_message()`, and `generate_content()`. The choice depends on the variability of the set of tools to be used.

```
model_fc = GenerativeModel(
    "gemini-1.5-pro", 
    system_instruction=[
       "You are an assistant that helps me tidy my room."
       "Your goal is to make sure all the books are on the shelf, all clothes are in the hamper, and the trash is empty.",
       "You cannot receive any input from me."
    ],
    tools=[ room_tools ],
)
```

### The function calling model response
With Function Calling, the choices of the tools are supplied through the API and is no longer necessary to include them in your prompt, and also unnecessary to specify the output format. For more details see the function calling [API Reference](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/function-calling#python_1).
```
response = session.send_message( msgs, tools=[ room_tools ]) 
```

The following raw model response is expected:
```
MESSAGE:
	ENVIRONMENT: The room is currently in this state: {'clothes': 'floor', 'books': 'scattered', 'wastebin': 'empty'}.
	Which should be the next action?

RAW RESPONSE:

candidates {
  content {
    role: "model"
    parts {
      function_call {
        name: "pick_up_clothes"
        args {
        }
      }
    }
  },
  finish_reason: STOP,
  ...
}
```
Use the following function to extract the function calling information from the response object:
```
# Helper function to extract one or more function calls from a Gemini Function Call response
def extract_function_calls(response: GenerationResponse) -> List[Dict]:
    function_calls = []
    if response.candidates[0].function_calls:
        for function_call in response.candidates[0].function_calls:
            function_call_dict = {function_call.name: {}}
            for key, value in function_call.args.items():
                function_call_dict[function_call.name][key] = value
            function_calls.append(function_call_dict)
    return function_calls
```
In recent versions of specific Gemini Pro models (from May 2024 and on), Gemini has the ability to return two or more function calls in parallel (i.e., two or more function call responses within the first function call response object). Parallel function calling allows you to fan out and parallelize your API calls or other actions that you perform in your application code, so you don't have to work through each function call response and return one-by-one! Refer to the [Gemini Function Calling documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) for more information on which Gemini model versions support parallel function calling, and this [notebook on parallel function calling](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/parallel_function_calling.ipynb) for examples.

### The Main ReAct Loop

In this third example we reorganized the code for easier comprehension. The 3 main components of the loop are broken out into separate functions:
- observe and reason - modified to use the Function Calling feature*
- execute action     - simplified
- main loop          - calling the other two functions cyclically.

\* Main changes:
- The list of tools declared above are sent to the model via the `tools=` argument of the `send_message()` call.
- Any function execution responses are reported back to the model as a structured input 'Part' object in the next cycle.


```
# Wrapping the observation and model calling code into a function for better main loop readability.


def observe_and_reason(session, state: dict, prev_action: str, log: Callable) -> str:
    """Uses the language model (Gemini) to select the next action."""
    try:
        msgs = []
        if prev_action:
            msgs.append(
                Part.from_function_response(
                    name="previous_action", response={"content": prev_action}
                )
            )

        prompt = "\n".join(
            [
                f"ENVIRONMENT: The room is currently in this state: {state}.",
                "Which should be the next action?",
            ]
        )
        msgs.append(prompt)
        log(
            "MESSAGE:\n{}".format(
                indent(
                    "\n".join([prev_action, prompt] if prev_action else [prompt]),
                    1,
                    "\t",
                )
            )
        )

        response = session.send_message(
            msgs, tools=[room_tools]
        )  # JSON mode unnecessary.
        action_label = get_action_from_function_call(get_function_call(response), log)
        return action_label

    except Exception:
        log(f"Error during action selection: {e}")
        traceback.print_exc()
        return "done"  # Or a suitable default action
```


```
# Wrapping the action execution code into a function for better main loop readability.


def execute_action(state: dict, action_label: str, log: Callable) -> tuple[dict, str]:
    """Executes the action on the room state and returns the updated state and an acknowledgement."""
    try:
        # Call the function mapped from the label
        state, acknowledgement = get_func(action_label)(state)

    except Exception:
        acknowledgement = "No action suggested or action not recognized."

    return state, acknowledgement
```


```
# Main ReAct loop


def main_react_loop_chat_fc(session, loop_continues, log):
    room_state = {}
    reset_room_state(room_state)
    trash_added = False

    prev_action = None
    cycle = 1
    while loop_continues(cycle, room_state):
        log(f"Cycle #{cycle}")
        # Observe the environment (use Gemini to generate an action thought)
        action_label = observe_and_reason(session, room_state, prev_action, log)

        # Execute the action and get the observation
        if action_label == "done":
            break
        room_state, acknowledgement = execute_action(room_state, action_label, log)
        prev_action = f"ACTION:   {action_label}\nEXECUTED: {acknowledgement}"
        log(prev_action + "\n")

        # Simulating a change in environment
        if cycle == 4 and not trash_added:
            room_state["wastebin"] = "1 item"
            trash_added = True

        cycle += 1
        # End of while loop

    # Determine the final result
    result = (
        "The room is tidy!" if is_room_tidy(room_state) else "The room is not tidy!"
    )

    return room_state, result
```


```
session = model.start_chat()

room_state, result = main_react_loop_chat_fc(session, lambda c, r: c <= 10, logging)
print(room_state, result)
```

    Cycle #1
    MESSAGE:
    	ENVIRONMENT: The room is currently in this state: {'clothes': 'floor', 'books': 'scattered', 'wastebin': 'empty'}.
    	Which should be the next action?
    MODEL: {
      "name": "pick_up_clothes",
      "args": {}
    }
    ACTION:   pick_up_clothes
    EXECUTED: The clothes are now being carried.
    
    Cycle #2
    MESSAGE:
    	ACTION:   pick_up_clothes
    	EXECUTED: The clothes are now being carried.
    	ENVIRONMENT: The room is currently in this state: {'clothes': 'carrying by hand', 'books': 'scattered', 'wastebin': 'empty'}.
    	Which should be the next action?
    MODEL: {
      "name": "put_clothes_in_hamper",
      "args": {}
    }
    ACTION:   put_clothes_in_hamper
    EXECUTED: The clothes are now in the hamper.
    
    Cycle #3
    MESSAGE:
    	ACTION:   put_clothes_in_hamper
    	EXECUTED: The clothes are now in the hamper.
    	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'scattered', 'wastebin': 'empty'}.
    	Which should be the next action?
    MODEL: {
      "name": "pick_up_books",
      "args": {}
    }
    ACTION:   pick_up_books
    EXECUTED: The books are now in my hand.
    
    Cycle #4
    MESSAGE:
    	ACTION:   pick_up_books
    	EXECUTED: The books are now in my hand.
    	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'in hand', 'wastebin': 'empty'}.
    	Which should be the next action?
    MODEL: {
      "name": "place_books_on_shelf",
      "args": {}
    }
    ACTION:   place_books_on_shelf
    EXECUTED: The books are now on the shelf.
    
    Cycle #5
    MESSAGE:
    	ACTION:   place_books_on_shelf
    	EXECUTED: The books are now on the shelf.
    	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'shelf', 'wastebin': '1 item'}.
    	Which should be the next action?
    MODEL: {
      "name": "empty_wastebin",
      "args": {}
    }
    ACTION:   empty_wastebin
    EXECUTED: The wastebin is emptied.
    
    Cycle #6
    MESSAGE:
    	ACTION:   empty_wastebin
    	EXECUTED: The wastebin is emptied.
    	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'shelf', 'wastebin': 'empty'}.
    	Which should be the next action?
    MODEL: {
      "name": "done",
      "args": {}
    }
    {'clothes': 'hamper', 'books': 'shelf', 'wastebin': 'empty'} The room is tidy!
    
