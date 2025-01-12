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

# Gemini API: Function calling with Python

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Function_calling.ipynb"><img src="https://github.com/Giom-V/gemini-api-cookbook/blob/function_calling/images/colab_logo_32px.png?raw=1" />Run in Google Colab</a>
  </td>
</table>


Function calling lets developers create a description of a function in their code, then pass that description to a language model in a request. The response from the model includes the name of a function that matches the description and the arguments to call it with. Function calling lets you use functions as tools in generative AI applications, and you can define more than one function within a single request.

This notebook provides code examples to help you get started.

### Install dependencies


```
!pip install -U -q "google-generativeai>=0.7.2"  # Install the Python SDK
```

    [?25l   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/164.0 kB[0m [31m?[0m eta [36m-:--:--[0m[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m [32m163.8/164.0 kB[0m [31m9.3 MB/s[0m eta [36m0:00:01[0m[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m164.0/164.0 kB[0m [31m2.9 MB/s[0m eta [36m0:00:00[0m
    [?25h[?25l   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/725.4 kB[0m [31m?[0m eta [36m-:--:--[0m[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m [32m716.8/725.4 kB[0m [31m37.4 MB/s[0m eta [36m0:00:01[0m[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m725.4/725.4 kB[0m [31m17.6 MB/s[0m eta [36m0:00:00[0m
    [?25h


```
import google.generativeai as genai
```

### Set up your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](../quickstarts/Authentication.ipynb) quickstart for an example.


```
from google.colab import userdata

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
```

## Function calling basics

To use function calling, pass a list of functions to the `tools` parameter when creating a [`GenerativeModel`](https://ai.google.dev/api/python/google/generativeai/GenerativeModel). The model uses the function name, docstring, parameters, and parameter type annotations to decide if it needs the function to best answer a prompt.

> Important: The SDK converts function parameter type annotations to a format the API understands (`genai.protos.FunctionDeclaration`). The API only supports a limited selection of parameter types, and the Python SDK's automatic conversion only supports a subset of that: `AllowedTypes = int | float | bool | str | list['AllowedTypes'] | dict`


```
def add(a: float, b: float):
    """returns a + b."""
    return a + b


def subtract(a: float, b: float):
    """returns a - b."""
    return a - b


def multiply(a: float, b: float):
    """returns a * b."""
    return a * b


def divide(a: float, b: float):
    """returns a / b."""
    return a / b


model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", tools=[add, subtract, multiply, divide]
)

model
```




    genai.GenerativeModel(
        model_name='models/gemini-1.5-flash',
        generation_config={},
        safety_settings={},
        tools=<google.generativeai.types.content_types.FunctionLibrary object at 0x7c77c21a6230>,
        system_instruction=None,
        cached_content=None
    )



## Automatic function calling

Function calls naturally fit in to [multi-turn chats](https://ai.google.dev/api/python/google/generativeai/GenerativeModel#multi-turn) as they capture a back and forth interaction between the user and model. The Python SDK's [`ChatSession`](https://ai.google.dev/api/python/google/generativeai/ChatSession) is a great interface for chats because handles the conversation history for you, and using the parameter `enable_automatic_function_calling` simplifies function calling even further:


```
chat = model.start_chat(enable_automatic_function_calling=True)
```

With automatic function calling enabled, `ChatSession.send_message` automatically calls your function if the model asks it to.

In the following example, the result appears to simply be a text response containing the correct answer:


```
response = chat.send_message(
    "I have 57 cats, each owns 44 mittens, how many mittens is that in total?"
)
response.text
```




    "That's a total of 2508 mittens. \n"




```
57 * 44
```




    2508



However, by examining the chat history, you can see the flow of the conversation and how function calls are integrated within it.

The `ChatSession.history` property stores a chronological record of the conversation between the user and the Gemini model. Each turn in the conversation is represented by a [`genai.protos.Content`](https://ai.google.dev/api/python/google/generativeai/protos/Content) object, which contains the following information:

*   **Role**: Identifies whether the content originated from the "user" or the "model".
*   **Parts**: A list of [`genai.protos.Part`](https://ai.google.dev/api/python/google/generativeai/protos/Part) objects that represent individual components of the message. With a text-only model, these parts can be:
    *   **Text**: Plain text messages.
    *   **Function Call** ([`genai.protos.FunctionCall`](https://ai.google.dev/api/python/google/generativeai/protos/FunctionCall)): A request from the model to execute a specific function with provided arguments.
    *   **Function Response** ([`genai.protos.FunctionResponse`](https://ai.google.dev/api/python/google/generativeai/protos/FunctionResponse)): The result returned by the user after executing the requested function.

 In the previous example with the mittens calculation, the history shows the following sequence:

1.  **User**: Asks the question about the total number of mittens.
1.  **Model**: Determines that the multiply function is helpful and sends a FunctionCall request to the user.
1.  **User**: The `ChatSession` automatically executes the function (due to `enable_automatic_function_calling` being set) and sends back a `FunctionResponse` with the calculated result.
1.  **Model**: Uses the function's output to formulate the final answer and presents it as a text response.


```
for content in chat.history:
    print(content.role, "->", [type(part).to_dict(part) for part in content.parts])
    print("-" * 80)
```

    user -> [{'text': 'I have 57 cats, each owns 44 mittens, how many mittens is that in total?'}]
    --------------------------------------------------------------------------------
    model -> [{'function_call': {'name': 'multiply', 'args': {'a': 57.0, 'b': 44.0}}}]
    --------------------------------------------------------------------------------
    user -> [{'function_response': {'name': 'multiply', 'response': {'result': 2508.0}}}]
    --------------------------------------------------------------------------------
    model -> [{'text': "That's a total of 2508 mittens. \n"}]
    --------------------------------------------------------------------------------
    

In general the state diagram is:

<img src="https://codelabs.developers.google.com/static/codelabs/gemini-function-calling/img/gemini-function-calling-overview_1440.png" alt="The model can always reply with text, or a FunctionCall. If the model sends a FunctionCall the user must reply with a FunctionResponse" width=50%>

The model can respond with multiple function calls before returning a text response, and function calls come before the text response.

## Manual function calling

For more control, you can process [`genai.protos.FunctionCall`](https://ai.google.dev/api/python/google/generativeai/protos/FunctionCall) requests from the model yourself. This would be the case if:

- You use a `ChatSession` with the default `enable_automatic_function_calling=False`.
- You use `GenerativeModel.generate_content` (and manage the chat history yourself).

The following example is a rough equivalent of the [function calling single-turn curl sample](https://ai.google.dev/docs/function_calling#function-calling-single-turn-curl-sample) in Python. It uses functions that return (mock) movie playtime information, possibly from a hypothetical API:


```
def find_movies(description: str, location: str = ""):
    """find movie titles currently playing in theaters based on any description, genre, title words, etc.

    Args:
        description: Any kind of description including category or genre, title words, attributes, etc.
        location: The city and state, e.g. San Francisco, CA or a zip code e.g. 95616
    """
    return ["Barbie", "Oppenheimer"]


def find_theaters(location: str, movie: str = ""):
    """Find theaters based on location and optionally movie title which are is currently playing in theaters.

    Args:
        location: The city and state, e.g. San Francisco, CA or a zip code e.g. 95616
        movie: Any movie title
    """
    return ["Googleplex 16", "Android Theatre"]


def get_showtimes(location: str, movie: str, theater: str, date: str):
    """
    Find the start times for movies playing in a specific theater.

    Args:
      location: The city and state, e.g. San Francisco, CA or a zip code e.g. 95616
      movie: Any movie title
      thearer: Name of the theater
      date: Date for requested showtime
    """
    return ["10:00", "11:00"]
```

Use a dictionary to make looking up functions by name easier later on. You can also use it to pass the array of functions to the `tools` parameter of `GenerativeModel`.


```
functions = {
    "find_movies": find_movies,
    "find_theaters": find_theaters,
    "get_showtimes": get_showtimes,
}

model = genai.GenerativeModel(model_name="gemini-1.5-flash", tools=functions.values())
```

After using `generate_content()` to ask a question, the model requests a `function_call`:


```
response = model.generate_content(
    "Which theaters in Mountain View show the Barbie movie?"
)
response.candidates[0].content.parts
```




    [function_call {
      name: "find_theaters"
      args {
        fields {
          key: "location"
          value {
            string_value: "Mountain View"
          }
        }
        fields {
          key: "movie"
          value {
            string_value: "Barbie"
          }
        }
      }
    }
    ]



Since this is not using a `ChatSession` with automatic function calling, you have to call the function yourself.

A very simple way to do this would be with `if` statements:

```python
if function_call.name == 'find_theaters':
  find_theaters(**function_call.args)
elif ...
```

However, since you already made the `functions` dictionary, this can be simplified to:


```
def call_function(function_call, functions):
    function_name = function_call.name
    function_args = function_call.args
    return functions[function_name](**function_args)


part = response.candidates[0].content.parts[0]

# Check if it's a function call; in real use you'd need to also handle text
# responses as you won't know what the model will respond with.
if part.function_call:
    result = call_function(part.function_call, functions)

print(result)
```

    ['Googleplex 16', 'Android Theatre']
    

Finally, pass the response plus the message history to the next `generate_content()` call to get a final text response from the model.


```
from google.protobuf.struct_pb2 import Struct

# Put the result in a protobuf Struct
s = Struct()
s.update({"result": result})

# Update this after https://github.com/google/generative-ai-python/issues/243
function_response = genai.protos.Part(
    function_response=genai.protos.FunctionResponse(name="find_theaters", response=s)
)

# Build the message history
messages = [
    # fmt: off
    {"role": "user",
     "parts": ["Which theaters in Mountain View show the Barbie movie?."]},
    {"role": "model",
     "parts": response.candidates[0].content.parts},
    {"role": "user",
     "parts": [function_response]},
    # fmt: on
]

# Generate the next response
response = model.generate_content(messages)
print(response.text)
```

    The following theaters in Mountain View show the Barbie movie: Googleplex 16 and Android Theatre. 
    
    

## Function calling chain

The model is not limited to one function call, it can chain them until it finds the right answer.


```
chat = model.start_chat(enable_automatic_function_calling=True)
response = chat.send_message(
    "Which comedy movies are shown tonight in Mountain view and at what time?"
)
for content in chat.history:
    print(content.role, "->", [type(part).to_dict(part) for part in content.parts])
    print("-" * 80)
```

    user -> [{'text': 'Which comedy movies are shown tonight in Mountain view and at what time?'}]
    --------------------------------------------------------------------------------
    model -> [{'function_call': {'name': 'find_movies', 'args': {'location': 'Mountain View, CA', 'description': 'comedy'}}}]
    --------------------------------------------------------------------------------
    user -> [{'function_response': {'name': 'find_movies', 'response': {'result': ['Barbie', 'Oppenheimer']}}}]
    --------------------------------------------------------------------------------
    model -> [{'function_call': {'name': 'find_theaters', 'args': {'movie': 'Barbie', 'location': 'Mountain View, CA'}}}]
    --------------------------------------------------------------------------------
    user -> [{'function_response': {'name': 'find_theaters', 'response': {'result': ['Googleplex 16', 'Android Theatre']}}}]
    --------------------------------------------------------------------------------
    model -> [{'function_call': {'name': 'get_showtimes', 'args': {'date': 'tonight', 'location': 'Mountain View, CA', 'theater': 'Googleplex 16', 'movie': 'Barbie'}}}]
    --------------------------------------------------------------------------------
    user -> [{'function_response': {'name': 'get_showtimes', 'response': {'result': ['10:00', '11:00']}}}]
    --------------------------------------------------------------------------------
    model -> [{'text': 'The comedy movie "Barbie" is showing at Googleplex 16 at 10:00 and 11:00 tonight. \n'}]
    --------------------------------------------------------------------------------
    

Here you can see that the model made three calls to answer your question and used the outputs of them in the subsequent calls and in the final answer.

## Parallel function calls

The Gemini API can call multiple functions in a single turn. This caters for scenarios where there are multiple function calls that can take place independently to complete a task.

First set the tools up. Unlike the movie example above, these functions do not require input from each other to be called so they should be good candidates for parallel calling.


```
def power_disco_ball(power: bool) -> bool:
    """Powers the spinning disco ball."""
    print(f"Disco ball is {'spinning!' if power else 'stopped.'}")
    return True


def start_music(energetic: bool, loud: bool, bpm: int) -> str:
    """Play some music matching the specified parameters.

    Args:
      energetic: Whether the music is energetic or not.
      loud: Whether the music is loud or not.
      bpm: The beats per minute of the music.

    Returns: The name of the song being played.
    """
    print(f"Starting music! {energetic=} {loud=}, {bpm=}")
    return "Never gonna give you up."


def dim_lights(brightness: float) -> bool:
    """Dim the lights.

    Args:
      brightness: The brightness of the lights, 0.0 is off, 1.0 is full.
    """
    print(f"Lights are now set to {brightness:.0%}")
    return True
```

Now call the model with an instruction that could use all of the specified tools.


```
# Set the model up with tools.
house_fns = [power_disco_ball, start_music, dim_lights]
# Try this out with Pro and Flash...
model = genai.GenerativeModel(model_name="gemini-1.5-flash", tools=house_fns)

# Call the API.
chat = model.start_chat()
response = chat.send_message("Turn this place into a party!")

# Print out each of the function calls requested from this single call.
for part in response.parts:
    if fn := part.function_call:
        args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
        print(f"{fn.name}({args})")
```

    power_disco_ball(power=True)
    start_music(energetic=True, bpm=120.0, loud=True)
    dim_lights(brightness=0.5)
    

Each of the printed results reflects a single function call that the model has requested. To send the results back, include the responses in the same order as they were requested.


```
# Simulate the responses from the specified tools.
responses = {
    "power_disco_ball": True,
    "start_music": "Never gonna give you up.",
    "dim_lights": True,
}

# Build the response parts.
response_parts = [
    genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn, response={"result": val}))
    for fn, val in responses.items()
]

response = chat.send_message(response_parts)
print(response.text)
```

    I've powered up the disco ball, started some energetic music at 120 BPM, and dimmed the lights.  Get ready to party! 
    
    

## Next Steps
### Useful API references:

- The [genai.GenerativeModel](https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/GenerativeModel.md) class
  - Its [GenerativeModel.generate_content](https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/GenerativeModel.md#generate_content) method builds a [genai.protos.GenerateContentRequest](https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/protos/GenerateContentRequest.md) behind the scenes.
    - The request's `.tools` field contains a list of 1 [genai.protos.Tool](https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/protos/Tool.md) object.
    - The tool's `function_declarations` attribute contains a list of [FunctionDeclarations](https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/protos/FunctionDeclaration.md) objects.
- The [response](https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/protos/GenerateContentResponse.md) may contain a [genai.protos.FunctionCall](https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/protos/FunctionCall.md), in `response.candidates[0].contents.parts[0]`.
- if `enable_automatic_function_calling` is set the [genai.ChatSession](https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/ChatSession.md) executes the call, and sends back the [genai.protos.FunctionResponse](https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/protos/FunctionResponse.md).
- In response to a [FunctionCall](https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/protos/FunctionCall.md) the model always expects a [FunctionResponse](https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/protos/FunctionResponse.md).
- If you reply manually using [chat.send_message](https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/ChatSession.md#send_message) or [model.generate_content](https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/GenerativeModel.md#generate_content) remember thart the API is stateless you have to send the whole conversation history (a list of [content](https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/protos/Content.md) objects), not just the last one containing the `FunctionResponse`.

### Related examples

Check those examples using function calling to give you more ideas on how to use that very useful feature:
* [Barista Bot](../examples/Agents_Function_Calling_Barista_Bot.ipynb), an agent to order coffee
* Using function calling to [re-rank seach results](../examples/Search_reranking_using_embeddings.ipynb)

### Continue your discovery of the Gemini API

Learn how to control how the Gemini API interact with your functions in the [function calling config](../quickstarts/Function_calling_config.ipynb) quickstart, discover how to control the model output in [JSON](../quickstarts/JSON_mode.ipynb) or using an [Enum](../quickstarts/Enum.ipynb) or learn how the Gemini API can generate and run code by itself using [Code execution](../quickstarts/Code_Execution.ipynb)
