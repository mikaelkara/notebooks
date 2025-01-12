# OpenAI Assistants APIs

The Assistants' API lets you create AI assistants in your applications. These assistants follow instructions and use models, tools, and knowledge to answer user questions. In this notebook we are going to use one of the tools, retriever,
to query against two pdf documents we will upload.

The architecture and data flow diagram below depicts the interaction among all components that comprise OpenAI Assistant APIs. Central to understand is the Threads and Runtime that executes asynchronously, adding and reading messages to the Threads.

For integrating the Assistants API:

1. Creat an Assistant with custom instructions and select a model. Optionally, enable tools like Code Interpreter, Retrieval, and Function Calling.

2. Initiate a Thread for each user conversation.
    
3. Add user queries as Messages to the Thread.

4.  Run the Assistant on the Thread for responses, which automatically utilizes the enabled tools

Below we follow those steps to demonstrate how to integrate Assistants API, using function tool, to ask our Assistant to interact with an external web services, such
as Google Search, Weather Stacks, and OpenAI too.

This external service could be any external [API Webserivce](https://apilayer.com/)

The OpenAI documentation describes in details [how Assistants work](https://platform.openai.com/docs/assistants/how-it-works).

<img src="images/assistant_ai_tools_parallel_functions.png">

**Note**: Much of the code and diagrams are inspired from  Randy Michak of [Empowerment AI](https://www.youtube.com/watch?v=yzNG3NnF0YE) and OpenAI guide examples


## How to use Assistant API using Tools: Parallel Function calling
In this example, we will use couple of external services. That is,
our function will call an external web services: Google Search API
to fetch the results of the query requested and query a weather web service.

This is an example of how an Assistant can employ an external tool, such as a web services, but in a parallel fashion. Our query could be part of a larger application using LLM and Assitant to respond to user queries to more than one web service, and then using the aggregated results fetched to use downstream.

Let's see how we can do it. The steps are not dissimilar to our
previous notebooks. The only difference here is that our function makes an external web service call to mulitple web services and we have a different function JSON definition to match the the arguments to our function call, which it can use to pass to an external respective web service.


```python
import warnings
import os
import json
import time

import openai
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Any
from assistant_utils import print_thread_messages, \
                            loop_until_completed, \
                            create_assistant_run 
from function_utils import get_weather_data, create_dalle_image
from google_search_utils import google_search
```

    Using MODEL=gpt-4-1106-preview; base=https://api.openai.com/v1
    

Load our .env file with respective API keys and base url endpoints. Here you can either use OpenAI or Anyscale Endpoints. **Note**: Assistant API calling for Anyscale Endpoints (which serves only OS modles) is not yet aviable).


```python
warnings.filterwarnings('ignore')

_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
google_api_key = os.getenv("GOOGLE_API_KEY", "")
weather_api_key = os.getenv("WEATHER_API_KEY", "")
MODEL = os.getenv("MODEL")
print(f"Using MODEL={MODEL}; base={openai.api_base}")
```

    Using MODEL=gpt-4-1106-preview; base=https://api.openai.com/v1
    

Create our OpenAI client, and use this in all subsequent Assistant API calls


```python
from openai import OpenAI

client = OpenAI(
    api_key = openai.api_key,
    base_url = openai.api_base
)
```

### Step 1: Create all our custom function definitions
This our JSON object definiton for all our functions:
* name of the function
* parameters for the funtion
* type of arguments
* descriptions for function and each parameter type


```python
# List of all the descriptions of functions the Assistant can 
# use to satisfy our request.
tools_list = []
```


```python
search_google_query = {
    "type": "function",
    "function": {
        "name": "google_search",
        "description": "A function takes in a search query, api key, and optionly num of results specified. ",
        "parameters": {
            "type": "object",
            "properties": {
                "query" : {
                    "type": "string",
                    "description" : "The search query to send to the Google Search Engine"
                },
                "api_key": {
                    "type": "string",
                    "description" : "Google Search API key"
    
                },
                "num_results" : {
                    "type": "integer",
                    "description" : "number of results. This is a optional one, default is 1"
    
                }
            }
        }
    },
    "required": ["query", "api_key"]
}
tools_list.append(search_google_query)
```


```python
weather_info_query = {
    "type": "function",
    "function": {
        "name": "get_weather_data",
        "description": "A function takes in a city, api key, and optionly a api base URL. ",
        "parameters": {
            "type": "object",
            "properties": {
                "query" : {
                    "type": "string",
                    "description" : "The name of the city  in the US"
                },
                "access_key": {
                    "type": "string",
                    "description" : "Weatherstack API key"
                }
            }
        }
    },
    "required": ["query", "access_key"]
}
tools_list.append(weather_info_query)
```


```python
dalle_3_image_query = {
    "type": "function",
    "function": {
        "name": "create_dalle_image",
        "description": "A function takes in a description to create a Dalle-3 image. ",
        "parameters": {
            "type": "object",
            "properties": {
                "query" : {
                    "type": "string",
                    "description" : "Vivid description of the desired image"
                }
            }
        }
    },
    "required": ["query"]
}
tools_list.append(dalle_3_image_query)
```

Our dispatch function table


```python
tools_function_dispatch_table = {
    "google_search": google_search,
    "get_weather_data": get_weather_data,
    "create_dalle_image": create_dalle_image
}
```

### Step 2: Create an Assistant 
Before you can start interacting with the Assistant to carry out any tasks, you need an AI assistant object. Supply the Assistant with a model to use, tools, i.e., all the functions


```python
instructions = """You are a knowledgeable and helpful chatbot trained to
interact with multiple external webservices such as Google, Weatherstack, and even
call into OpenAI vision model, via help of function calls.
"""
assistant = client.beta.assistants.create(name="AI Assistant for Multiple Web services",
                                           instructions=instructions,
                                           model=MODEL,
                                           tools=tools_list)  # list of functions for the Assistant
assistant
```




    Assistant(id='asst_mbLJ1t0MajpjhhtO57zGyiQ2', created_at=1706048071, description=None, file_ids=[], instructions='You are a knowledgeable and helpful chatbot trained to\ninteract with multiple external webservices such as Google, Weatherstack, and even\ncall into OpenAI vision model, via help of function calls.\n', metadata={}, model='gpt-4-1106-preview', name='AI Assistant for Multiple Web services', object='assistant', tools=[ToolFunction(function=FunctionDefinition(name='google_search', description='A function takes in a search query, api key, and optionly num of results specified. ', parameters={'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'The search query to send to the Google Search Engine'}, 'api_key': {'type': 'string', 'description': 'Google Search API key'}, 'num_results': {'type': 'integer', 'description': 'number of results. This is a optional one, default is 1'}}}), type='function'), ToolFunction(function=FunctionDefinition(name='get_weather_data', description='A function takes in a city, api key, and optionly a api base URL. ', parameters={'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'The name of the city  in the US'}, 'access_key': {'type': 'string', 'description': 'Weatherstack API key'}}}), type='function'), ToolFunction(function=FunctionDefinition(name='create_dalle_image', description='A function takes in a description to create a Dalle-3 image. ', parameters={'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'Vivid description of the desired image'}}}), type='function')])



### Step 3: Create an empty thread 
As the diagram above shows, the Thread is the object with which the AI Assistant runs will interact with, by fetching messages and putting messages to it. Think of a thread as a "conversation session between an Assistant and a user. Threads store Messages and automatically handle truncation to fit content into a model’s context window."


```python
thread = client.beta.threads.create()
thread
```




    Thread(id='thread_iokRRWQvTG02XDaH0kPCCM0Y', created_at=1706048077, metadata={}, object='thread')



### Step 4: Add your message query to the thread for the Assistant


```python
content = """Search Google for top 5 coffe houses or cafes in
San Francisco, get weather information for the city, and generate an image
of a young attractive couple of mixed african and east-indian 
racial heritage, both of them wearing a matching light fabric summer scarve, sitting 
at an outside cafe table having a cup of coffee together, with the San Francisco Golden 
Gate Bridge in the background while the sun is setting in the west. The sunset lights 
up the sky with a beautiful orange glow, partly reflecting on the body of water under the bridge.
To the right of the couple on the wall is a hanging sign with the name of the Caffe Golden Gate.
."""

message = client.beta.threads.messages.create(
    thread_id=thread.id, 
    role="user",
    content=content
)
message
```




    ThreadMessage(id='msg_6YCmgjdgp2URVhgvMG85I3Gs', assistant_id=None, content=[MessageContentText(text=Text(annotations=[], value='Search the Google for top 5 coffe houses or cafes in\nSan Francisco, get weather information for the city, and generate an image\nof a young attractive couple of mixed african and east-indian \nracial heritage, both of them wearing a matching light fabric summer scarve, sitting \nat an outside cafe table having a cup of coffee together with the San Francisco Golden \nGate Bridge in the background while the sun is setting in the west. The sunset lights \nup the sky with a beautiful orange glow, partly reflecting on the body of water under the bridge.\nTo the right of the couple on the wall is a hanging sign with the name of the Caffe Golden Gate\n.'), type='text')], created_at=1706048086, file_ids=[], metadata={}, object='thread.message', role='user', run_id=None, thread_id='thread_iokRRWQvTG02XDaH0kPCCM0Y')



### Step 5: Create a Run for the Assistant
A Run is an invocation of an Assistant on a Thread. The Assistant uses its configuration and the Thread’s Messages to perform tasks by calling models and tools. As part of a Run, the Assistant appends Messages to the Thread.

Note that Assistance will run asychronously: the run has the following
lifecycle and states: [*expired, completed, requires, failed, cancelled*]. Run objects can have multiple statuses.

<img src="https://cdn.openai.com/API/docs/images/diagram-1.png">


```python
instruction_msg = """Please address the user as Jules Dmatrix."""
run = create_assistant_run(client, assistant, thread, instruction_msg)
print(run.model_dump_json(indent=2))
```

    {
      "id": "run_xMb8lrYlZqN0Gh7maOENOa46",
      "assistant_id": "asst_mbLJ1t0MajpjhhtO57zGyiQ2",
      "cancelled_at": null,
      "completed_at": null,
      "created_at": 1706048090,
      "expires_at": 1706048690,
      "failed_at": null,
      "file_ids": [],
      "instructions": "Please address the user as Jules Dmatrix.",
      "last_error": null,
      "metadata": {},
      "model": "gpt-4-1106-preview",
      "object": "thread.run",
      "required_action": null,
      "started_at": null,
      "status": "queued",
      "thread_id": "thread_iokRRWQvTG02XDaH0kPCCM0Y",
      "tools": [
        {
          "function": {
            "name": "google_search",
            "description": "A function takes in a search query, api key, and optionly num of results specified. ",
            "parameters": {
              "type": "object",
              "properties": {
                "query": {
                  "type": "string",
                  "description": "The search query to send to the Google Search Engine"
                },
                "api_key": {
                  "type": "string",
                  "description": "Google Search API key"
                },
                "num_results": {
                  "type": "integer",
                  "description": "number of results. This is a optional one, default is 1"
                }
              }
            }
          },
          "type": "function"
        },
        {
          "function": {
            "name": "get_weather_data",
            "description": "A function takes in a city, api key, and optionly a api base URL. ",
            "parameters": {
              "type": "object",
              "properties": {
                "query": {
                  "type": "string",
                  "description": "The name of the city  in the US"
                },
                "access_key": {
                  "type": "string",
                  "description": "Weatherstack API key"
                }
              }
            }
          },
          "type": "function"
        },
        {
          "function": {
            "name": "create_dalle_image",
            "description": "A function takes in a description to create a Dalle-3 image. ",
            "parameters": {
              "type": "object",
              "properties": {
                "query": {
                  "type": "string",
                  "description": "Vivid description of the desired image"
                }
              }
            }
          },
          "type": "function"
        }
      ],
      "usage": null
    }
    

### Step 6: Retrieve the status and loop until the Assistant run status is `completed`
Loop until run status is **required_action**, which is a trigger notification to extract arguments generated by the LLM model and carry onto the next step: invoke the function with the generated arguments.


```python
while True:
    time.sleep(3)
    # Retrieve the run status
    run_status = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    print(run_status.status)
    
    # If run is completed, get all the messages
    # on the thread, inserted by the Assistant's run
    if run_status.status == 'completed':
        messages = client.beta.threads.messages.list(
            thread_id=thread.id)

        # Loop through messages and print content based on role
        # and break out of the while loop
        print("\nFinal output from the Assistant run:")
        print_thread_messages(client, thread)        
        break
    elif run_status.status == 'requires_action':
        print("Assistant taking required action: Function calling...")
        required_actions = run_status.required_action.submit_tool_outputs.model_dump()
        
        # Aggregate output from any function
        tool_outputs = []
        
        import json
        for action in required_actions["tool_calls"]:
            func_name = action['function']['name']
            func_args = json.loads(action['function']['arguments'])
            if func_name == "get_weather_data":
                func_args["access_key"] = weather_api_key
            elif func_name == "google_search":
                func_args["api_key"] = google_api_key

            # Use the dispatch function table to invoke
            # our function
            func = tools_function_dispatch_table[func_name]
            func_results = func(func_args)
            if func_name == "get_weather_data":
                output = f"The temperature in {func_results['location']['name']} is {func_results['current']['temperature']}, with {func_results['current']['weather_descriptions'][0]}"
                tool_outputs.append({"tool_call_id": action['id'], "output": output})
            elif func_name == "create_dalle_image":
                output = f"The generated image of the couple at the cafe is at url: {func_results}"
                tool_outputs.append({"tool_call_id": action['id'], "output": output})
            elif func_name == "google_search":
                output = f"Top Coffee houses in San Francisco: {func_results}"
                tool_outputs.append({"tool_call_id": action['id'], "output": output})
            else: 
                raise ValueError(f"Unknown function encountered: {func_name}")

        # Sending outputs from the function call back to the Assistant
        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs)
    else:
        print(f"Assistant run state: '{run_status.status}' ...")
        time.sleep(3)
```

    completed
    
    Final output from the Assistant run:
    ('assistant:Jules Dmatrix, here are the results of your queries:\n'
     '\n'
     '**Top Coffee Houses or Cafes in San Francisco:**\n'
     '1. Wrecking Ball Coffee Roasters: [Visit '
     'Website](https://sf.eater.com/maps/best-coffee-shops-san-francisco)\n'
     '2. York Street Cafe: [Visit '
     'Website](https://sf.eater.com/maps/best-coffee-shops-san-francisco)\n'
     '3. Andytown Coffee Roasters: [Read '
     'More](https://www.sfgate.com/food/article/best-coffee-san-francisco-18163638.php)\n'
     '4. Grand Coffee: [Read '
     'More](https://www.sfgate.com/food/article/best-coffee-san-francisco-18163638.php)\n'
     '5. Saint Frank (along with others mentioned): [Visit Reddit '
     'discussion](https://www.reddit.com/r/sanfrancisco/comments/13hg993/great_local_coffee_shops/)\n'
     '\n'
     '**Weather Information for San Francisco:**\n'
     'The temperature in San Francisco is currently 15°C with partly cloudy '
     'conditions.\n'
     '\n'
     '**Generated Image:**\n'
     'An image has been created according to your description. You can view it by '
     'clicking on the link below:\n'
     '![Couples at '
     'Cafe](https://oaidalleapiprodscus.blob.core.windows.net/private/org-qzeI10umC5EtiLO3PNBCtvem/user-lAhLkDUug18HotjQS5xs6Nan/img-n0xrVuYtLnQagLY9KsACdG7s.png?st=2024-01-23T21%3A15%3A40Z&se=2024-01-23T23%3A15%3A40Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2024-01-23T22%3A05%3A51Z&ske=2024-01-24T22%3A05%3A51Z&sks=b&skv=2021-08-06&sig=Qdtr7R49ib0I1HFNo7LOIZWx4V6oP/oCcrwul2aBp2s%3D)\n'
     '\n'
     'Please click on the relevant links for further information and viewing the '
     'image. Enjoy your virtual San Francisco experience!')
    ('user:Search the Google for top 5 coffe houses or cafes in\n'
     'San Francisco, get weather information for the city, and generate an image\n'
     'of a young attractive couple of mixed african and east-indian \n'
     'racial heritage, both of them wearing a matching light fabric summer scarve, '
     'sitting \n'
     'at an outside cafe table having a cup of coffee together with the San '
     'Francisco Golden \n'
     'Gate Bridge in the background while the sun is setting in the west. The '
     'sunset lights \n'
     'up the sky with a beautiful orange glow, partly reflecting on the body of '
     'water under the bridge.\n'
     'To the right of the couple on the wall is a hanging sign with the name of '
     'the Caffe Golden Gate\n'
     '.')
    


```python
# Delete the assistant. 
response = client.beta.assistants.delete(assistant.id)
print(response)
```

    AssistantDeleted(id='asst_mbLJ1t0MajpjhhtO57zGyiQ2', deleted=True, object='assistant.deleted')
    
