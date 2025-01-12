# OpenAI Assistants APIs

The Assistants' API lets you create AI assistants in your applications. These assistants follow instructions. They use models, tools, and knowledge to answer user questions. In this notebook we are going to use one of the tools, retriever, to query against two pdf documents we will upload.

The architecture and data flow diagram below depicts the interaction among all components that comprise OpenAI Assistant APIs. Central to understand is the Threads and Runtime that executes asynchronously, adding and reading messages to the Threads.

For integrating the Assistants API:

1. Creat an Assistant with custom instructions and select a model. Optionally, enable tools like Code Interpreter, Retrieval, and Function Calling.

2. Initiate a Thread for each user conversation.
    
3. Add user queries as Messages to the Thread.

4.  Run the Assistant on the Thread for responses, which automatically utilizes the enabled tools

Below we follow those steps to demonstrate how to integrate Assistants API, using function tool, to ask our Assistant to interact with an external webservice, such
as Google Search. This external service could be any external [API Webserivce](https://apilayer.com/)

The OpenAI documentation describes in details [how Assistants work](https://platform.openai.com/docs/assistants/how-it-works).

<img src="./images/assistant_ai_tools_functions_google.png">

**Note**: Much of the code and diagrams are inspired from  Randy Michak of [Empowerment AI](https://www.youtube.com/watch?v=yzNG3NnF0YE)


## How to use Assistant API using Tools: Function calling
In this example, we will use an external service. That is,
our function will call an external web service: Google Search API
to fetch the results of the query requested. 

This is an example of how an Assistant can employ an external tool, such as a webservice. Our query could be part of a larger
application using LLM and Assitant to respond to user query, and then using the results fetched to use downstream.

Let's see how we can do it. The steps are not dissimilar to our
previous notebook. The only difference here is that our function is make an external webservice call and we have a different function JSON definition to match the the arguments to our function call, which it can use to pass to an external webservice.


```python
import warnings
import os
import json
import time

import openai
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Any
from assistant_utils import print_thread_messages, upload_files, \
                            loop_until_completed, create_assistant_run
from function_utils import add_prime_numbers
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
MODEL = os.getenv("MODEL")
print(f"Using MODEL={MODEL}; base={openai.api_base}")
```

    Using MODEL=gpt-4-1106-preview; base=https://api.openai.com/v1
    


```python
from openai import OpenAI

client = OpenAI(
    api_key = openai.api_key,
    base_url = openai.api_base
)
```

### Step 1: Create our custom function definition
This our JSON object definiton for our function:
* name of the function
* parameters for the funtion
* type of arguments
* descriptions for function and each parameter type


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
tools = [search_google_query]
```

### Step 2: Create an Assistant 
Before you can start interacting with the Assistant to carry out any tasks, you need an AI assistant object. Supply the Assistant with a model to use, tools, i.e., functions


```python
instructions = """You are a knowledgeable and helpful chatbot trained to resolve Google interact
with external webservices such as Google via help of function calls
"""
assistant = client.beta.assistants.create(name="AI Assistant for Web services",
                                           instructions=instructions,
                                           model=MODEL,
                                           tools=tools)
assistant
```




    Assistant(id='asst_J3ex2nQI30pXQEjcXygB8042', created_at=1706048744, description=None, file_ids=[], instructions='You are a knowledgeable and helpful chatbot trained to resolve Google interact\nwith external webservices such as Google via help of function calls\n', metadata={}, model='gpt-4-1106-preview', name='AI Assistant for Web services', object='assistant', tools=[ToolFunction(function=FunctionDefinition(name='google_search', description='A function takes in a search query, api key, and optionly num of results specified. ', parameters={'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'The search query to send to the Google Search Engine'}, 'api_key': {'type': 'string', 'description': 'Google Search API key'}, 'num_results': {'type': 'integer', 'description': 'number of results. This is a optional one, default is 1'}}}), type='function')])



### Step 3: Create an empty thread 
As the diagram above shows, the Thread is the object with which the AI Assistant runs will interact with, by fetching messages and putting messages to it. Think of a thread as a "conversation session between an Assistant and a user. Threads store Messages and automatically handle truncation to fit content into a model’s context window."


```python
thread = client.beta.threads.create()
thread
```




    Thread(id='thread_dCXSYZyjLJzaH8zaJkDs7CwA', created_at=1706048745, metadata={}, object='thread')



### Step 4: Add your message query to the thread for the Assistant


```python
content = """Search Google for the top 5 Italian resturants in San Francisco.
    """
message = client.beta.threads.messages.create(
    thread_id=thread.id, 
    role="user",
    content=content
)
message
```




    ThreadMessage(id='msg_GsH6gkIbmNBsVbYtpBqi3Vc4', assistant_id=None, content=[MessageContentText(text=Text(annotations=[], value='Search Google for the top 5 Italian resturants in San Francisco.\n    '), type='text')], created_at=1706048747, file_ids=[], metadata={}, object='thread.message', role='user', run_id=None, thread_id='thread_dCXSYZyjLJzaH8zaJkDs7CwA')



### Step 5: Create a Run for the Assistant
A Run is an invocation of an Assistant on a Thread. The Assistant uses it’s configuration and the Thread’s Messages to perform tasks by calling models and tools. As part of a Run, the Assistant appends Messages to the Thread.

Note that Assistance will run asychronously: the run has the following
lifecycle and states: [*expired, completed, requires, failed, cancelled*]. Run objects can have multiple statuses.

<img src="https://cdn.openai.com/API/docs/images/diagram-1.png">


```python
instruction_msg = content
run = create_assistant_run(client, assistant, thread, instruction_msg)
print(run.model_dump_json(indent=2))
```

    {
      "id": "run_DcFBxYdA1F82PFVWAmkg3ZzD",
      "assistant_id": "asst_J3ex2nQI30pXQEjcXygB8042",
      "cancelled_at": null,
      "completed_at": null,
      "created_at": 1706048748,
      "expires_at": 1706049348,
      "failed_at": null,
      "file_ids": [],
      "instructions": "Search Google for the top 5 Italian resturants in San Francisco.\n    ",
      "last_error": null,
      "metadata": {},
      "model": "gpt-4-1106-preview",
      "object": "thread.run",
      "required_action": null,
      "started_at": null,
      "status": "queued",
      "thread_id": "thread_dCXSYZyjLJzaH8zaJkDs7CwA",
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
        }
      ],
      "usage": null
    }
    

### Step 6: Retrieve the status and loop until the Assistant run status is `completed`
Loop until run status is **required_action**, which is a trigger notification to extract arguments generated by the LLM model and carry onto the next step: invoke the function with the generated arguments.


```python
while True:
    # Wait for 2.5 seconds
    time.sleep(2.5)

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
        print("\nFinal output from the run:")
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
            if func_name == "google_search":
                params = {
                    "query": func_args["query"],
                    "api_key": google_api_key,
                    "num_results": func_args["num_results"]}
                search_results = google_search(params)
                output = f"Top Five Italian resturants: {search_results}"
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": output})
                
            else:
                raise ValueError(f"Unknown function: {func_name}")
            
        # Sending outputs from the function call back to the Assistant
        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs)
    else:
        print(f"Assistant state: {run_status.status} waiting Assistant to process...")
        time.sleep(2.5)
```

    requires_action
    Assistant taking required action: Function calling...
    in_progress
    Assistant state: in_progress waiting Assistant to process...
    in_progress
    Assistant state: in_progress waiting Assistant to process...
    in_progress
    Assistant state: in_progress waiting Assistant to process...
    in_progress
    Assistant state: in_progress waiting Assistant to process...
    in_progress
    Assistant state: in_progress waiting Assistant to process...
    completed
    
    Final output from the run:
    ('assistant:Here are the top 5 Italian restaurants in San Francisco according '
     'to various sources:\n'
     '\n'
     "1. **Original Joe's**\n"
     '   - Address: 601 Union St, San Francisco, CA 94133\n'
     '   - Phone: (415) 775-4877\n'
     "   - [Original Joe's "
     'Information](https://sf.eater.com/maps/best-italian-restaurants-san-francisco)\n'
     '\n'
     '2. **A16**\n'
     '   - Address: 2355 Chestnut St, San Francisco, CA 94123\n'
     '   - Phone: (415) 771-2216\n'
     '   - [More about '
     'A16](https://sf.eater.com/maps/best-italian-restaurants-san-francisco)\n'
     '\n'
     '3. **Seven Hills**\n'
     '   - TripAdvisor ratings mention this place as one of the top Italian '
     'restaurants.\n'
     '   - [Seven Hills on '
     'TripAdvisor](https://www.tripadvisor.com/Restaurants-g60713-c26-San_Francisco_California.html)\n'
     '\n'
     '4. **Acquerello**\n'
     "   - It's recommended for an upscale Italian dining experience.\n"
     '   - [Acquerello Reddit '
     'Mention](https://www.reddit.com/r/AskSF/comments/13awfnm/best_italian_food/)\n'
     '\n'
     '5. **Cotogna**\n'
     '   - Included in a list of 10 Italian Restaurants that can transport you to '
     'Rome.\n'
     '   - [Cotogna '
     'Listing](https://secretsanfrancisco.com/italian-restaurants-sf/)\n'
     '\n'
     'Keep in mind that openings, closings, and service levels may vary, so it '
     'would be best to check the most current status directly with each restaurant '
     'or to consult the most recent reviews and recommendations.')
    'user:Search Google for the top 5 Italian resturants in San Francisco.\n    '
    


```python
# Delete the assistant. 
response = client.beta.assistants.delete(assistant.id)
print(response)
```

    AssistantDeleted(id='asst_J3ex2nQI30pXQEjcXygB8042', deleted=True, object='assistant.deleted')
    
