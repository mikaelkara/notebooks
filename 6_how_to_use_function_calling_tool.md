# OpenAI Assistants APIs

The Assistants' API lets you create AI assistants in your applications. These assistants follow instructions. Thy use models, tools, and knowledge to answer user questions. In this notebook we are going to use one of the tools, retriever,
to query against two pdf documents we will upload.

The architecture and data flow diagram below depicts the interaction among all components that comprise OpenAI Assistant APIs. Central to understand is the Threads and Runtime that executes asynchronously, adding and reading messages to the Threads.

For integrating the Assistants API:

1. Creat an Assistant with custom instructions and select a model. Optionally, enable tools like Code Interpreter, Retrieval, and Function Calling.

2. Initiate a Thread for each user conversation.
    
3. Add user queries as Messages to the Thread.

4.  Run the Assistant on the Thread for responses, which automatically utilizes the enabled tools

Below we follow those steps to demonstrate how to integrate Assistants API, using function tool, to ask our Assistant to solve simple and complex Maths problems.

The OpenAI documentation describes in details [how Assistants work](https://platform.openai.com/docs/assistants/how-it-works).

<img src="./images/assistant_ai_tools_functions.png">

**Note**: Much of the code and diagrams are inspired from  Randy Michak of [Empowerment AI](https://www.youtube.com/watch?v=yzNG3NnF0YE)

## How to use Assistant API using Tools: Function calling


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
```

Load our .env file with respective API keys and base url endpoints. Here you can either use OpenAI or Anyscale Endpoints. **Note**: Assistant API calling for Anyscale Endpoints (which serves only OS modles) is not yet aviable).


```python
warnings.filterwarnings('ignore')

_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
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
sum_of_prime_numbers = {
        "name": "add_prime_numbers",
        "description": "Add a list of 25 integer prime numbers between 2 and 100",
        "parameters": {
            "type": "object",
            "properties": {
                "prime_numbers": {
                    "type": "array",
                    "items": {
                    "type": "integer",
                        "description": "A integer list of 25 random prime numbers betwee 2 and 100"
                         },
                        "description": "List of of 25 prime numbers to be added"
                    }
                },
                "required": ["add_prime_numbers"]
            }
        }

tools = [{'type': 'function', 'function': sum_of_prime_numbers}]
```

### Step 2: Create an Assistant 
Before you can start interacting with the Assistant to carry out any tasks, you need an AI assistant object. Supply the Assistant with a model to use, tools, i.e., functions


```python
assistant = client.beta.assistants.create(name="AI Math Tutor",
                                           instructions="""You are a knowledgeable chatbot trained to help 
                                               solve basic and advanced grade 8-12 Maths problems.
                                               Use a neutral, teacher and  advisory tone""",
                                           model=MODEL,
                                           tools=tools)
assistant
```




    Assistant(id='asst_WKK1dkECzsIXO79mcx5s57P4', created_at=1703728183, description=None, file_ids=[], instructions='You are a knowledgeable chatbot trained to help \n                                               solve basic and advanced grade 8-12 Maths problems.\n                                               Use a neutral, teacher and  advisory tone', metadata={}, model='gpt-4-1106-preview', name='AI Math Tutor', object='assistant', tools=[ToolFunction(function=FunctionDefinition(name='add_prime_numbers', description='Add a list of 25 integer prime numbers between 2 and 100', parameters={'type': 'object', 'properties': {'prime_numbers': {'type': 'array', 'items': {'type': 'integer', 'description': 'A integer list of 25 random prime numbers betwee 2 and 100'}, 'description': 'List of of 25 prime numbers to be added'}}, 'required': ['add_prime_numbers']}), type='function')])



### Step 3: Create an empty thread 
As the diagram above shows, the Thread is the object with which the AI Assistant runs will interact with, by fetching messages and putting messages to it. Think of a thread as a "conversation session between an Assistant and a user. Threads store Messages and automatically handle truncation to fit content into a model’s context window."


```python
thread = client.beta.threads.create()
thread
```




    Thread(id='thread_FHtsNH37IC3lF8wzNaYpLg3e', created_at=1703728184, metadata={}, object='thread')



### Step 4: Add your message query to the thread for the Assistant


```python
message = client.beta.threads.messages.create(
    thread_id=thread.id, 
    role="user",
    content="""Generate 25 random prime numbers between 2 and 100
    as a list, and add the numbers in this generated list
    """
)
message
```




    ThreadMessage(id='msg_b2VlWFzbHVFa5ZVbSFqzgdZc', assistant_id=None, content=[MessageContentText(text=Text(annotations=[], value='Generate 25 random prime numbers between 2 and 100\n    as a list, and add the numbers in this generated list\n    '), type='text')], created_at=1703728185, file_ids=[], metadata={}, object='thread.message', role='user', run_id=None, thread_id='thread_FHtsNH37IC3lF8wzNaYpLg3e')



### Step 5: Create a Run for the Assistant
A Run is an invocation of an Assistant on a Thread. The Assistant uses it’s configuration and the Thread’s Messages to perform tasks by calling models and tools. As part of a Run, the Assistant appends Messages to the Thread.

Note that Assistance will run asychronously: the run has the following
lifecycle and states: [*expired, completed, requires, failed, cancelled*]. Run objects can have multiple statuses.

<img src="https://cdn.openai.com/API/docs/images/diagram-1.png">


```python
instruction_msg = """Generate a random list of 25 prime numbers between 
2 and 100."""
run = create_assistant_run(client, assistant, thread, instruction_msg)
print(run.model_dump_json(indent=2))
```

    {
      "id": "run_60k8auleHJmxXXvYy7df561b",
      "assistant_id": "asst_WKK1dkECzsIXO79mcx5s57P4",
      "cancelled_at": null,
      "completed_at": null,
      "created_at": 1703728187,
      "expires_at": 1703728787,
      "failed_at": null,
      "file_ids": [],
      "instructions": "Generate a random list of 25 prime numbers between \n2 and 100.",
      "last_error": null,
      "metadata": {},
      "model": "gpt-4-1106-preview",
      "object": "thread.run",
      "required_action": null,
      "started_at": null,
      "status": "queued",
      "thread_id": "thread_FHtsNH37IC3lF8wzNaYpLg3e",
      "tools": [
        {
          "function": {
            "name": "add_prime_numbers",
            "description": "Add a list of 25 integer prime numbers between 2 and 100",
            "parameters": {
              "type": "object",
              "properties": {
                "prime_numbers": {
                  "type": "array",
                  "items": {
                    "type": "integer",
                    "description": "A integer list of 25 random prime numbers betwee 2 and 100"
                  },
                  "description": "List of of 25 prime numbers to be added"
                }
              },
              "required": [
                "add_prime_numbers"
              ]
            }
          },
          "type": "function"
        }
      ]
    }
    

### Step 6: Retrieve the status and loop until the Assistant run status is `completed`


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
    # from on thread inserted by the Assistant's run
    if run_status.status == 'completed':
        messages = client.beta.threads.messages.list(
            thread_id=thread.id)

        # Loop through messages and print content based on role
        # and break out of the while loop
        print("Final output from the run:\n")
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
            if func_name == "add_prime_numbers":
                sum_of_primes = add_prime_numbers(func_args)
                type = "even" if sum_of_primes % 2 == 0 else "odd"
                output = f"The sum of random prime numbers between 2 and 100 is {sum_of_primes} and it's an {type} number"
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": output})
            else:
                raise ValueError(f"Unknown function: {func_name}")
            
        # Sending outputs from the function call back to the Assistant
        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )
    else:
        print(f"Assistant state: {run_status.status} waiting Assistant to process...")
        time.sleep(2.5)
```

    requires_action
    Assistant taking required action:Function calling...
    in_progress
    Assistant state: in_progress waiting Assistant to process...
    completed
    Final output from the run:
    
    ('assistant:I have generated a list of 25 random prime numbers between 2 and '
     '100 and summed them up. The total sum of these prime numbers is 1060, and it '
     'is an even number.')
    ('user:Generate 25 random prime numbers between 2 and 100\n'
     '    as a list, and add the numbers in this generated list\n'
     '    ')
    


```python
# Delete the assistant. 
response = client.beta.assistants.delete(assistant.id)
print(response)
```

    AssistantDeleted(id='asst_WKK1dkECzsIXO79mcx5s57P4', deleted=True, object='assistant.deleted')
    
