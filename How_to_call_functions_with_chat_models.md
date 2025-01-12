# How to call functions with chat models

This notebook covers how to use the Chat Completions API in combination with external functions to extend the capabilities of GPT models.

`tools` is an optional parameter in the Chat Completion API which can be used to provide function specifications. The purpose of this is to enable models to generate function arguments which adhere to the provided specifications. Note that the API will not actually execute any function calls. It is up to developers to execute function calls using model outputs.

Within the `tools` parameter, if the `functions` parameter is provided then by default the model will decide when it is appropriate to use one of the functions. The API can be forced to use a specific function by setting the `tool_choice` parameter to `{"type": "function", "function": {"name": "my_function"}}`. The API can also be forced to not use any function by setting the `tool_choice` parameter to `"none"`. If a function is used, the output will contain `"finish_reason": "tool_calls"` in the response, as well as a `tool_calls` object that has the name of the function and the generated function arguments.

### Overview

This notebook contains the following 2 sections:

- **How to generate function arguments:** Specify a set of functions and use the API to generate function arguments.
- **How to call functions with model generated arguments:** Close the loop by actually executing functions with model generated arguments.

## How to generate function arguments


```python
!pip install scipy --quiet
!pip install tenacity --quiet
!pip install tiktoken --quiet
!pip install termcolor --quiet
!pip install openai --quiet
```


```python
import json
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored  

GPT_MODEL = "gpt-4o"
client = OpenAI()
```

### Utilities

First let's define a few utilities for making calls to the Chat Completions API and for maintaining and keeping track of the conversation state.


```python
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

```


```python
def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    
    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))

```

### Basic concepts

Let's create some function specifications to interface with a hypothetical weather API. We'll pass these function specification to the Chat Completions API in order to generate function arguments that adhere to the specification.


```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "format", "num_days"]
            },
        }
    },
]
```

If we prompt the model about the current weather, it will respond with some clarifying questions.


```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "What's the weather like today"})
chat_response = chat_completion_request(
    messages, tools=tools
)
assistant_message = chat_response.choices[0].message
messages.append(assistant_message)
assistant_message

```




    ChatCompletionMessage(content='Sure, can you please provide me with the name of your city and state?', role='assistant', function_call=None, tool_calls=None)



Once we provide the missing information, it will generate the appropriate function arguments for us.


```python
messages.append({"role": "user", "content": "I'm in Glasgow, Scotland."})
chat_response = chat_completion_request(
    messages, tools=tools
)
assistant_message = chat_response.choices[0].message
messages.append(assistant_message)
assistant_message

```




    ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_xb7QwwNnx90LkmhtlW0YrgP2', function=Function(arguments='{"location":"Glasgow, Scotland","format":"celsius"}', name='get_current_weather'), type='function')])



By prompting it differently, we can get it to target the other function we've told it about.


```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "what is the weather going to be like in Glasgow, Scotland over the next x days"})
chat_response = chat_completion_request(
    messages, tools=tools
)
assistant_message = chat_response.choices[0].message
messages.append(assistant_message)
assistant_message

```




    ChatCompletionMessage(content='To provide you with the weather forecast for Glasgow, Scotland, could you please specify the number of days you would like the forecast for?', role='assistant', function_call=None, tool_calls=None)



Once again, the model is asking us for clarification because it doesn't have enough information yet. In this case it already knows the location for the forecast, but it needs to know how many days are required in the forecast.


```python
messages.append({"role": "user", "content": "5 days"})
chat_response = chat_completion_request(
    messages, tools=tools
)
chat_response.choices[0]

```




    Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_34PBraFdNN6KR95uD5rHF8Aw', function=Function(arguments='{"location":"Glasgow, Scotland","format":"celsius","num_days":5}', name='get_n_day_weather_forecast'), type='function')]))



#### Forcing the use of specific functions or no function

We can force the model to use a specific function, for example get_n_day_weather_forecast by using the function_call argument. By doing so, we force the model to make assumptions about how to use it.


```python
# in this cell we force the model to use get_n_day_weather_forecast
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "Give me a weather report for Toronto, Canada."})
chat_response = chat_completion_request(
    messages, tools=tools, tool_choice={"type": "function", "function": {"name": "get_n_day_weather_forecast"}}
)
chat_response.choices[0].message
```




    ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_FImGxrLowOAOszCaaQqQWmEN', function=Function(arguments='{"location":"Toronto, Canada","format":"celsius","num_days":7}', name='get_n_day_weather_forecast'), type='function')])




```python
# if we don't force the model to use get_n_day_weather_forecast it may not
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "Give me a weather report for Toronto, Canada."})
chat_response = chat_completion_request(
    messages, tools=tools
)
chat_response.choices[0].message
```




    ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_n84kYFqjNFDPNGDEnjnrd2KC', function=Function(arguments='{"location": "Toronto, Canada", "format": "celsius"}', name='get_current_weather'), type='function'), ChatCompletionMessageToolCall(id='call_AEs3AFhJc9pn42hWSbHTaIDh', function=Function(arguments='{"location": "Toronto, Canada", "format": "celsius", "num_days": 3}', name='get_n_day_weather_forecast'), type='function')])



We can also force the model to not use a function at all. By doing so we prevent it from producing a proper function call.


```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "Give me the current weather (use Celcius) for Toronto, Canada."})
chat_response = chat_completion_request(
    messages, tools=tools, tool_choice="none"
)
chat_response.choices[0].message

```




    ChatCompletionMessage(content="Sure, I'll get the current weather for Toronto, Canada in Celsius.", role='assistant', function_call=None, tool_calls=None)



### Parallel Function Calling

Newer models such as gpt-4o or gpt-3.5-turbo can call multiple functions in one turn.


```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "what is the weather going to be like in San Francisco and Glasgow over the next 4 days"})
chat_response = chat_completion_request(
    messages, tools=tools, model=GPT_MODEL
)

assistant_message = chat_response.choices[0].message.tool_calls
assistant_message
```




    [ChatCompletionMessageToolCall(id='call_ObhLiJwaHwc3U1KyB4Pdpx8y', function=Function(arguments='{"location": "San Francisco, CA", "format": "fahrenheit", "num_days": 4}', name='get_n_day_weather_forecast'), type='function'),
     ChatCompletionMessageToolCall(id='call_5YRgeZ0MGBMFKE3hZiLouwg7', function=Function(arguments='{"location": "Glasgow, SCT", "format": "celsius", "num_days": 4}', name='get_n_day_weather_forecast'), type='function')]



## How to call functions with model generated arguments

In our next example, we'll demonstrate how to execute functions whose inputs are model-generated, and use this to implement an agent that can answer questions for us about a database. For simplicity we'll use the [Chinook sample database](https://www.sqlitetutorial.net/sqlite-sample-database/).

*Note:* SQL generation can be high-risk in a production environment since models are not perfectly reliable at generating correct SQL.

### Specifying a function to execute SQL queries

First let's define some helpful utility functions to extract data from a SQLite database.


```python
import sqlite3

conn = sqlite3.connect("data/Chinook.db")
print("Opened database successfully")
```

    Opened database successfully
    


```python
def get_table_names(conn):
    """Return a list of table names."""
    table_names = []
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for table in tables.fetchall():
        table_names.append(table[0])
    return table_names


def get_column_names(conn, table_name):
    """Return a list of column names."""
    column_names = []
    columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    for col in columns:
        column_names.append(col[1])
    return column_names


def get_database_info(conn):
    """Return a list of dicts containing the table name and columns for each table in the database."""
    table_dicts = []
    for table_name in get_table_names(conn):
        columns_names = get_column_names(conn, table_name)
        table_dicts.append({"table_name": table_name, "column_names": columns_names})
    return table_dicts

```

Now can use these utility functions to extract a representation of the database schema.


```python
database_schema_dict = get_database_info(conn)
database_schema_string = "\n".join(
    [
        f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
        for table in database_schema_dict
    ]
)
```

As before, we'll define a function specification for the function we'd like the API to generate arguments for. Notice that we are inserting the database schema into the function specification. This will be important for the model to know about.


```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "ask_database",
            "description": "Use this function to answer user questions about music. Input should be a fully formed SQL query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"""
                                SQL query extracting info to answer the user's question.
                                SQL should be written using this database schema:
                                {database_schema_string}
                                The query should be returned in plain text, not in JSON.
                                """,
                    }
                },
                "required": ["query"],
            },
        }
    }
]
```

### Executing SQL queries

Now let's implement the function that will actually excute queries against the database.


```python
def ask_database(conn, query):
    """Function to query SQLite database with a provided SQL query."""
    try:
        results = str(conn.execute(query).fetchall())
    except Exception as e:
        results = f"query failed with error: {e}"
    return results
```

##### Steps to invoke a function call using Chat Completions API: 

**Step 1**: Prompt the model with content that may result in model selecting a tool to use. The description of the tools such as a function names and signature is defined in the 'Tools' list and passed to the model in API call. If selected, the function name and parameters are included in the response.<br>
  
**Step 2**: Check programmatically if model wanted to call a function. If true, proceed to step 3. <br>  
**Step 3**: Extract the function name and parameters from response, call the function with parameters. Append the result to messages. <br>    
**Step 4**: Invoke the chat completions API with the message list to get the response. 


```python
# Step #1: Prompt with content that may result in function call. In this case the model can identify the information requested by the user is potentially available in the database schema passed to the model in Tools description. 
messages = [{
    "role":"user", 
    "content": "What is the name of the album with the most tracks?"
}]

response = client.chat.completions.create(
    model='gpt-4o', 
    messages=messages, 
    tools= tools, 
    tool_choice="auto"
)

# Append the message to messages list
response_message = response.choices[0].message 
messages.append(response_message)

print(response_message)
```

    ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_wDN8uLjq2ofuU6rVx1k8Gw0e', function=Function(arguments='{"query":"SELECT Album.Title, COUNT(Track.TrackId) AS TrackCount FROM Album INNER JOIN Track ON Album.AlbumId = Track.AlbumId GROUP BY Album.Title ORDER BY TrackCount DESC LIMIT 1;"}', name='ask_database'), type='function')])
    


```python
# Step 2: determine if the response from the model includes a tool call.   
tool_calls = response_message.tool_calls
if tool_calls:
    # If true the model will return the name of the tool / function to call and the argument(s)  
    tool_call_id = tool_calls[0].id
    tool_function_name = tool_calls[0].function.name
    tool_query_string = json.loads(tool_calls[0].function.arguments)['query']

    # Step 3: Call the function and retrieve results. Append the results to the messages list.      
    if tool_function_name == 'ask_database':
        results = ask_database(conn, tool_query_string)
        
        messages.append({
            "role":"tool", 
            "tool_call_id":tool_call_id, 
            "name": tool_function_name, 
            "content":results
        })
        
        # Step 4: Invoke the chat completions API with the function response appended to the messages list
        # Note that messages with role 'tool' must be a response to a preceding message with 'tool_calls'
        model_response_with_function_call = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        print(model_response_with_function_call.choices[0].message.content)
    else: 
        print(f"Error: function {tool_function_name} does not exist")
else: 
    # Model did not identify a function to call, result can be returned to the user 
    print(response_message.content) 
```

    The album with the most tracks is titled "Greatest Hits," which contains 57 tracks.
    

## Next Steps

See our other [notebook](How_to_call_functions_for_knowledge_retrieval.ipynb) that demonstrates how to use the Chat Completions API and functions for knowledge retrieval to interact conversationally with a knowledge base.
