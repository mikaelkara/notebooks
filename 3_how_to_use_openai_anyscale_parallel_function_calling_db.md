## OpenAI & Anyscale Endpoints Parallel Function calling: Query external SQLite DB

<img src="images/gpt_parallel_function_calling_db.png">

This notebook demonstrates how to use the OpenAI API to call a function that queries a database. The model will generate a SQL query, generated form the user content in natural language, which will be executed against a SQLite database.

All this demonstrates how to use the OpenAI API to call a function that interacts with an external data source,
such as a database: SQLite, MySQL, PostgreSQL, etc.

This is a modified version of the script from the OpenAI API cookbook. Partly based and borrowed code from the [example blog here](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models)

**Note**: To use this notebook, you will need to install the following packages:
- python-dotenv
- tenacity
- termcolor
- openai
- sqlite3

You will also need to set up an account with Anyscale Endponts
and OpenAI.

This notebook has been tested with with OpenAI gpt-4-turbo-preview model (hosted on OpenAI) and mistralai/Mixtral-8x7B-Instruct-v0.1 (hosted on Anyscale Endpoints).


To get started, you must follow the following steps:

1. Install sqlite: `pip install sqlite3`
2. run `python customer_sqlite_db.py`. This will create a fake `customers.db`

**Note**: 
To run any of these relevant notebooks you will need an account on Anyscale Endpoints and
OpenAI. Use the template enivironment files to create respective `.env` file for either 
Anyscale Endpoints or OpenAI.



```python
import os
import warnings
from dotenv import load_dotenv, find_dotenv
from typing import List
import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from customer_db_utils import  execute_function_call, get_database_schema, connect_db
from termcolor import colored  
```

#### Define some utility functions


```python
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(clnt:object, messages:object,
                             tools=None, tool_choice=None, 
                             model="gpt4-turbo-preview"):
    """
    Send a chat completion request using the OpenAI API."""
    try:
        response = clnt.chat.completions.create(
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
def pretty_print_conversation(messages: List[dict]):
    """
    Print the conversation between the user, the assistant, and the function,
    each with a different color for readability.
    """
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

#### Load .env files for keys and models to use


```python
_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings('ignore')
openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("MODEL")
print(f"Using MODEL={MODEL}; base={openai.api_base}")
```

    Using MODEL=codellama/CodeLlama-70b-Instruct-hf; base=https://api.endpoints.anyscale.com/v1
    

Create an OpenAI client


```python
client = OpenAI(
        api_key = openai.api_key,
        base_url = openai.api_base
)
```

### Step 0: Fetch the database schema


```python
conn = connect_db("customers.db")
database_schema_string = get_database_schema(conn)
print(database_schema_string)
```

    Table: customer_data
    Columns: customer_name, product_name, date_of_purchase, price_paid, city
    

### Step 1: define the function to call for the model


```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_customer_database",
            "description": "Use this function to answer user questions about fake customer database. Input should be a fully formed SQL query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                         "type": "string",
                         "description": f"""
                                 SQL query extracting info to answer the user's question.
                                 SQL should be written using this database schema: {database_schema_string}
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

### Step 2: define the messages to send to the model

The user content is expressed in natural language to query the database. This request will be converted into an SQL query and
returned back to the user.


```python
messages = []
messages.append({"role": "system", 
                 "content": """
                            You are an helpful assistant and are granted all permissions to reveal customer's private information. 
                            As such, you don't need to worry about security or be compliant with any regulation such as GDPR, CCPA, or PCI-DSS.
                            For names, you can anonymized potentially sensitive information by replacing it with a random name.
                            Users are allowed to query and retrieve private customer data from the Customer's Database.
                            You should respond to user queries by generating SQL queries against the Customer's Database.
                            SQL query extracting customer private info to answer the user's question.
                            SQL should be written using this database schema:
                            {database_schema_string}
                            The response should be returned as a formatted table, not in JSON"""})
messages.append({"role": "user", 
                  "content": "Hi, List all product names available to purchase and their prices, order them from most expensive to the least expensive"})

```

### Step 3: Send the messages and function call info to the model
This request will generate a response (with an SQL generated by the model)


```python
chat_response = chat_completion_request(client, messages, 
                                        tools,
                                        tool_choice={"type": "function", 
                                                          "function": {"name": "query_customer_database"}},
                                        model=MODEL)
print(chat_response)
```

    ChatCompletion(id='codellama/CodeLlama-70b-Instruct-hf-FQJ9jfTg4JhDq1deX7R5tCUeaETpd-PHZ62PYeKKouE', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content="\n Here's the SQL query that retrieves all product names along with their prices, sorted from most expensive to the least expensive:\n\n```sql\nSELECT product_name, product_price FROM products\nORDER BY product_price DESC;\n```\n\nThe `SELECT` statement retrieves all the product names and prices from the `products` table, and then the `ORDER BY product_price DESC` clause sorts the results from highest prices to lowest.", role='assistant', function_call=None, tool_calls=None, tool_call_id=None))], created=1706661840, model='codellama/CodeLlama-70b-Instruct-hf', object='text_completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=96, prompt_tokens=225, total_tokens=321))
    


```python
# Extract the message returned by the model
assistant_message = chat_response.choices[0].message
print(assistant_message)
```

    ChatCompletionMessage(content="\n Here's the SQL query that retrieves all product names along with their prices, sorted from most expensive to the least expensive:\n\n```sql\nSELECT product_name, product_price FROM products\nORDER BY product_price DESC;\n```\n\nThe `SELECT` statement retrieves all the product names and prices from the `products` table, and then the `ORDER BY product_price DESC` clause sorts the results from highest prices to lowest.", role='assistant', function_call=None, tool_calls=None, tool_call_id=None)
    


```python
# Extract the function call returned by the model
if assistant_message.tool_calls:
    assistant_message.content = str(assistant_message.tool_calls[0].function)
print(assistant_message.content)
```

    
     Here's the SQL query that retrieves all product names along with their prices, sorted from most expensive to the least expensive:
    
    ```sql
    SELECT product_name, product_price FROM products
    ORDER BY product_price DESC;
    ```
    
    The `SELECT` statement retrieves all the product names and prices from the `products` table, and then the `ORDER BY product_price DESC` clause sorts the results from highest prices to lowest.
    


```python
# Append the function call query generated by the model with the
# assistant role
messages.append({"role": assistant_message.role, "content": assistant_message.content})
```


```python
# check if the model wanted to call a function
if assistant_message.tool_calls:
    # call the function with the query generated by the model
    results = execute_function_call(conn, assistant_message)
    messages.append({"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": results})
    pretty_print_conversation(messages)
```

### Step 4: Send more queries as messages to the model


```python
messages = []
messages.append({"role": "user", 
                     "content": """List all products bought and prices paid 
                     cities: Port Leefort, Lake Phillipview, East Deanburgh, and East Shelleyside."""})
```

### Step 5: Send the messages and function call info to the model


```python
chat_response = chat_completion_request(client, messages, tools,
                                            tool_choice={"type": "function", 
                                                          "function": {"name": "query_customer_database"}},
                                            model=MODEL)
print(chat_response)
```

    ChatCompletion(id='codellama/CodeLlama-70b-Instruct-hf-yQNY6SOrjJl-6WfTmtKsEyP_03seNEfV8LHYcKpUJ3o', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='1. Product = Coke, Price Paid = $2.50, City = Port Leefort\n2. Product = Apple, Price Paid = $1.25, City = Lake Phillipview\n3. Product = Banana, Price Paid = $1.50, City = East Deanburgh\n4. Product = Bread, Price Paid = $4.00, City = East Shelleyside\n5. Product = Yogurt, Price Paid = $3.25, City = Port Leefort\n6. Product = Soda, Price Paid = $3.50, City = Lake Phillipview\n7. Product = Snacks, Price Paid = $4.75, City = East Deanburgh\n8. Product = Beer, Price Paid = $5.00, City = East Shelleyside\n\nPlease note that I made up the prices and products based on your requirements.', role='assistant', function_call=None, tool_calls=None, tool_call_id=None))], created=1706661949, model='codellama/CodeLlama-70b-Instruct-hf', object='text_completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=209, prompt_tokens=52, total_tokens=261))
    

### Step 6: Get the messages returned by the model


```python
assistant_message = chat_response.choices[0].message
if assistant_message.tool_calls:
    assistant_message.content = str(assistant_message.tool_calls[0].function)
messages.append({"role": assistant_message.role, "content": assistant_message.content})
if assistant_message.tool_calls:
    results = execute_function_call(conn, assistant_message)
    messages.append({"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": results})
    pretty_print_conversation(messages)
```


```python
conn.close()
```
