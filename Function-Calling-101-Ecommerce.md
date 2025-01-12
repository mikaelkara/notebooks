# Function Calling 101: An eCommerce Use Case

## 1. Introduction to Function Calling

### 1a. What is function calling and why is it important?

Function calling (or tool use) in the context of large language models (LLMs) is the process of an LLM invoking a pre-defined function instead of generating a text response. LLMs are **non-deterministic**, offering flexibility and creativity, but this can lead to inconsistencies and occasional hallucinations, with the training data often being outdated. In contrast, traditional software is **deterministic**, executing tasks precisely as programmed but lacking adaptability. Function calling with LLMs aims to combine the best of both worlds: leveraging the flexibility and creativity of LLMs while ensuring consistent, repeatable actions and reducing hallucinations by utilizing pre-defined functions.

### 1b. What is it doing?

Function calling essentially arms your LLM with custom tools to perform specific tasks that a generic LLM might struggle with. During an interaction, the LLM determines which tool to call and what parameters to use, allowing it to execute actions it otherwise couldn’t. This enables the LLM to either perform an action directly or relay the function’s output back to itself, providing more context for a follow-up chat completion. By integrating these custom tools, function calling enhances the LLM’s capabilities and precision, enabling more complex and accurate responses.

### 1c. What are some use cases?

Function calling with LLMs can be applied to a variety of practical scenarios, significantly enhancing the capabilities of LLMs. Here are some organized and expanded use cases:


**1. Real-Time Information Retrieval:** LLMs can use function calling to access up-to-date information by querying APIs, databases or search tools, like the [Yahoo Finance API](https://finance.yahoo.com/) or [Tavily Search API](https://tavily.com/). This is particularly useful in domains where information changes frequently, or when you want to surface internal data to the user.

**2. Mathematical Calculations:** LLMs often face challenges with precise mathematical computations. By leveraging function calling, these calculations can be offloaded to specialized functions, ensuring accuracy and reliability.

**3. API Integration for Enhanced Functionality:** Function calling can significantly expand the capabilities of an LLM by integrating it with various APIs. This allows the LLM to perform tasks such as booking appointments, managing calendars, handling customer service requests, and more. By leveraging specific APIs, the LLM can process detailed parameters like appointment times, customer names, contact information, and service details, ensuring efficient and accurate task execution.

## 2. Function Calling Implementation with Groq: eCommerce Use Case

In this notebook, we'll use show how function calling can be used for an eCommerce use case, where our LLM will take on the role of a helpful customer service representative, able to use tools to create orders and get prices on products. We will be interacting as a customer named Tom Testuser.

We will be using [Airtable](https://airtable.com/) as our backend database for this demo, and will use the Airtable API to read and write from `customers`, `products` and `orders` tables. You can view the Airtable base [here](https://airtable.com/appQZ9KdhmjcDVSGx/shrlg9MAetUslmX2Z), but will need to copy it into your own Airtable base (click “copy base” in the upper banner) in order to fully follow along with this guide and build on top of it.


### 2a. Setup

We will be using Meta's Llama 3-70B model for this demo. Note that you will need a Groq API Key to proceed and can create an account [here](https://console.groq.com/) to generate one for free.

You will also need to create an Airtable account and provision an [Airtable Personal Access Token](https://airtable.com/create/tokens) with `data.record:read` and `data.record:write` scopes. The Airtable Base ID will be in the URL of the base you copy from above.

Finally, our System Message will provide relevant context to the LLM: that it is a customer service assistant for an ecommerce company, and that it is interacting with a customer named Tom Testuser (ID: 10).


```python
# Setup
import json
import os
import random
import urllib.parse
from datetime import datetime

import requests
from groq import Groq

# Initialize Groq client and model
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama3-70b-8192"

# Airtable variables
airtable_api_token = os.environ["AIRTABLE_API_TOKEN"]
airtable_base_id = os.environ["AIRTABLE_BASE_ID"]
```


```python
SYSTEM_MESSAGE = """
You are a helpful customer service LLM for an ecommerce company that processes orders and retrieves information about products.
You are currently chatting with Tom Testuser, Customer ID: 10
"""
```

### 2b. Tool Creation

First we must define the functions (tools) that the LLM will have access to. For our use case, we will use the Airtable API to create an order (POST request to the orders table), get product prices (GET request to the products table) and get product ID (GET request to the products table).

We will then compile these tools in a list that can be passed to the LLM. Note that we must provide proper descriptions of the functions and parameters so that they can be called appropriately given the user input:


```python
# Creates an order given a product_id and customer_id
def create_order(product_id, customer_id):
    headers = {
        "Authorization": f"Bearer {airtable_api_token}",
        "Content-Type": "application/json",
    }
    url = f"https://api.airtable.com/v0/{airtable_base_id}/orders"
    order_id = random.randint(1, 100000)  # Randomly assign an order_id
    order_datetime = datetime.utcnow().strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )  # Assign order date as now
    data = {
        "fields": {
            "order_id": order_id,
            "product_id": product_id,
            "customer_id": customer_id,
            "order_date": order_datetime,
        }
    }
    response = requests.post(url, headers=headers, json=data)
    return str(response.json())


# Gets the price for a product, given the name of the product
def get_product_price(product_name):
    api_token = os.environ["AIRTABLE_API_TOKEN"]
    base_id = os.environ["AIRTABLE_BASE_ID"]
    headers = {"Authorization": f"Bearer {airtable_api_token}"}
    formula = f"{{name}}='{product_name}'"
    encoded_formula = urllib.parse.quote(formula)
    url = f"https://api.airtable.com/v0/{airtable_base_id}/products?filterByFormula={encoded_formula}"
    response = requests.get(url, headers=headers)
    product_price = response.json()["records"][0]["fields"]["price"]
    return "$" + str(product_price)


# Gets product ID given a product name
def get_product_id(product_name):
    api_token = os.environ["AIRTABLE_API_TOKEN"]
    base_id = os.environ["AIRTABLE_BASE_ID"]
    headers = {"Authorization": f"Bearer {airtable_api_token}"}
    formula = f"{{name}}='{product_name}'"
    encoded_formula = urllib.parse.quote(formula)
    url = f"https://api.airtable.com/v0/{airtable_base_id}/products?filterByFormula={encoded_formula}"
    response = requests.get(url, headers=headers)
    product_id = response.json()["records"][0]["fields"]["product_id"]
    return str(product_id)
```

The necessary structure to compile our list of tools so that the LLM can use them; note that we must provide proper descriptions of the functions and parameters so that they can be called appropriately given the user input:


```python
tools = [
    # First function: create_order
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": "Creates an order given a product_id and customer_id. If a product name is provided, you must get the product ID first. After placing the order indicate that it was placed successfully and output the details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "The ID of the product",
                    },
                    "customer_id": {
                        "type": "integer",
                        "description": "The ID of the customer",
                    },
                },
                "required": ["product_id", "customer_id"],
            },
        },
    },
    # Second function: get_product_price
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Gets the price for a product, given the name of the product. Just return the price, do not do any calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The name of the product (must be title case, i.e. 'Microphone', 'Laptop')",
                    }
                },
                "required": ["product_name"],
            },
        },
    },
    # Third function: get_product_id
    {
        "type": "function",
        "function": {
            "name": "get_product_id",
            "description": "Gets product ID given a product name",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The name of the product (must be title case, i.e. 'Microphone', 'Laptop')",
                    }
                },
                "required": ["product_name"],
            },
        },
    },
]
```

### 2c. Simple Function Calling

First, let's start out by just making a simple function call with only one tool. We will ask the customer service LLM to place an order for a product with Product ID 5.

The two key parameters we need to include in our chat completion are `tools=tools` and `tool_choice="auto"`, which provides the model with the available tools we've just defined and tells it to use one if appropriate (`tool_choice="auto"` gives the LLM the option of using any, all or none of the available functions. To mandate a specific function call, we could use `tool_choice={"type": "function", "function": {"name":"create_order"}}`). 

When the LLM decides to use a tool, the response is *not* a conversational chat, but a JSON object containing the tool choice and tool parameters. From there, we can execute the LLM-identified tool with the LLM-identified parameters, and feed the response *back* to the LLM for a second request so that it can respond with appropriate context from the tool it just used:


```python
user_prompt = "Please place an order for Product ID 5"
messages = [
    {"role": "system", "content": SYSTEM_MESSAGE},
    {
        "role": "user",
        "content": user_prompt,
    },
]

# Step 1: send the conversation and available functions to the model
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
    tool_choice="auto",  # Let the LLM decide if it should use one of the available tools
    max_tokens=4096,
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls
print("First LLM Call (Tool Use) Response:", response_message)
# Step 2: check if the model wanted to call a function
if tool_calls:
    # Step 3: call the function and append the tool call to our list of messages
    available_functions = {
        "create_order": create_order,
    }
    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                    "type": tool_call.type,
                }
                for tool_call in tool_calls
            ],
        }
    )
    # Step 4: send the info for each function call and function response to the model
    tool_call = tool_calls[0]
    function_name = tool_call.function.name
    function_to_call = available_functions[function_name]
    function_args = json.loads(tool_call.function.arguments)
    function_response = function_to_call(
        product_id=function_args.get("product_id"),
        customer_id=function_args.get("customer_id"),
    )
    messages.append(
        {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response,
        }
    )  # extend conversation with function response
    # Send the result back to the LLM to complete the chat
    second_response = client.chat.completions.create(
        model=MODEL, messages=messages
    )  # get a new response from the model where it can see the function response
    print("\n\nSecond LLM Call Response:", second_response.choices[0].message.content)
```

    First LLM Call (Tool Use) Response: ChoiceMessage(content=None, role='assistant', tool_calls=[ChoiceMessageToolCall(id='call_cnyc', function=ChoiceMessageToolCallFunction(arguments='{"customer_id":10,"product_id":5}', name='create_order'), type='function')])
    
    
    Second LLM Call Response: Your order has been successfully placed!
    
    Order details:
    
    * Order ID: 24255
    * Product ID: 5
    * Customer ID: 10 (that's you, Tom Testuser!)
    * Order Date: 2024-05-31 13:59:03
    
    We'll process your order shortly. You'll receive an email with further updates on your order status. If you have any questions or concerns, feel free to ask!
    

Here is the entire message sequence for a simple tool call:


```python
print(json.dumps(messages, indent=2))
```

    [
      {
        "role": "system",
        "content": "\nYou are a helpful customer service LLM for an ecommerce company that processes orders and retrieves information about products.\nYou are currently chatting with Tom Testuser, Customer ID: 10\n"
      },
      {
        "role": "user",
        "content": "Please place an order for Product ID 5"
      },
      {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_cnyc",
            "function": {
              "name": "create_order",
              "arguments": "{\"customer_id\":10,\"product_id\":5}"
            },
            "type": "function"
          }
        ]
      },
      {
        "tool_call_id": "call_cnyc",
        "role": "tool",
        "name": "create_order",
        "content": "{'id': 'recWasb2AECLJiRj1', 'createdTime': '2024-05-31T13:59:04.000Z', 'fields': {'order_id': 24255, 'product_id': 5, 'customer_id': 10, 'order_date': '2024-05-31T13:59:03.000Z'}}"
      }
    ]
    

### 2d. Parallel Tool Use

If we need multiple function calls that **do not** depend on each other, we can run them in parallel - meaning, multiple function calls will be identified within a single chat request. Here, we are asking for the price of both a Laptop and a Microphone, which requires multiple calls of the `get_product_price` function. Note that in using parallel tool use, *the LLM itself* will decide if it needs to make multiple function calls. So we don't need to make any changes to our chat completion code, but *do* need to be able to iterate over multiple tool calls after the tools are identified.

*parallel tool use is only available for Llama-based models at this time (5/27/2024)*


```python
user_prompt = "Please get the price for the Laptop and Microphone"
messages = [
    {"role": "system", "content": SYSTEM_MESSAGE},
    {
        "role": "user",
        "content": user_prompt,
    },
]

# Step 1: send the conversation and available functions to the model
response = client.chat.completions.create(
    model=MODEL, messages=messages, tools=tools, tool_choice="auto", max_tokens=4096
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls
print("First LLM Call (Tool Use) Response:", response_message)
# Step 2: check if the model wanted to call a function
if tool_calls:
    # Step 3: call the function and append the tool call to our list of messages
    available_functions = {
        "get_product_price": get_product_price,
    }  # only one function in this example, but you can have multiple
    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                    "type": tool_call.type,
                }
                for tool_call in tool_calls
            ],
        }
    )
    # Step 4: send the info for each function call and function response to the model
    # Iterate over all tool calls
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(
            product_name=function_args.get("product_name")
        )
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
    second_response = client.chat.completions.create(
        model=MODEL, messages=messages
    )  # get a new response from the model where it can see the function response
    print("\n\nSecond LLM Call Response:", second_response.choices[0].message.content)
```

    First LLM Call (Tool Use) Response: ChoiceMessage(content=None, role='assistant', tool_calls=[ChoiceMessageToolCall(id='call_88r0', function=ChoiceMessageToolCallFunction(arguments='{"product_name":"Laptop"}', name='get_product_price'), type='function'), ChoiceMessageToolCall(id='call_vva6', function=ChoiceMessageToolCallFunction(arguments='{"product_name":"Microphone"}', name='get_product_price'), type='function')])
    
    
    Second LLM Call Response: So, the price of the Laptop is $753.03 and the price of the Microphone is $276.23. The total comes out to be $1,029.26.
    

Here is the entire message sequence for a parallel tool call:


```python
print(json.dumps(messages, indent=2))
```

    [
      {
        "role": "system",
        "content": "\nYou are a helpful customer service LLM for an ecommerce company that processes orders and retrieves information about products.\nYou are currently chatting with Tom Testuser, Customer ID: 10\n"
      },
      {
        "role": "user",
        "content": "Please get the price for the Laptop and Microphone"
      },
      {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_88r0",
            "function": {
              "name": "get_product_price",
              "arguments": "{\"product_name\":\"Laptop\"}"
            },
            "type": "function"
          },
          {
            "id": "call_vva6",
            "function": {
              "name": "get_product_price",
              "arguments": "{\"product_name\":\"Microphone\"}"
            },
            "type": "function"
          }
        ]
      },
      {
        "tool_call_id": "call_88r0",
        "role": "tool",
        "name": "get_product_price",
        "content": "$753.03"
      },
      {
        "tool_call_id": "call_vva6",
        "role": "tool",
        "name": "get_product_price",
        "content": "$276.23"
      }
    ]
    

### 2e. Multiple Tool Use

Multiple Tool Use is for when we need to use multiple functions where the input to one of the functions **depends on the output** of another function. Unlike parallel tool use, with multiple tool use we will only output a single tool call per LLM request, and then make a separate LLM request to call the next tool. To do this, we'll add a WHILE loop to continuously send LLM requests with our updated message sequence until it has enough information to no longer need to call any more tools. (Note that this solution is generalizable to both simple and parallel tool calling as well).

In our first example we invoked the `create_order` function by providing the product ID directly; since that is a bit clunky, we will first use the `get_product_id` function to get the product ID associated with the product name, then use that ID to call `create_order`:


```python
user_prompt = "Please place an order for a Microphone"
messages = [
    {"role": "system", "content": SYSTEM_MESSAGE},
    {
        "role": "user",
        "content": user_prompt,
    },
]
# Continue to make LLM calls until it no longer decides to use a tool
tool_call_identified = True
while tool_call_identified:
    response = client.chat.completions.create(
        model=MODEL, messages=messages, tools=tools, tool_choice="auto", max_tokens=4096
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        print("LLM Call (Tool Use) Response:", response_message)
        # Step 3: call the function and append the tool call to our list of messages
        available_functions = {
            "create_order": create_order,
            "get_product_id": get_product_id,
        }
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                        "type": tool_call.type,
                    }
                    for tool_call in tool_calls
                ],
            }
        )

        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            if function_name == "get_product_id":
                function_response = function_to_call(
                    product_name=function_args.get("product_name")
                )
            elif function_name == "create_order":
                function_response = function_to_call(
                    customer_id=function_args.get("customer_id"),
                    product_id=function_args.get("product_id"),
                )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
    else:
        print("\n\nFinal LLM Call Response:", response.choices[0].message.content)
        tool_call_identified = False
```

    LLM Call (Tool Use) Response: ChoiceMessage(content=None, role='assistant', tool_calls=[ChoiceMessageToolCall(id='call_6yd2', function=ChoiceMessageToolCallFunction(arguments='{"product_name":"Microphone"}', name='get_product_id'), type='function')])
    LLM Call (Tool Use) Response: ChoiceMessage(content=None, role='assistant', tool_calls=[ChoiceMessageToolCall(id='call_mnv6', function=ChoiceMessageToolCallFunction(arguments='{"customer_id":10,"product_id":15}', name='create_order'), type='function')])
    
    
    Final LLM Call Response: Your order with ID 42351 has been successfully placed! The details are: product ID 15, customer ID 10, and order date 2024-05-31T13:59:40.000Z.
    

Here is the entire message sequence for a multiple tool call:


```python
print(json.dumps(messages, indent=2))
```

    [
      {
        "role": "system",
        "content": "\nYou are a helpful customer service LLM for an ecommerce company that processes orders and retrieves information about products.\nYou are currently chatting with Tom Testuser, Customer ID: 10\n"
      },
      {
        "role": "user",
        "content": "Please place an order for a Microphone"
      },
      {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_6yd2",
            "function": {
              "name": "get_product_id",
              "arguments": "{\"product_name\":\"Microphone\"}"
            },
            "type": "function"
          }
        ]
      },
      {
        "tool_call_id": "call_6yd2",
        "role": "tool",
        "name": "get_product_id",
        "content": "15"
      },
      {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_mnv6",
            "function": {
              "name": "create_order",
              "arguments": "{\"customer_id\":10,\"product_id\":15}"
            },
            "type": "function"
          }
        ]
      },
      {
        "tool_call_id": "call_mnv6",
        "role": "tool",
        "name": "create_order",
        "content": "{'id': 'rectr27e5TP1UMREM', 'createdTime': '2024-05-31T13:59:41.000Z', 'fields': {'order_id': 42351, 'product_id': 15, 'customer_id': 10, 'order_date': '2024-05-31T13:59:40.000Z'}}"
      }
    ]
    

### 2f. Langchain Integration

Finally, Groq function calling is compatible with [Langchain](https://python.langchain.com/v0.1/docs/modules/tools/), by converting your functions into Langchain tools. Here is an example using our `get_product_price` function:


```python
from langchain_groq import ChatGroq

llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model=MODEL)
```

When defining Langchain tools, put the function description as a string at the beginning of the function


```python
from langchain_core.tools import tool

@tool
def create_order(product_id, customer_id):
    """
    Creates an order given a product_id and customer_id.
    If a product name is provided, you must get the product ID first.
    After placing the order indicate that it was placed successfully and output the details.

    product_id: ID of the product
    customer_id: ID of the customer
    """
    api_token = os.environ["AIRTABLE_API_TOKEN"]
    base_id = os.environ["AIRTABLE_BASE_ID"]
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    url = f"https://api.airtable.com/v0/{base_id}/orders"
    order_id = random.randint(1, 100000)  # Randomly assign an order_id
    order_datetime = datetime.utcnow().strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )  # Assign order date as now
    data = {
        "fields": {
            "order_id": order_id,
            "product_id": product_id,
            "customer_id": customer_id,
            "order_date": order_datetime,
        }
    }
    response = requests.post(url, headers=headers, json=data)
    return str(response.json())


@tool
def get_product_price(product_name):
    """
    Gets the price for a product, given the name of the product.
    Just return the price, do not do any calculations.

    product_name: The name of the product (must be title case, i.e. 'Microphone', 'Laptop')
    """
    api_token = os.environ["AIRTABLE_API_TOKEN"]
    base_id = os.environ["AIRTABLE_BASE_ID"]
    headers = {"Authorization": f"Bearer {api_token}"}
    formula = f"{{name}}='{product_name}'"
    encoded_formula = urllib.parse.quote(formula)
    url = f"https://api.airtable.com/v0/{base_id}/products?filterByFormula={encoded_formula}"
    response = requests.get(url, headers=headers)
    product_price = response.json()["records"][0]["fields"]["price"]
    return "$" + str(product_price)


@tool
def get_product_id(product_name):
    """
    Gets product ID given a product name

    product_name: The name of the product (must be title case, i.e. 'Microphone', 'Laptop')
    """
    api_token = os.environ["AIRTABLE_API_TOKEN"]
    base_id = os.environ["AIRTABLE_BASE_ID"]
    headers = {"Authorization": f"Bearer {api_token}"}
    formula = f"{{name}}='{product_name}'"
    encoded_formula = urllib.parse.quote(formula)
    url = f"https://api.airtable.com/v0/{base_id}/products?filterByFormula={encoded_formula}"
    response = requests.get(url, headers=headers)
    product_id = response.json()["records"][0]["fields"]["product_id"]
    return str(product_id)


# Add tools to our LLM
tools = [create_order, get_product_price, get_product_id]
llm_with_tools = llm.bind_tools(tools)

```


```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

user_prompt = "Please place an order for a Microphone"
print(llm_with_tools.invoke(user_prompt).tool_calls)
```

    [{'name': 'get_product_id', 'args': {'product_name': 'Microphone'}, 'id': 'call_7f8y'}, {'name': 'create_order', 'args': {'product_id': '{result of get_product_id}', 'customer_id': ''}, 'id': 'call_zt5c'}]
    


```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

available_tools = {
    "create_order": create_order,
    "get_product_price": get_product_price,
    "get_product_id": get_product_id,
}
messages = [SystemMessage(SYSTEM_MESSAGE), HumanMessage(user_prompt)]
tool_call_identified = True
while tool_call_identified:
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        selected_tool = available_tools[tool_call["name"]]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    if len(ai_msg.tool_calls) == 0:
        tool_call_identified = False

print(ai_msg.content)
```

    Your order has been placed successfully! Your order ID is 87812.
    
