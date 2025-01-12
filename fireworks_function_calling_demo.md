[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m7Bk1360CFI50y24KBVxRAKYuEU3pbPU?usp=sharing)

# Fireworks Function Calling API - Demo Notebook

This notebook includes a complete example where the user asks for Nike's net income for 2022. The model decides to call the `get_financial_data` function, and the user manually executes the function call and provides the response.

## Setup Instructions

### **Step 1: Create a Virtual Environment**

To keep your dependencies isolated, create a virtual environment in your terminal:

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
.\venv\Scripts\activate   # On Windows


### Step 2: Install Required Libraries
Install the necessary packages:
```bash
pip install jupyter openai python-dotenv
```


```python
import openai
import json
```

### Step 3: Set Up Your API Key
You can set your API key in the terminal:
- **On macOS/Linux**:
  ```bash
  export FIREWORKS_API_KEY=<YOUR_FIREWORKS_API_KEY>
  ```
- **On Windows**:
  ```bash
  set FIREWORKS_API_KEY=<YOUR_FIREWORKS_API_KEY>

Alternatively, create a `.env` file in the project directory with:
```
FIREWORKS_API_KEY=<YOUR_FIREWORKS_API_KEY>
```

Load the `.env` file in your Python code with:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Step 4: Launch Jupyter Notebook
Start the Jupyter Notebook server:
```bash
jupyter notebook
```
Open this notebook file (`fireworks_demo.ipynb`) and proceed to run the cells.

## How Function Calling Works

The function-calling process involves the following steps:

1. **Define User Query and Tools**: Specify the user query and the available tools using the `messages` and `tools` arguments.
2. **Model Decision**: The model determines whether to respond directly or generate a tool call based on the user query.
3. **User Executes Tool Call**: If the model generates a tool call, the user must execute the function manually and provide the result back to the model.
4. **Response Generation**: The model uses the tool call result to generate a final response.

For more details, refer to:
- [Fireworks Blog Post on FireFunction-v2](https://fireworks.ai/blog/firefunction-v2-launch-post)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)

# 2. Initialize OpenAI Client

The Fireworks API client is initialized using the base URL and the API key.


```python
client = openai.OpenAI(
    base_url = "https://api.fireworks.ai/inference/v1",
    api_key = "<YOUR_FIREWORKS_API_KEY>"
)
```

# 3. Define User and System Messages

- The system message sets the behavior of the assistant.
- The user message asks about Nike's net income for the year 2022.


```python
messages = [
    {"role": "system", "content": f"You are a helpful assistant with access to functions. Use them if required."},
    {"role": "user", "content": "What are Nike's net income in 2022?"}
]
```

# 4. Define Available Tools (Function Metadata)

This defines a get_financial_data function with the required parameters (metric, financial_year, and company). This function is accessible to the model, which can invoke it if needed.


```python
tools = [
    {
        "type": "function",
        "function": {
            # name of the function 
            "name": "get_financial_data",
            # a good, detailed description for what the function is supposed to do
            "description": "Get financial data for a company given the metric and year.",
            # a well defined json schema: https://json-schema.org/learn/getting-started-step-by-step#define
            "parameters": {
                # for OpenAI compatibility, we always declare a top level object for the parameters of the function
                "type": "object",
                # the properties for the object would be any arguments you want to provide to the function
                "properties": {
                    "metric": {
                        # JSON Schema supports string, number, integer, object, array, boolean and null
                        # for more information, please check out https://json-schema.org/understanding-json-schema/reference/type
                        "type": "string",
                        # You can restrict the space of possible values in an JSON Schema
                        # you can check out https://json-schema.org/understanding-json-schema/reference/enum for more examples on how enum works
                        "enum": ["net_income", "revenue", "ebdita"],
                    },
                    "financial_year": {
                        "type": "integer", 
                        # If the model does not understand how it is supposed to fill the field, a good description goes a long way 
                        "description": "Year for which we want to get financial data."
                    },
                    "company": {
                        "type": "string",
                        "description": "Name of the company for which we want to get financial data."
                    }
                },
                # You can specify which of the properties from above are required
                # for more info on `required` field, please check https://json-schema.org/understanding-json-schema/reference/object#required
                "required": ["metric", "financial_year", "company"],
            },
        },
    }
]
```

# 5. Generate a Chat Completion

- The model is called with the defined messages and tools.
- The temperature parameter controls the randomness of the response (0.1 makes it deterministic).
- The response will likely include a function call if the model decides the question requires invoking a tool.


```python
chat_completion = client.chat.completions.create(
    model="accounts/fireworks/models/firefunction-v2",
    messages=messages,
    tools=tools,
    temperature=0.1
)
```

# 6. Inspect the Model's Response

In our case, the model decides to invoke the tool get_financial_data with some specific set of arguments. Note: The model itself won’t invoke the tool. It just specifies the argument. When the model issues a function call - the completion reason would be set to tool_calls. The API caller is responsible for parsing the function name and arguments supplied by the model & invoking the appropriate tool.


```python
print(chat_completion.choices[0].message.model_dump_json(indent=4))
```

    {
        "content": null,
        "refusal": null,
        "role": "assistant",
        "function_call": null,
        "tool_calls": [
            {
                "id": "call_rRqaGO18caS3QXwJR5mW2pQx",
                "function": {
                    "arguments": "{\"metric\": \"net_income\", \"financial_year\": 2022, \"company\": \"Nike\"}",
                    "name": "get_financial_data"
                },
                "type": "function",
                "index": 0
            }
        ]
    }
    


```python
def get_financial_data(metric: str, financial_year: int, company: str):
    print(f"{metric=} {financial_year=} {company=}")
    if metric == "net_income" and financial_year == 2022 and company == "Nike":
        return {"net_income": 6_046_000_000}
    else:
        raise NotImplementedError()

function_call = chat_completion.choices[0].message.tool_calls[0].function
tool_response = locals()[function_call.name](**json.loads(function_call.arguments))
print(tool_response)
```

    metric='net_income' financial_year=2022 company='Nike'
    {'net_income': 6046000000}
    

The API caller obtains the response from the tool invocation & passes its response back to the model for generating a response.


```python
agent_response = chat_completion.choices[0].message
```


```python
# Append the response from the agent
messages.append(
    {
        "role": agent_response.role, 
        "content": "",
        "tool_calls": [
            tool_call.model_dump()
            for tool_call in chat_completion.choices[0].message.tool_calls
        ]
    }
)
```


```python
# Append the response from the tool 
messages.append(
    {
        "role": "tool",
        "content": json.dumps(tool_response)
    }
)
```


```python
next_chat_completion = client.chat.completions.create(
    model="accounts/fireworks/models/firefunction-v2",
    messages=messages,
    tools=tools,
    temperature=0.1
)
```


```python
print(next_chat_completion.choices[0].message.content)
```

    Nike's net income in 2022 was 6.046 billion dollars.
    
