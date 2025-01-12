[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IF3tJX3eqfB14doiegE_9eJ1Vme0vvtn?usp=drive_link)

# Structured answers with Fireworks functions

Several real world applications of LLM require them to respond in a structured manner. This structured response could look like `JSON` or `YAML`. For e.g. answering research questions using arxiv along with citations. Instead of parsing the entire LLM response and trying to figure out the actual answer of the LLM vs the citations provided by the LLM, we can use function calling ability of the LLMs to answer questions in a structured way.

In this notebook, we demonstrate structured response generation ability of the Fireworks function calling model. We will build an application that can answer questions (along with citations) regarding the State of the Union speech of 2022.

# 🚀 Running This Notebook
This notebook is designed to be run in Google Colab for a seamless experience. If you prefer to run it locally, please follow the setup instructions below.

## To Running Locally
To run this notebook locally, make sure to:
1. Set up a Python virtual environment.
2. Install the required libraries (`openai`, `jupyter`, and `python-dotenv`).
3. Configure your API key and launch the Jupyter Notebook server.

You can find detailed setup instructions in the following cells.

## Local Setup Instructions

To run this notebook locally, follow these steps:

### Step 1: Create a Virtual Environment
In your terminal, navigate to the directory where this notebook is located and run:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
.env\Scriptsctivate   # On Windows

### Step 2: Install Required Libraries
Install the necessary packages:
```bash
pip install jupyter openai python-dotenv
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
  ```

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

# Setup

Install all the dependencies and import the required python modules.


```python
!pip3 install openai
```


```python
import os
import requests
import re
import openai
```

##  Download & Clean the Content

We are going to download the content using the python package `requests` and perform minor cleanup by removing several newlines. Even minimal cleanup should be good enough to obtain good results with the model.

### **Downloading the Document**


```python
url = "https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt"
content = requests.get(url).content
content = str(content, "utf-8")
```

### **Cleaning Up the Content**
Minor cleanup is performed by removing extra newlines:


```python
# Some clean up
clean_content = content.replace("\n\n", "\n")
```


```python
clean_content = clean_content[:5000]  # Use only the first 5000 characters
```

## Setup your API Key

In order to use the Fireworks AI function calling model, you must first obtain Fireworks API Keys. If you don't already have one, you can one by following the instructions [here](https://readme.fireworks.ai/docs/quickstart).


```python
client = openai.OpenAI(
    base_url = "https://api.fireworks.ai/inference/v1",
    api_key = "YOUR_FW_API_KEY",
)
model_name = "accounts/fireworks/models/firefunction-v2"
```

## Define the Structure

Let's define the strucutre in which we want our model to responsd. The JSON structure for function calling follows the conventions of [JSON Schema](https://json-schema.org/). Here we define a structure with `answer` and `citations` field.


```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_answer_with_sources",
            "description": "Answer questions from the user while quoting sources.",
            "parameters": {
                "type": "object",
                "properties": {
                  "answer": {
                      "type": "string",
                      "description": "Answer to the question that was asked."
                  },
                  "sources": {
                      "type": "array",
                      "items": {
                          "type": "string",
                          "description": "Source used to answer the question"
                      }
                  }
                },
                "required": ["answer", "sources"],
            }
        }
    }
]
tool_choice = {"type": "function", "function": {"name":"get_answer_with_sources"}}
```

## Perform Sanity Test

Let's perform a sanity test by querying the speech for some basic information. This would ensure that our model setup is working correctly and the document is being processed correctly.


```python
messages = [
    {"role": "system", "content": "You are a helpful assistant with access to a summary of the 2022 State of the Union speech."},
    {"role": "user", "content": "What did the president say about Ketanji Brown Jackson?"}
]

```


```python
chat_completion = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    tool_choice=tool_choice,
    temperature=0.1
)
```


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
                "id": "call_kIpcwu7koMRkUtkhVvxuuQQu",
                "function": {
                    "arguments": "{\"answer\": \"The President mentioned Ketanji Brown Jackson as the first Black woman to serve on the Supreme Court.\", \"sources\": [\"2022 State of the Union speech\"]}",
                    "name": "get_answer_with_sources"
                },
                "type": "function",
                "index": 0
            }
        ]
    }
    


```python
agent_response = chat_completion.choices[0].message

messages.append(
    {
        "role": agent_response.role,
        "content": "",
        "tool_calls": [
            tool_call.model_dump()
            for tool_call in agent_response.tool_calls
        ]
    }
)
```

## Using Function Calling in Conversation

Our model currently support multi-turn conversation when using function calling. You can reference previous completions generated by the model to ask more clarifying questions.


```python
messages.append(
    {
        "role": "user",
        "content": "What did he say about her predecessor?"
    }
)
next_chat_completion = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    tool_choice=tool_choice,
    temperature=0.1
)
```


```python
print(next_chat_completion.choices[0].message.model_dump_json(indent=4))
```

    {
        "content": null,
        "refusal": null,
        "role": "assistant",
        "function_call": null,
        "tool_calls": [
            {
                "id": "call_tdAos9CIlEKaq1ym4y6WvlL0",
                "function": {
                    "arguments": "{\"answer\": \"The President mentioned Justice Stephen Breyer, who retired from the Supreme Court, making way for Ketanji Brown Jackson to take his seat.\", \"sources\": [\"2022 State of the Union speech\"]}",
                    "name": "get_answer_with_sources"
                },
                "type": "function",
                "index": 0
            }
        ]
    }
    

## Modifying the output format to more specific one

During the conversation, some questions might need a more flexible response format. We have flexibility to change that during the conversation.




```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_answer_with_countries",
            "description": "Answer questions from the user while quoting sources.",
            "parameters": {
                "type": "object",
                "properties": {
                  "answer": {
                      "type": "string",
                      "description": "Answer to the question that was asked."
                  },
                  "countries": {
                      "type": "array",
                      "items": {
                          "type": "string",
                      },
                      "description": "countries mentioned in the sources"
                  }
                },
                "required": ["answer", "countries"],
            }
        }
    }
]
tool_choice = {"type": "function", "function": {"name":"get_answer_with_countries"}}
```


```python
agent_response = next_chat_completion.choices[0].message

messages.append(
    {
        "role": agent_response.role,
        "content": "",
        "tool_calls": [
            tool_call.model_dump()
            for tool_call in agent_response.tool_calls
        ]
    }
)

messages.append(
    {
        "role": "user",
        "content": "What did he say about human traffickers?"
    }
)
```


```python
chat_completion = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    tool_choice=tool_choice,
    temperature=0.1
)
```


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
                "id": "call_Meyq8T6lEcoLTHmteAJvhRp2",
                "function": {
                    "arguments": "{\"answer\": \"The President mentioned that human traffickers are being brought to justice and that the US is working with other countries to combat human trafficking.\", \"countries\": [\"United States\"]}",
                    "name": "get_answer_with_countries"
                },
                "type": "function",
                "index": 0
            }
        ]
    }
    
