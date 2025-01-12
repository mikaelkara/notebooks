# Create a Transformers Agent from any LLM inference provider
_Authored by: [Aymeric Roucher](https://huggingface.co/m-ric)_

> This tutorial builds upon agent knowledge: to know more about agents, you can start with [this introductory notebook](agents)

[Transformers Agents](https://huggingface.co/docs/transformers/en/agents) is a library to build agents, using an LLM to power it in the `llm_engine` argument. This argument was designed to leave the user maximal freedom to choose any LLM.

Let's see how to build this `llm_engine` from the APIs of a few leading providers.

## HuggingFace Serverless API and Dedicated Endpoints

Transformers agents provides a built-in `HfEngine` class that lets you use any model on the Hub via the Serverless API or your own dedicated Endpoint. This is the preferred way to use HF agents.


```python
!pip install openai anthropic "transformers[agents]" --upgrade -q
```


```python
from huggingface_hub import notebook_login

notebook_login()
```


```python
from transformers.agents import HfApiEngine, ReactCodeAgent

repo_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
endpoint_url = "your_endpoint_url"

llm_engine = HfApiEngine(model=repo_id)  # you could use model=endpoint_url here

agent = ReactCodeAgent(tools=[], llm_engine=llm_engine)

agent.run("What's the 10th Fibonacci number?")
```

    [33;1m======== New task ========[0m
    [37;1mWhat's the 10th Fibonacci number?[0m
    

    ['unicodedata', 're', 'math', 'collections', 'queue', 'itertools', 'random', 'time', 'stat', 'statistics']
    

    [33;1m==== Agent is executing the code below:[0m
    [0m[38;5;7ma[39m[38;5;7m,[39m[38;5;7m [39m[38;5;7mb[39m[38;5;7m [39m[38;5;109;01m=[39;00m[38;5;7m [39m[38;5;139m0[39m[38;5;7m,[39m[38;5;7m [39m[38;5;139m1[39m
    [38;5;109;01mfor[39;00m[38;5;7m [39m[38;5;7m_[39m[38;5;7m [39m[38;5;109;01min[39;00m[38;5;7m [39m[38;5;109mrange[39m[38;5;7m([39m[38;5;139m9[39m[38;5;7m)[39m[38;5;7m:[39m
    [38;5;7m    [39m[38;5;7ma[39m[38;5;7m,[39m[38;5;7m [39m[38;5;7mb[39m[38;5;7m [39m[38;5;109;01m=[39;00m[38;5;7m [39m[38;5;7mb[39m[38;5;7m,[39m[38;5;7m [39m[38;5;7ma[39m[38;5;7m [39m[38;5;109;01m+[39;00m[38;5;7m [39m[38;5;7mb[39m
    [38;5;109mprint[39m[38;5;7m([39m[38;5;7mb[39m[38;5;7m)[39m[0m
    [33;1m====[0m
    [33;1mPrint outputs:[0m
    [32;20m55
    [0m
    [33;1m==== Agent is executing the code below:[0m
    [0m[38;5;7ma[39m[38;5;7m,[39m[38;5;7m [39m[38;5;7mb[39m[38;5;7m [39m[38;5;109;01m=[39;00m[38;5;7m [39m[38;5;139m0[39m[38;5;7m,[39m[38;5;7m [39m[38;5;139m1[39m
    [38;5;109;01mfor[39;00m[38;5;7m [39m[38;5;7m_[39m[38;5;7m [39m[38;5;109;01min[39;00m[38;5;7m [39m[38;5;109mrange[39m[38;5;7m([39m[38;5;139m9[39m[38;5;7m)[39m[38;5;7m:[39m
    [38;5;7m    [39m[38;5;7ma[39m[38;5;7m,[39m[38;5;7m [39m[38;5;7mb[39m[38;5;7m [39m[38;5;109;01m=[39;00m[38;5;7m [39m[38;5;7mb[39m[38;5;7m,[39m[38;5;7m [39m[38;5;7ma[39m[38;5;7m [39m[38;5;109;01m+[39;00m[38;5;7m [39m[38;5;7mb[39m[0m
    [33;1m====[0m
    [33;1mPrint outputs:[0m
    [32;20m[0m
    [33;1m==== Agent is executing the code below:[0m
    [0m[38;5;7mfinal_answer[39m[38;5;7m([39m[38;5;7mb[39m[38;5;7m)[39m[0m
    [33;1m====[0m
    [33;1mPrint outputs:[0m
    [32;20m[0m
    [33;1m>>> Final answer:[0m
    [32;20m55[0m
    




    55



The `llm_engine` initialization arg of the agent could be a simple callable such as:
```py
def llm_engine(messages, stop_sequences=[]) -> str:
    return response(messages)
```
This callable is the heart of the llm engine. It should respect these requirements:
- takes as input a list of messages in [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating) format and outputs a `str`.
- accepts a `stop_sequences` argument where the agent system will pass it sequences where it should stop generation.

Let's take a closer look at the code for the `HfEngine` that we used:


```python
from typing import List, Dict
from transformers.agents.llm_engine import MessageRole, get_clean_message_list
from huggingface_hub import InferenceClient

llama_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


class HfApiEngine:
    def __init__(self, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model = model
        self.client = InferenceClient(model=self.model, timeout=120)

    def __call__(self, messages: List[Dict[str, str]], stop_sequences=[]) -> str:
        # Get clean message list
        messages = get_clean_message_list(
            messages, role_conversions=llama_role_conversions
        )

        # Get LLM output
        response = self.client.chat_completion(
            messages, stop=stop_sequences, max_tokens=1500
        )
        response = response.choices[0].message.content

        # Remove stop sequences from LLM output
        for stop_seq in stop_sequences:
            if response[-len(stop_seq) :] == stop_seq:
                response = response[: -len(stop_seq)]
        return response
```

Here the engine is not a function, but a class with a `__call__` method, which adds the possibility to store attributes such as the client.

We also use `get_clean_message_list()` utility to concatenate successive messages to the same role
This method takes a `role_conversions` arg to convert the range of roles supported in Transformers Agents to only the ones accepted by your LLM.


This recipe can be adapted for any LLM! Let's look at other examples.

## Adapting the recipe for any LLM

Using the above recipe, you can use any LLM inference source as your `llm_engine`.
Just keep in mind the two main constraints:
- `llm_engine` is a callable that takes as input a list of messages in [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating) format and outputs a `str`.
- It accepts a `stop_sequences` argument.


### OpenAI


```python
import os
from openai import OpenAI

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


class OpenAIEngine:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(
            messages, role_conversions=openai_role_conversions
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=0.5,
        )
        return response.choices[0].message.content
```

### Anthropic


```python
from anthropic import Anthropic, AnthropicBedrock


# Cf this page for using Anthropic from Bedrock: https://docs.anthropic.com/en/api/claude-on-amazon-bedrock
class AnthropicEngine:
    def __init__(self, model_name="claude-3-5-sonnet-20240620", use_bedrock=False):
        self.model_name = model_name
        if use_bedrock:
            self.model_name = "anthropic.claude-3-5-sonnet-20240620-v1:0"
            self.client = AnthropicBedrock(
                aws_access_key=os.getenv("AWS_BEDROCK_ID"),
                aws_secret_key=os.getenv("AWS_BEDROCK_KEY"),
                aws_region="us-east-1",
            )
        else:
            self.client = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(
            messages, role_conversions=openai_role_conversions
        )
        index_system_message, system_prompt = None, None
        for index, message in enumerate(messages):
            if message["role"] == MessageRole.SYSTEM:
                index_system_message = index
                system_prompt = message["content"]
        if system_prompt is None:
            raise Exception("No system prompt found!")

        filtered_messages = [
            message for i, message in enumerate(messages) if i != index_system_message
        ]
        if len(filtered_messages) == 0:
            print("Error, no user message:", messages)
            assert False

        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=filtered_messages,
            stop_sequences=stop_sequences,
            temperature=0.5,
            max_tokens=2000,
        )
        full_response_text = ""
        for content_block in response.content:
            if content_block.type == "text":
                full_response_text += content_block.text
        return full_response_text
```

### Next steps

Go on and implement your `llm_engine` for `transformers.agents` with your own LLM inference provider!

Then to use this shiny new `llm_engine`, check out these use cases:
- [Agentic RAG: turbocharge your RAG with query reformulation and self-query](agent_rag)
- [Agent for text-to-SQL with automatic error correction](agent_text_to_sql)
