# Exploring OpenAI V1 functionality

On 11.06.23 OpenAI released a number of new features, and along with it bumped their Python SDK to 1.0.0. This notebook shows off the new features and how to use them with LangChain.


```python
# need openai>=1.1.0, langchain>=0.0.335, langchain-experimental>=0.0.39
!pip install -U openai langchain langchain-experimental
```


```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
```

## [Vision](https://platform.openai.com/docs/guides/vision)

OpenAI released multi-modal models, which can take a sequence of text and images as input.


```python
chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=256)
chat.invoke(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "What is this image showing"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/langchain_stack.png",
                        "detail": "auto",
                    },
                },
            ]
        )
    ]
)
```




    AIMessage(content='The image appears to be a diagram representing the architecture or components of a software system or framework related to language processing, possibly named LangChain or associated with a project or product called LangChain, based on the prominent appearance of that term. The diagram is organized into several layers or aspects, each containing various elements or modules:\n\n1. **Protocol**: This may be the foundational layer, which includes "LCEL" and terms like parallelization, fallbacks, tracing, batching, streaming, async, and composition. These seem related to communication and execution protocols for the system.\n\n2. **Integrations Components**: This layer includes "Model I/O" with elements such as the model, output parser, prompt, and example selector. It also has a "Retrieval" section with a document loader, retriever, embedding model, vector store, and text splitter. Lastly, there\'s an "Agent Tooling" section. These components likely deal with the interaction with external data, models, and tools.\n\n3. **Application**: The application layer features "LangChain" with chains, agents, agent executors, and common application logic. This suggests that the system uses a modular approach with chains and agents to process language tasks.\n\n4. **Deployment**: This contains "Lang')



## [OpenAI assistants](https://platform.openai.com/docs/assistants/overview)

> The Assistants API allows you to build AI assistants within your own applications. An Assistant has instructions and can leverage models, tools, and knowledge to respond to user queries. The Assistants API currently supports three types of tools: Code Interpreter, Retrieval, and Function calling


You can interact with OpenAI Assistants using OpenAI tools or custom tools. When using exclusively OpenAI tools, you can just invoke the assistant directly and get final answers. When using custom tools, you can run the assistant and tool execution loop using the built-in AgentExecutor or easily write your own executor.

Below we show the different ways to interact with Assistants. As a simple example, let's build a math tutor that can write and run code.

### Using only OpenAI tools


```python
from langchain.agents.openai_assistant import OpenAIAssistantRunnable
```


```python
interpreter_assistant = OpenAIAssistantRunnable.create_assistant(
    name="langchain assistant",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview",
)
output = interpreter_assistant.invoke({"content": "What's 10 - 4 raised to the 2.7"})
output
```




    [ThreadMessage(id='msg_g9OJv0rpPgnc3mHmocFv7OVd', assistant_id='asst_hTwZeNMMphxzSOqJ01uBMsJI', content=[MessageContentText(text=Text(annotations=[], value='The result of \\(10 - 4^{2.7}\\) is approximately \\(-32.224\\).'), type='text')], created_at=1699460600, file_ids=[], metadata={}, object='thread.message', role='assistant', run_id='run_nBIT7SiAwtUfSCTrQNSPLOfe', thread_id='thread_14n4GgXwxgNL0s30WJW5F6p0')]



### As a LangChain agent with arbitrary tools

Now let's recreate this functionality using our own tools. For this example we'll use the [E2B sandbox runtime tool](https://e2b.dev/docs?ref=landing-page-get-started).


```python
!pip install e2b duckduckgo-search
```


```python
from langchain.tools import DuckDuckGoSearchRun, E2BDataAnalysisTool

tools = [E2BDataAnalysisTool(api_key="..."), DuckDuckGoSearchRun()]
```


```python
agent = OpenAIAssistantRunnable.create_assistant(
    name="langchain assistant e2b tool",
    instructions="You are a personal math tutor. Write and run code to answer math questions. You can also search the internet.",
    tools=tools,
    model="gpt-4-1106-preview",
    as_agent=True,
)
```

#### Using AgentExecutor


```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools)
agent_executor.invoke({"content": "What's the weather in SF today divided by 2.7"})
```




    {'content': "What's the weather in SF today divided by 2.7",
     'output': "The weather in San Francisco today is reported to have temperatures as high as 66 °F. To get the temperature divided by 2.7, we will calculate that:\n\n66 °F / 2.7 = 24.44 °F\n\nSo, when the high temperature of 66 °F is divided by 2.7, the result is approximately 24.44 °F. Please note that this doesn't have a meteorological meaning; it's purely a mathematical operation based on the given temperature."}



#### Custom execution


```python
agent = OpenAIAssistantRunnable.create_assistant(
    name="langchain assistant e2b tool",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=tools,
    model="gpt-4-1106-preview",
    as_agent=True,
)
```


```python
from langchain_core.agents import AgentFinish


def execute_agent(agent, tools, input):
    tool_map = {tool.name: tool for tool in tools}
    response = agent.invoke(input)
    while not isinstance(response, AgentFinish):
        tool_outputs = []
        for action in response:
            tool_output = tool_map[action.tool].invoke(action.tool_input)
            print(action.tool, action.tool_input, tool_output, end="\n\n")
            tool_outputs.append(
                {"output": tool_output, "tool_call_id": action.tool_call_id}
            )
        response = agent.invoke(
            {
                "tool_outputs": tool_outputs,
                "run_id": action.run_id,
                "thread_id": action.thread_id,
            }
        )

    return response
```


```python
response = execute_agent(agent, tools, {"content": "What's 10 - 4 raised to the 2.7"})
print(response.return_values["output"])
```

    e2b_data_analysis {'python_code': 'print(10 - 4 ** 2.7)'} {"stdout": "-32.22425314473263", "stderr": "", "artifacts": []}
    
    \( 10 - 4^{2.7} \) is approximately \(-32.22425314473263\).
    


```python
next_response = execute_agent(
    agent, tools, {"content": "now add 17.241", "thread_id": response.thread_id}
)
print(next_response.return_values["output"])
```

    e2b_data_analysis {'python_code': 'result = 10 - 4 ** 2.7\nprint(result + 17.241)'} {"stdout": "-14.983253144732629", "stderr": "", "artifacts": []}
    
    When you add \( 17.241 \) to \( 10 - 4^{2.7} \), the result is approximately \( -14.98325314473263 \).
    

## [JSON mode](https://platform.openai.com/docs/guides/text-generation/json-mode)

Constrain the model to only generate valid JSON. Note that you must include a system message with instructions to use JSON for this mode to work.

Only works with certain models. 


```python
chat = ChatOpenAI(model="gpt-3.5-turbo-1106").bind(
    response_format={"type": "json_object"}
)

output = chat.invoke(
    [
        SystemMessage(
            content="Extract the 'name' and 'origin' of any companies mentioned in the following statement. Return a JSON list."
        ),
        HumanMessage(
            content="Google was founded in the USA, while Deepmind was founded in the UK"
        ),
    ]
)
print(output.content)
```


```python
import json

json.loads(output.content)
```

## [System fingerprint](https://platform.openai.com/docs/guides/text-generation/reproducible-outputs)

OpenAI sometimes changes model configurations in a way that impacts outputs. Whenever this happens, the system_fingerprint associated with a generation will change.


```python
chat = ChatOpenAI(model="gpt-3.5-turbo-1106")
output = chat.generate(
    [
        [
            SystemMessage(
                content="Extract the 'name' and 'origin' of any companies mentioned in the following statement. Return a JSON list."
            ),
            HumanMessage(
                content="Google was founded in the USA, while Deepmind was founded in the UK"
            ),
        ]
    ]
)
print(output.llm_output)
```

## Breaking changes to Azure classes

OpenAI V1 rewrote their clients and separated Azure and OpenAI clients. This has led to some changes in LangChain interfaces when using OpenAI V1.

BREAKING CHANGES:
- To use Azure embeddings with OpenAI V1, you'll need to use the new `AzureOpenAIEmbeddings` instead of the existing `OpenAIEmbeddings`. `OpenAIEmbeddings` continue to work when using Azure with `openai<1`.
```python
from langchain_openai import AzureOpenAIEmbeddings
```


RECOMMENDED CHANGES:
- When using `AzureChatOpenAI` or `AzureOpenAI`, if passing in an Azure endpoint (eg https://example-resource.azure.openai.com/) this should be specified via the `azure_endpoint` parameter or the `AZURE_OPENAI_ENDPOINT`. We're maintaining backwards compatibility for now with specifying this via `openai_api_base`/`base_url` or env var `OPENAI_API_BASE` but this shouldn't be relied upon.
- When using Azure chat or embedding models, pass in API keys either via `openai_api_key` parameter or `AZURE_OPENAI_API_KEY` parameter. We're maintaining backwards compatibility for now with specifying this via `OPENAI_API_KEY` but this shouldn't be relied upon.

## Tools

Use tools for parallel function calling.


```python
from typing import Literal

from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


class GetCurrentWeather(BaseModel):
    """Get the current weather in a location."""

    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="fahrenheit", description="The temperature unit, default to fahrenheit"
    )


prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant"), ("user", "{input}")]
)
model = ChatOpenAI(model="gpt-3.5-turbo-1106").bind(
    tools=[convert_pydantic_to_openai_tool(GetCurrentWeather)]
)
chain = prompt | model | PydanticToolsParser(tools=[GetCurrentWeather])

chain.invoke({"input": "what's the weather in NYC, LA, and SF"})
```




    [GetCurrentWeather(location='New York, NY', unit='fahrenheit'),
     GetCurrentWeather(location='Los Angeles, CA', unit='fahrenheit'),
     GetCurrentWeather(location='San Francisco, CA', unit='fahrenheit')]


