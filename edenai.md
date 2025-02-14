# Eden AI

Eden AI is revolutionizing the AI landscape by uniting the best AI providers, empowering users to unlock limitless possibilities and tap into the true potential of artificial intelligence. With an all-in-one comprehensive and hassle-free platform, it allows users to deploy AI features to production lightning fast, enabling effortless access to the full breadth of AI capabilities via a single API. (website: https://edenai.co/)

This example goes over how to use LangChain to interact with Eden AI models

-----------------------------------------------------------------------------------

`EdenAI` goes beyond mere model invocation. It empowers you with advanced features, including:

- **Multiple Providers**: Gain access to a diverse range of language models offered by various providers, giving you the freedom to choose the best-suited model for your use case.

- **Fallback Mechanism**: Set a fallback mechanism to ensure seamless operations even if the primary provider is unavailable, you can easily switches to an alternative provider.

- **Usage Tracking**: Track usage statistics on a per-project and per-API key basis. This feature allows you to monitor and manage resource consumption effectively.

- **Monitoring and Observability**: `EdenAI` provides comprehensive monitoring and observability tools on the platform. Monitor the performance of your language models, analyze usage patterns, and gain valuable insights to optimize your applications.


Accessing the EDENAI's API requires an API key, 

which you can get by creating an account https://app.edenai.run/user/register  and heading here https://app.edenai.run/admin/iam/api-keys

Once we have a key we'll want to set it as an environment variable by running:

```bash
export EDENAI_API_KEY="..."
```

You can find more details on the API reference : https://docs.edenai.co/reference

If you'd prefer not to set an environment variable you can pass the key in directly via the edenai_api_key named parameter

 when initiating the EdenAI Chat Model class.


```python
from langchain_community.chat_models.edenai import ChatEdenAI
from langchain_core.messages import HumanMessage
```


```python
chat = ChatEdenAI(
    edenai_api_key="...", provider="openai", temperature=0.2, max_tokens=250
)
```


```python
messages = [HumanMessage(content="Hello !")]
chat.invoke(messages)
```




    AIMessage(content='Hello! How can I assist you today?')




```python
await chat.ainvoke(messages)
```




    AIMessage(content='Hello! How can I assist you today?')



## Streaming and Batching

`ChatEdenAI` supports streaming and batching. Below is an example.


```python
for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)
```

    Hello! How can I assist you today?


```python
chat.batch([messages])
```




    [AIMessage(content='Hello! How can I assist you today?')]



## Fallback mecanism

With Eden AI you can set a fallback mechanism to ensure seamless operations even if the primary provider is unavailable, you can easily switches to an alternative provider.


```python
chat = ChatEdenAI(
    edenai_api_key="...",
    provider="openai",
    temperature=0.2,
    max_tokens=250,
    fallback_providers="google",
)
```

In this example, you can use Google as a backup provider if OpenAI encounters any issues.

For more information and details about Eden AI, check out this link: : https://docs.edenai.co/docs/additional-parameters

## Chaining Calls



```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "What is a good name for a company that makes {product}?"
)
chain = prompt | chat
```


```python
chain.invoke({"product": "healthy snacks"})
```




    AIMessage(content='VitalBites')



## Tools

### bind_tools()

With `ChatEdenAI.bind_tools`, we can easily pass in Pydantic classes, dict schemas, LangChain tools, or even functions as tools to the model.


```python
from pydantic import BaseModel, Field

llm = ChatEdenAI(provider="openai", temperature=0.2, max_tokens=500)


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([GetWeather])
```


```python
ai_msg = llm_with_tools.invoke(
    "what is the weather like in San Francisco",
)
ai_msg
```




    AIMessage(content='', response_metadata={'openai': {'status': 'success', 'generated_text': None, 'message': [{'role': 'user', 'message': 'what is the weather like in San Francisco', 'tools': [{'name': 'GetWeather', 'description': 'Get the current weather in a given location', 'parameters': {'type': 'object', 'properties': {'location': {'description': 'The city and state, e.g. San Francisco, CA', 'type': 'string'}}, 'required': ['location']}}], 'tool_calls': None}, {'role': 'assistant', 'message': None, 'tools': None, 'tool_calls': [{'id': 'call_tRpAO7KbQwgTjlka70mCQJdo', 'name': 'GetWeather', 'arguments': '{"location":"San Francisco"}'}]}], 'cost': 0.000194}}, id='run-5c44c01a-d7bb-4df6-835e-bda596080399-0', tool_calls=[{'name': 'GetWeather', 'args': {'location': 'San Francisco'}, 'id': 'call_tRpAO7KbQwgTjlka70mCQJdo'}])




```python
ai_msg.tool_calls
```




    [{'name': 'GetWeather',
      'args': {'location': 'San Francisco'},
      'id': 'call_tRpAO7KbQwgTjlka70mCQJdo'}]



### with_structured_output()

The BaseChatModel.with_structured_output interface makes it easy to get structured output from chat models. You can use ChatEdenAI.with_structured_output, which uses tool-calling under the hood), to get the model to more reliably return an output in a specific format:



```python
structured_llm = llm.with_structured_output(GetWeather)
structured_llm.invoke(
    "what is the weather like in San Francisco",
)
```




    GetWeather(location='San Francisco')



### Passing Tool Results to model

Here is a full example of how to use a tool. Pass the tool output to the model, and get the result back from the model


```python
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


llm = ChatEdenAI(
    provider="openai",
    max_tokens=1000,
    temperature=0.2,
)

llm_with_tools = llm.bind_tools([add], tool_choice="required")

query = "What is 11 + 11?"

messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

tool_call = ai_msg.tool_calls[0]
tool_output = add.invoke(tool_call["args"])

# This append the result from our tool to the model
messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

llm_with_tools.invoke(messages).content
```




    '11 + 11 = 22'



### Streaming

Eden AI does not currently support streaming tool calls. Attempting to stream will yield a single final message.


```python
list(llm_with_tools.stream("What's 9 + 9"))
```

    /home/eden/Projects/edenai-langchain/libs/community/langchain_community/chat_models/edenai.py:603: UserWarning: stream: Tool use is not yet supported in streaming mode.
      warnings.warn("stream: Tool use is not yet supported in streaming mode.")
    




    [AIMessageChunk(content='', id='run-fae32908-ec48-4ab2-ad96-bb0d0511754f', tool_calls=[{'name': 'add', 'args': {'a': 9, 'b': 9}, 'id': 'call_n0Tm7I9zERWa6UpxCAVCweLN'}], tool_call_chunks=[{'name': 'add', 'args': '{"a": 9, "b": 9}', 'id': 'call_n0Tm7I9zERWa6UpxCAVCweLN', 'index': 0}])]


