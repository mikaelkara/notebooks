# OpenAI Adapter

**Please ensure OpenAI library is version 1.0.0 or higher; otherwise, refer to the older doc [OpenAI Adapter(Old)](/docs/integrations/adapters/openai-old/).**

A lot of people get started with OpenAI but want to explore other models. LangChain's integrations with many model providers make this easy to do so. While LangChain has it's own message and model APIs, we've also made it as easy as possible to explore other models by exposing an adapter to adapt LangChain models to the OpenAI api.

At the moment this only deals with output and does not return other information (token counts, stop reasons, etc).


```python
import openai
from langchain_community.adapters import openai as lc_openai
```

## chat.completions.create


```python
messages = [{"role": "user", "content": "hi"}]
```

Original OpenAI call


```python
result = openai.chat.completions.create(
    messages=messages, model="gpt-3.5-turbo", temperature=0
)
result.choices[0].message.model_dump()
```




    {'content': 'Hello! How can I assist you today?',
     'role': 'assistant',
     'function_call': None,
     'tool_calls': None}



LangChain OpenAI wrapper call


```python
lc_result = lc_openai.chat.completions.create(
    messages=messages, model="gpt-3.5-turbo", temperature=0
)

lc_result.choices[0].message  # Attribute access
```




    {'role': 'assistant', 'content': 'Hello! How can I help you today?'}




```python
lc_result["choices"][0]["message"]  # Also compatible with index access
```




    {'role': 'assistant', 'content': 'Hello! How can I help you today?'}



Swapping out model providers


```python
lc_result = lc_openai.chat.completions.create(
    messages=messages, model="claude-2", temperature=0, provider="ChatAnthropic"
)
lc_result.choices[0].message
```




    {'role': 'assistant', 'content': 'Hello! How can I assist you today?'}



## chat.completions.stream

Original OpenAI call


```python
for c in openai.chat.completions.create(
    messages=messages, model="gpt-3.5-turbo", temperature=0, stream=True
):
    print(c.choices[0].delta.model_dump())
```

    {'content': '', 'function_call': None, 'role': 'assistant', 'tool_calls': None}
    {'content': 'Hello', 'function_call': None, 'role': None, 'tool_calls': None}
    {'content': '!', 'function_call': None, 'role': None, 'tool_calls': None}
    {'content': ' How', 'function_call': None, 'role': None, 'tool_calls': None}
    {'content': ' can', 'function_call': None, 'role': None, 'tool_calls': None}
    {'content': ' I', 'function_call': None, 'role': None, 'tool_calls': None}
    {'content': ' assist', 'function_call': None, 'role': None, 'tool_calls': None}
    {'content': ' you', 'function_call': None, 'role': None, 'tool_calls': None}
    {'content': ' today', 'function_call': None, 'role': None, 'tool_calls': None}
    {'content': '?', 'function_call': None, 'role': None, 'tool_calls': None}
    {'content': None, 'function_call': None, 'role': None, 'tool_calls': None}
    

LangChain OpenAI wrapper call


```python
for c in lc_openai.chat.completions.create(
    messages=messages, model="gpt-3.5-turbo", temperature=0, stream=True
):
    print(c.choices[0].delta)
```

    {'role': 'assistant', 'content': ''}
    {'content': 'Hello'}
    {'content': '!'}
    {'content': ' How'}
    {'content': ' can'}
    {'content': ' I'}
    {'content': ' assist'}
    {'content': ' you'}
    {'content': ' today'}
    {'content': '?'}
    {}
    

Swapping out model providers


```python
for c in lc_openai.chat.completions.create(
    messages=messages,
    model="claude-2",
    temperature=0,
    stream=True,
    provider="ChatAnthropic",
):
    print(c["choices"][0]["delta"])
```

    {'role': 'assistant', 'content': ''}
    {'content': 'Hello'}
    {'content': '!'}
    {'content': ' How'}
    {'content': ' can'}
    {'content': ' I'}
    {'content': ' assist'}
    {'content': ' you'}
    {'content': ' today'}
    {'content': '?'}
    {}
    
