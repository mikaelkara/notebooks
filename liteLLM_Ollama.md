```python
!pip install litellm # version 0.1.724 or higher 
```

## Call Ollama - llama2 with Streaming


```python
from litellm import completion

response = completion(
    model="ollama/llama2", 
    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
    api_base="http://localhost:11434",
    stream=True
)
print(response)
for chunk in response:
    print(chunk['choices'][0]['delta'])

```

    <generator object get_ollama_response_stream at 0x109096c10>
    {'role': 'assistant', 'content': ' I'}
    {'role': 'assistant', 'content': "'"}
    {'role': 'assistant', 'content': 'm'}
    {'role': 'assistant', 'content': ' L'}
    {'role': 'assistant', 'content': 'La'}
    {'role': 'assistant', 'content': 'MA'}
    {'role': 'assistant', 'content': ','}
    {'role': 'assistant', 'content': ' an'}
    {'role': 'assistant', 'content': ' A'}
    {'role': 'assistant', 'content': 'I'}
    {'role': 'assistant', 'content': ' assistant'}
    {'role': 'assistant', 'content': ' developed'}
    {'role': 'assistant', 'content': ' by'}
    {'role': 'assistant', 'content': ' Meta'}
    {'role': 'assistant', 'content': ' A'}
    {'role': 'assistant', 'content': 'I'}
    {'role': 'assistant', 'content': ' that'}
    {'role': 'assistant', 'content': ' can'}
    {'role': 'assistant', 'content': ' understand'}
    {'role': 'assistant', 'content': ' and'}
    {'role': 'assistant', 'content': ' respond'}
    {'role': 'assistant', 'content': ' to'}
    {'role': 'assistant', 'content': ' human'}
    {'role': 'assistant', 'content': ' input'}
    {'role': 'assistant', 'content': ' in'}
    {'role': 'assistant', 'content': ' a'}
    {'role': 'assistant', 'content': ' convers'}
    {'role': 'assistant', 'content': 'ational'}
    {'role': 'assistant', 'content': ' manner'}
    {'role': 'assistant', 'content': '.'}
    

## Call Ollama - Llama2 with Acompletion + Streaming


```python
# litellm uses async_generator for ollama async streaming, ensure it's installed
!pip install async_generator
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: async_generator in /Users/ishaanjaffer/Library/Python/3.9/lib/python/site-packages (1.10)
    


```python
import litellm

async def async_ollama():
    response = await litellm.acompletion(
        model="ollama/llama2", 
        messages=[{ "content": "what's the weather" ,"role": "user"}], 
        api_base="http://localhost:11434", 
        stream=True
    )
    async for chunk in response:
        print(chunk)

result = await async_ollama()
print(result)

try:
    async for chunk in result:
        print(chunk)
except TypeError: # the last chunk is None from Ollama, this raises an error with async streaming
    pass
```

    {'choices': [{'delta': {'role': 'assistant', 'content': ' I'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': "'"}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': 'm'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' just'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' an'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' A'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': 'I'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ','}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' I'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' don'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': "'"}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': 't'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' have'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' access'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' to'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' real'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': '-'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': 'time'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' weather'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' information'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' or'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' current'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' conditions'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' in'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' your'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' specific'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' location'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': '.'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' живело'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' can'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' provide'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' you'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' with'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' weather'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' forec'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': 'asts'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' and'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' information'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' for'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' your'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' location'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' if'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' you'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' would'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' like'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': '.'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' Please'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' let'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' me'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' know'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' where'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' you'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' are'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' located'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ','}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' and'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' I'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' will'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' do'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' my'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' best'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' to'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' assist'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': ' you'}}]}
    {'choices': [{'delta': {'role': 'assistant', 'content': '.'}}]}
    None
    

## Completion Call


```python
from litellm import completion

response = completion(
    model="ollama/llama2", 
    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
    api_base="http://localhost:11434"
)
print(response)

```

    {
      "object": "chat.completion",
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": " I'm LLaMA, an AI assistant developed by Meta AI that can understand and respond to human input in a conversational manner.",
            "role": "assistant",
            "logprobs": null
          }
        }
      ],
      "id": "chatcmpl-ea7b8242-791f-4656-ba12-e098edeb960e",
      "created": 1695324686.6696231,
      "response_ms": 4072.3050000000003,
      "model": "ollama/llama2",
      "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 27,
        "total_tokens": 37
      }
    }
    


```python

```
