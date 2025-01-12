## 🚅 liteLLM Quick Start Demo
### TLDR: Call 50+ LLM APIs using chatGPT Input/Output format
https://github.com/BerriAI/litellm

liteLLM is package to simplify calling **OpenAI, Azure, Llama2, Cohere, Anthropic, Huggingface API Endpoints**. LiteLLM manages



## Installation and setting Params


```python
!pip install litellm
```


```python
from litellm import completion
import os
```

## Set your API keys
- liteLLM reads your .env, env variables or key manager for Auth

Set keys for the models you want to use below


```python
# Only set keys for the LLMs you want to use
os.environ['OPENAI_API_KEY'] = "" #@param
os.environ["ANTHROPIC_API_KEY"] = "" #@param
os.environ["REPLICATE_API_KEY"] = "" #@param
os.environ["COHERE_API_KEY"] = "" #@param
os.environ["AZURE_API_BASE"] = "" #@param
os.environ["AZURE_API_VERSION"] = "" #@param
os.environ["AZURE_API_KEY"] = "" #@param
```

## Call chatGPT


```python
completion(model="gpt-3.5-turbo", messages=[{ "content": "what's the weather in SF","role": "user"}])
```




    <OpenAIObject chat.completion id=chatcmpl-820kPkRwSLml4X6165fWbZlEDOedr at 0x12ff93630> JSON: {
      "id": "chatcmpl-820kPkRwSLml4X6165fWbZlEDOedr",
      "object": "chat.completion",
      "created": 1695490221,
      "model": "gpt-3.5-turbo-0613",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "I'm sorry, but as an AI text-based model, I don't have real-time information. However, you can check the current weather in San Francisco by searching for \"weather in SF\" on any search engine or checking a weather website or app."
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 13,
        "completion_tokens": 51,
        "total_tokens": 64
      },
      "response_ms": 2385.592
    }



## Call Claude-2


```python
completion(model="claude-2", messages=[{ "content": "what's the weather in SF","role": "user"}])
```




    <ModelResponse chat.completion id=chatcmpl-6d1a40c0-19c0-4bd7-9ca2-a91d8b8c2295 at 0x12ff85a40> JSON: {
      "object": "chat.completion",
      "choices": [
        {
          "finish_reason": "stop_sequence",
          "index": 0,
          "message": {
            "content": " Unfortunately I don't have enough context to know the exact location you are asking about when you say \"SF\". SF could refer to San Francisco, California, or potentially other cities that go by SF as an abbreviation. To get an accurate weather report, it would be helpful if you could provide the full city name and state/country. If you are looking for the weather in San Francisco, California, I would be happy to provide that forecast. Please let me know the specific location you want the weather for.",
            "role": "assistant",
            "logprobs": null
          }
        }
      ],
      "id": "chatcmpl-6d1a40c0-19c0-4bd7-9ca2-a91d8b8c2295",
      "created": 1695490260.983768,
      "response_ms": 6351.544,
      "model": "claude-2",
      "usage": {
        "prompt_tokens": 14,
        "completion_tokens": 102,
        "total_tokens": 116
      }
    }



## Call llama2 on replicate


```python
model = "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1"
completion(model=model, messages=[{ "content": "what's the weather in SF","role": "user"}])
```




    <ModelResponse chat.completion id=chatcmpl-3151c2eb-b26f-4c96-89b5-ed1746b219e0 at 0x138b87e50> JSON: {
      "object": "chat.completion",
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": " I'm happy to help! However, I must point out that the question \"what's the weather in SF\" doesn't make sense as \"SF\" could refer to multiple locations. Could you please clarify which location you are referring to? San Francisco, California or Sioux Falls, South Dakota? Once I have more context, I would be happy to provide you with accurate and reliable information.",
            "role": "assistant",
            "logprobs": null
          }
        }
      ],
      "id": "chatcmpl-3151c2eb-b26f-4c96-89b5-ed1746b219e0",
      "created": 1695490237.714101,
      "response_ms": 12109.565,
      "model": "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
      "usage": {
        "prompt_tokens": 6,
        "completion_tokens": 78,
        "total_tokens": 84
      },
      "ended": 1695490249.821266
    }



## Call Command-Nightly


```python
completion(model="command-nightly", messages=[{ "content": "what's the weather in SF","role": "user"}])
```




    <ModelResponse chat.completion id=chatcmpl-dc0d8ead-071d-486c-a111-78975b38794b at 0x1389725e0> JSON: {
      "object": "chat.completion",
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": " As an AI model I don't have access to real-time data, so I can't tell",
            "role": "assistant",
            "logprobs": null
          }
        }
      ],
      "id": "chatcmpl-dc0d8ead-071d-486c-a111-78975b38794b",
      "created": 1695490235.936903,
      "response_ms": 1022.6759999999999,
      "model": "command-nightly",
      "usage": {
        "prompt_tokens": 6,
        "completion_tokens": 19,
        "total_tokens": 25
      }
    }



## Call Azure OpenAI

For azure openai calls ensure to add the `azure/` prefix to `model`. If your deployment-id is `chatgpt-test` set `model` = `azure/chatgpt-test`


```python
completion(model="azure/chatgpt-v-2", messages=[{ "content": "what's the weather in SF","role": "user"}])
```




    <OpenAIObject chat.completion id=chatcmpl-820kZyCwbNvZATiLkNmXmpxxzvTKO at 0x138b84ae0> JSON: {
      "id": "chatcmpl-820kZyCwbNvZATiLkNmXmpxxzvTKO",
      "object": "chat.completion",
      "created": 1695490231,
      "model": "gpt-35-turbo",
      "choices": [
        {
          "index": 0,
          "finish_reason": "stop",
          "message": {
            "role": "assistant",
            "content": "Sorry, as an AI language model, I don't have real-time information. Please check your preferred weather website or app for the latest weather updates of San Francisco."
          }
        }
      ],
      "usage": {
        "completion_tokens": 33,
        "prompt_tokens": 14,
        "total_tokens": 47
      },
      "response_ms": 1499.529
    }




```python

```
