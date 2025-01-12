# LiteLLM Bedrock Usage
Important Note: For Bedrock Requests you need to ensure you have `pip install boto3>=1.28.57`, boto3 supports bedrock from `boto3>=1.28.57` and higher 

## Pre-Requisites


```python
!pip install litellm
!pip install boto3>=1.28.57 # this version onwards has bedrock support
```

## Set Bedrock/AWS Credentials


```python
import os
os.environ["AWS_ACCESS_KEY_ID"] = "" # Access key
os.environ["AWS_SECRET_ACCESS_KEY"] = "" # Secret access key
os.environ["AWS_REGION_NAME"] = ""
```

## Anthropic Requests


```python
from litellm import completion

response = completion(
            model="bedrock/anthropic.claude-instant-v1",
            messages=[{ "content": "Hello, how are you?","role": "user"}]
)
print("Claude instant 1, response")
print(response)


response = completion(
            model="bedrock/anthropic.claude-v2",
            messages=[{ "content": "Hello, how are you?","role": "user"}]
)
print("Claude v2, response")
print(response)
```

    Claude instant 1, response
    {
      "object": "chat.completion",
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": " I'm doing well, thanks for asking!",
            "role": "assistant",
            "logprobs": null
          }
        }
      ],
      "id": "chatcmpl-4f2e64a1-56d2-43f2-90d3-60ffd6f5086d",
      "created": 1696256761.3265705,
      "model": "anthropic.claude-instant-v1",
      "usage": {
        "prompt_tokens": 11,
        "completion_tokens": 9,
        "total_tokens": 20
      },
      "finish_reason": "stop_sequence"
    }
    Claude v2, response
    {
      "object": "chat.completion",
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": " I'm doing well, thanks for asking!",
            "role": "assistant",
            "logprobs": null
          }
        }
      ],
      "id": "chatcmpl-34f59b33-f94e-40c2-8bdb-f4af0813405e",
      "created": 1696256762.2137017,
      "model": "anthropic.claude-v2",
      "usage": {
        "prompt_tokens": 11,
        "completion_tokens": 9,
        "total_tokens": 20
      },
      "finish_reason": "stop_sequence"
    }
    

## Anthropic Requests - With Streaming


```python
from litellm import completion

response = completion(
            model="bedrock/anthropic.claude-instant-v1",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            stream=True,
)
print("Claude instant 1, response")
for chunk in response:
  print(chunk)


response = completion(
            model="bedrock/anthropic.claude-v2",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            stream=True
)
print("Claude v2, response")
print(response)
for chunk in response:
  print(chunk)
```

## A121 Requests


```python
response = completion(
            model="bedrock/ai21.j2-ultra",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
)
print("J2 ultra response")
print(response)

response = completion(
            model="bedrock/ai21.j2-mid",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
)
print("J2 mid response")
print(response)
```

    J2 ultra response
    {
      "object": "chat.completion",
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": "\nHi, I'm doing well, thanks for asking! How about you?",
            "role": "assistant",
            "logprobs": null
          }
        }
      ],
      "id": "chatcmpl-f2de678f-0e70-4e36-a01f-8b184c2e4d50",
      "created": 1696257116.044311,
      "model": "ai21.j2-ultra",
      "usage": {
        "prompt_tokens": 6,
        "completion_tokens": 16,
        "total_tokens": 22
      }
    }
    J2 mid response
    {
      "object": "chat.completion",
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": "\nGood. And you?",
            "role": "assistant",
            "logprobs": null
          }
        }
      ],
      "id": "chatcmpl-420d6bf9-36d8-484b-93b4-4c9e00f7ce2e",
      "created": 1696257116.5756805,
      "model": "ai21.j2-mid",
      "usage": {
        "prompt_tokens": 6,
        "completion_tokens": 6,
        "total_tokens": 12
      }
    }
    


```python

```
