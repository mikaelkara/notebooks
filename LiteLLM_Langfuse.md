## Use LiteLLM with Langfuse
https://docs.litellm.ai/docs/observability/langfuse_integration

## Install Dependencies


```python
!pip install litellm langfuse
```

## Set Env Variables


```python
import litellm
from litellm import completion
import os

# from https://cloud.langfuse.com/
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""


# OpenAI and Cohere keys
# You can use any of the litellm supported providers: https://docs.litellm.ai/docs/providers
os.environ['OPENAI_API_KEY']=""
os.environ['COHERE_API_KEY']=""

```

## Set LangFuse as a callback for sending data
## OpenAI completion call


```python
# set langfuse as a callback, litellm will send the data to langfuse
litellm.success_callback = ["langfuse"]

# openai call
response = completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ]
)

print(response)
```

    {
      "id": "chatcmpl-85nP4xHdAP3jAcGneIguWATS9qdoO",
      "object": "chat.completion",
      "created": 1696392238,
      "model": "gpt-3.5-turbo-0613",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Hello! How can I assist you today?"
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 9,
        "total_tokens": 24
      }
    }
    


```python
# we set langfuse as a callback in the prev cell
# cohere call
response = completion(
  model="command-nightly",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm cohere"}
  ]
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
            "content": " Nice to meet you, Cohere! I'm excited to be meeting new members of the AI community",
            "role": "assistant",
            "logprobs": null
          }
        }
      ],
      "id": "chatcmpl-a14e903f-4608-4ceb-b996-8ebdf21360ca",
      "created": 1696392247.3313863,
      "model": "command-nightly",
      "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 20,
        "total_tokens": 28
      }
    }
    
