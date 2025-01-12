# Use LiteLLM to calculate costs for all your completion calls
In this notebook we'll use `litellm.completion_cost` to get completion costs


```python
!pip install litellm==0.1.549 # use 0.1.549  or later
```

## Calculating costs for gpt-3.5 turbo completion()


```python
from litellm import completion, completion_cost
import os
os.environ['OPENAI_API_KEY'] = ""

messages = [{ "content": "Hello, how are you?","role": "user"}]
response = completion(
            model="gpt-3.5-turbo",
            messages=messages,
)

print(response)

cost = completion_cost(completion_response=response)
formatted_string = f"Cost for completion call: ${float(cost):.10f}"
print(formatted_string)

```

    got response
    {
      "id": "chatcmpl-7vyCApIZaCxP36kb9meUMN2DFSJPh",
      "object": "chat.completion",
      "created": 1694050442,
      "model": "gpt-3.5-turbo-0613",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Hello! I'm an AI and I don't have feelings, but I'm here to help you. How can I assist you today?"
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 13,
        "completion_tokens": 28,
        "total_tokens": 41
      }
    }
    Cost for completion call: $0.0000755000
    

## Calculating costs for Together Computer completion()


```python
from litellm import completion, completion_cost
import os
os.environ['TOGETHERAI_API_KEY'] = ""

messages = [{ "content": "Hello, how are you?","role": "user"}]
response = completion(
            model="togethercomputer/llama-2-70b-chat",
            messages=messages,
)

print(response)

cost = completion_cost(completion_response=response)
formatted_string = f"Cost for completion call: ${float(cost):.10f}"
print(formatted_string)

```

    {
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": "Hello! I'm doing well, thanks for asking. I hope you're having a great",
            "role": "assistant",
            "logprobs": null
          }
        }
      ],
      "created": 1694050771.2821715,
      "model": "togethercomputer/llama-2-70b-chat",
      "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 18,
        "total_tokens": 30
      }
    }
    Cost for completion call: $0.0000900000
    

## Calculating costs for Replicate Llama2 completion()


```python
from litellm import completion, completion_cost
import os
os.environ['REPLICATE_API_KEY'] = ""

messages = [{ "content": "Hello, how are you?","role": "user"}]
response = completion(
            model="replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
            messages=messages,
)

print(response)

cost = completion_cost(completion_response=response)
formatted_string = f"Cost for completion call: ${float(cost):.10f}"
print(formatted_string)

```

    {
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": " Hello! I'm doing well, thanks for asking. How about you? Is there anything you need help with today?",
            "role": "assistant",
            "logprobs": null
          }
        }
      ],
      "created": 1694050893.4534576,
      "model": "replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
      "usage": {
        "prompt_tokens": 6,
        "completion_tokens": 24,
        "total_tokens": 30
      },
      "ended": 1694050896.6689413
    }
    total_replicate_run_time 3.2154836654663086
    Cost for completion call: $0.0045016771
    
