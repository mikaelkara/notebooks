# LiteLLM Batch Completions Example

* This tutorial walks through using `batch_completion`
* Docs: https://docs.litellm.ai/docs/completion/batching


```python
!pip install litellm
```

## Import Batch Completion


```python
import litellm
import os
from litellm import batch_completion

# set your API_KEY
os.environ['ANTHROPIC_API_KEY'] = ""
```

## Calling `litellm.batch_completion`

In the batch_completion method, you provide a list of messages where each sub-list of messages is passed to litellm.completion(), allowing you to process multiple prompts efficiently in a single API call.


```python
import litellm
import os
from litellm import batch_completion

os.environ['ANTHROPIC_API_KEY'] = ""


responses = batch_completion(
    model="claude-2",
    messages = [
        [
            {
                "role": "user",
                "content": "good morning? "
            }
        ],
        [
            {
                "role": "user",
                "content": "what's the time? "
            }
        ]
    ]
)
responses
```




    [<ModelResponse at 0x7a164eed4450> JSON: {
       "choices": [
         {
           "finish_reason": "stop",
           "index": 0,
           "message": {
             "content": " Good morning!",
             "role": "assistant",
             "logprobs": null
           }
         }
       ],
       "created": 1694030351.309254,
       "model": "claude-2",
       "usage": {
         "prompt_tokens": 11,
         "completion_tokens": 3,
         "total_tokens": 14
       }
     },
     <ModelResponse at 0x7a164eed5800> JSON: {
       "choices": [
         {
           "finish_reason": "stop",
           "index": 0,
           "message": {
             "content": " I'm an AI assistant created by Anthropic. I don't actually have a concept of the current time.",
             "role": "assistant",
             "logprobs": null
           }
         }
       ],
       "created": 1694030352.1215081,
       "model": "claude-2",
       "usage": {
         "prompt_tokens": 13,
         "completion_tokens": 22,
         "total_tokens": 35
       }
     }]


