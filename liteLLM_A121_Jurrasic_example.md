# LiteLLM A121 Tutorial

This walks through using A121 Jurassic models
* j2-light
* j2-mid
* j2-ultra


```python
!pip install litellm
```


```python
from litellm import completion
import os
```

## Set A121 Keys
You can get a free key from https://studio.ai21.com/account/api-key


```python
os.environ["AI21_API_KEY"] = ""
```

# A121 Supported Models:
https://studio.ai21.com/foundation-models

## J2-light Call


```python
messages = [{ "content": "Hello, how are you?","role": "user"}]
response = completion(model="j2-light", messages=messages)
response
```




    <ModelResponse at 0x7b2c2902e610> JSON: {
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": " However, I have an important question to ask you\nMy name is X, and I was wondering if you would be willing to help me.",
            "role": "assistant"
          }
        }
      ],
      "created": 1692761063.5189915,
      "model": "j2-light",
      "usage": {
        "prompt_tokens": null,
        "completion_tokens": null,
        "total_tokens": null
      }
    }



# J2-Mid


```python
messages = [{ "content": "what model are you","role": "user"}]
response = completion(model="j2-mid", messages=messages)
response
```




    <ModelResponse at 0x7b2c2902f6a0> JSON: {
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": "\nplease choose the model from the list below\nModel view in Tekla Structures",
            "role": "assistant"
          }
        }
      ],
      "created": 1692761140.0017524,
      "model": "j2-mid",
      "usage": {
        "prompt_tokens": null,
        "completion_tokens": null,
        "total_tokens": null
      }
    }



# J2-Ultra


```python
messages = [{ "content": "what model are you","role": "user"}]
response = completion(model="j2-ultra", messages=messages)
response
```




    <ModelResponse at 0x7b2c28fd4090> JSON: {
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": "\nI am not a specific model, but I can provide information and assistance based on my training data. Please let me know if there is anything you",
            "role": "assistant"
          }
        }
      ],
      "created": 1692761157.8675153,
      "model": "j2-ultra",
      "usage": {
        "prompt_tokens": null,
        "completion_tokens": null,
        "total_tokens": null
      }
    }


