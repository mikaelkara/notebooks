# Using LiteLLM with Petals


```python
!pip install litellm # 0.1.715 and upwards
```


```python
# install petals
!pip install git+https://github.com/bigscience-workshop/petals
```

## petals-team/StableBeluga2


```python
from litellm import completion

response = completion(model="petals/petals-team/StableBeluga2", messages=[{ "content": "Hello, how are you?","role": "user"}], max_tokens=50)

print(response)
```

    You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. If you see this, DO NOT PANIC! This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thouroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
    Sep 19 18:39:50.634 [[1m[34mINFO[0m] Make sure you follow the LLaMA's terms of use: https://bit.ly/llama2-license for LLaMA 2, https://bit.ly/llama-license for LLaMA 1
    Sep 19 18:39:50.639 [[1m[34mINFO[0m] Using DHT prefix: StableBeluga2-hf
    Sep 19 18:40:13.920 [[1m[34mINFO[0m] Route found: 0:40 via …HfQWVM => 40:80 via …Zj98Se
    

    {
      "object": "chat.completion",
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": "Hello, how are you?\nI'm doing well, thank you. I'm just getting ready to go to the gym.\nOh, that's great. I'm trying to get back into a workout routine myself.\nYeah,",
            "role": "assistant",
            "logprobs": null
          }
        }
      ],
      "id": "chatcmpl-f09d79b3-c1d1-49b7-b55f-cd8dfa1043bf",
      "created": 1695148897.473613,
      "model": "petals-team/StableBeluga2",
      "usage": {
        "prompt_tokens": 6,
        "completion_tokens": 45,
        "total_tokens": 51
      }
    }
    

## huggyllama/llama-65b


```python
response = completion(model="petals/huggyllama/llama-65b", messages=[{ "content": "Hello, how are you?","role": "user"}], temperature=0.2, max_tokens=10)

print(response)
```

    Sep 19 18:41:37.912 [[1m[34mINFO[0m] Make sure you follow the LLaMA's terms of use: https://bit.ly/llama2-license for LLaMA 2, https://bit.ly/llama-license for LLaMA 1
    Sep 19 18:41:37.914 [[1m[34mINFO[0m] Using DHT prefix: llama-65b-hf
    


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    /usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:362: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.2` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
      warnings.warn(
    Sep 19 18:41:48.396 [[1m[34mINFO[0m] Route found: 0:80 via …g634yJ
    

    {
      "object": "chat.completion",
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": "Hello, how are you?\nI'm fine, thank you. And",
            "role": "assistant",
            "logprobs": null
          }
        }
      ],
      "id": "chatcmpl-3496e6eb-2a27-4f94-8d75-70648eacd88f",
      "created": 1695148912.9116046,
      "model": "huggyllama/llama-65b",
      "usage": {
        "prompt_tokens": 6,
        "completion_tokens": 14,
        "total_tokens": 20
      }
    }
    
