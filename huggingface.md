# ChatHuggingFace

This will help you getting started with `langchain_huggingface` [chat models](/docs/concepts/chat_models). For detailed documentation of all `ChatHuggingFace` features and configurations head to the [API reference](https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html). For a list of models supported by Hugging Face check out [this page](https://huggingface.co/models).

## Overview
### Integration details

### Integration details

| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatHuggingFace](https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html) | [langchain-huggingface](https://python.langchain.com/api_reference/huggingface/index.html) | ✅ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_huggingface?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_huggingface?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | 

## Setup

To access Hugging Face models you'll need to create a Hugging Face account, get an API key, and install the `langchain-huggingface` integration package.

### Credentials

Generate a [Hugging Face Access Token](https://huggingface.co/docs/hub/security-tokens) and store it as an environment variable: `HUGGINGFACEHUB_API_TOKEN`.


```python
import getpass
import os

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")
```

### Installation

| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatHuggingFace](https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html) | [langchain_huggingface](https://python.langchain.com/api_reference/huggingface/index.html) | ✅ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_huggingface?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_huggingface?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 

## Setup

To access `langchain_huggingface` models you'll need to create a/an `Hugging Face` account, get an API key, and install the `langchain_huggingface` integration package.

### Credentials

You'll need to have a [Hugging Face Access Token](https://huggingface.co/docs/hub/security-tokens) saved as an environment variable: `HUGGINGFACEHUB_API_TOKEN`.


```python
import getpass
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass(
    "Enter your Hugging Face API key: "
)
```


```python
%pip install --upgrade --quiet  langchain-huggingface text-generation transformers google-search-results numexpr langchainhub sentencepiece jinja2 bitsandbytes accelerate
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.0[0m[39;49m -> [0m[32;49m24.1.2[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.
    

## Instantiation

You can instantiate a `ChatHuggingFace` model in two different ways, either from a `HuggingFaceEndpoint` or from a `HuggingFacePipeline`.

### `HuggingFaceEndpoint`


```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

chat_model = ChatHuggingFace(llm=llm)
```

    The token has not been saved to the git credentials helper. Pass `add_to_git_credential=True` in this function directly or `--add-to-git-credential` if using via `huggingface-cli` if you want to set the git credential as well.
    Token is valid (permission: fineGrained).
    Your token has been saved to /Users/isaachershenson/.cache/huggingface/token
    Login successful
    

### `HuggingFacePipeline`


```python
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)

chat_model = ChatHuggingFace(llm=llm)
```


    config.json:   0%|          | 0.00/638 [00:00<?, ?B/s]



    model.safetensors.index.json:   0%|          | 0.00/23.9k [00:00<?, ?B/s]



    Downloading shards:   0%|          | 0/8 [00:00<?, ?it/s]



    model-00001-of-00008.safetensors:   0%|          | 0.00/1.89G [00:00<?, ?B/s]



    model-00002-of-00008.safetensors:   0%|          | 0.00/1.95G [00:00<?, ?B/s]



    model-00003-of-00008.safetensors:   0%|          | 0.00/1.98G [00:00<?, ?B/s]



    model-00004-of-00008.safetensors:   0%|          | 0.00/1.95G [00:00<?, ?B/s]



    model-00005-of-00008.safetensors:   0%|          | 0.00/1.98G [00:00<?, ?B/s]



    model-00006-of-00008.safetensors:   0%|          | 0.00/1.95G [00:00<?, ?B/s]



    model-00007-of-00008.safetensors:   0%|          | 0.00/1.98G [00:00<?, ?B/s]



    model-00008-of-00008.safetensors:   0%|          | 0.00/816M [00:00<?, ?B/s]



    Loading checkpoint shards:   0%|          | 0/8 [00:00<?, ?it/s]



    generation_config.json:   0%|          | 0.00/111 [00:00<?, ?B/s]


### Instatiating with Quantization

To run a quantized version of your model, you can specify a `bitsandbytes` quantization config as follows:


```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)
```

and pass it to the `HuggingFacePipeline` as a part of its `model_kwargs`:


```python
llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
    model_kwargs={"quantization_config": quantization_config},
)

chat_model = ChatHuggingFace(llm=llm)
```

## Invocation


```python
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]

ai_msg = chat_model.invoke(messages)
```


```python
print(ai_msg.content)
```

    According to the popular phrase and hypothetical scenario, when an unstoppable force meets an immovable object, a paradoxical situation arises as both forces are seemingly contradictory. On one hand, an unstoppable force is an entity that cannot be stopped or prevented from moving forward, while on the other hand, an immovable object is something that cannot be moved or displaced from its position. 
    
    In this scenario, it is un
    

## API reference

For detailed documentation of all `ChatHuggingFace` features and configurations head to the API reference: https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html

## API reference

For detailed documentation of all ChatHuggingFace features and configurations head to the API reference: https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html
