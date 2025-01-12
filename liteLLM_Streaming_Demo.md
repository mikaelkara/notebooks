# [STREAMING]  OpenAI, Anthropic, Replicate, Cohere using liteLLM
In this tutorial:
Note: All inputs/outputs are in the format used by `gpt-3.5-turbo`

- Call all models in the same input format [**with streaming**]:

  `completion(model, messages, stream=True)`
- All streaming generators are accessed at `chunk['choices'][0]['delta']`

The following Models are covered in this tutorial
- [GPT-3.5-Turbo](https://platform.openai.com/docs/models/gpt-3-5)
- [Claude-2](https://www.anthropic.com/index/claude-2)
- [StableLM Tuned Alpha 7B](https://replicate.com/stability-ai/stablelm-tuned-alpha-7b)
- [A16z infra-LLAMA-2 7B Chat](https://replicate.com/a16z-infra/llama-2-7b-chat)
- [Vicuna 13B](https://replicate.com/replicate/vicuna-13b)
- [Cohere - Command Nightly]()






```python
# install liteLLM
!pip install litellm==0.1.369
```

## Imports & Set ENV variables
Get your API Keys

https://platform.openai.com/account/api-keys

https://replicate.com/account/api-tokens

https://console.anthropic.com/account/keys

https://dashboard.cohere.ai/api-keys



```python
from litellm import completion
import os

os.environ['OPENAI_API_KEY'] = '' # @param
os.environ['REPLICATE_API_TOKEN'] = '' # @param
os.environ['ANTHROPIC_API_KEY'] = '' # @param
os.environ['COHERE_API_KEY'] = '' # @param
```

### Set Messages


```python
user_message = "Hello, whats the weather in San Francisco??"
messages = [{ "content": user_message,"role": "user"}]
```

## Calling Models using liteLLM Streaming -

## `completion(model, messages, stream)`


```python
# replicate models #######
stability_ai = "stability-ai/stablelm-tuned-alpha-7b:c49dae362cbaecd2ceabb5bd34fdb68413c4ff775111fea065d259d577757beb"
llama_2_7b = "a16z-infra/llama-2-7b-chat:4f0b260b6a13eb53a6b1891f089d57c08f41003ae79458be5011303d81a394dc"
vicuna = "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b"

models = ["gpt-3.5-turbo", "claude-2", stability_ai, llama_2_7b, vicuna, "command-nightly"] # command-nightly is Cohere
for model in models:
  replicate = (model == stability_ai or model==llama_2_7b or model==vicuna) # let liteLLM know if a model is replicate, using this optional param, `replicate=True`
  response = completion(model=model, messages=messages, stream=True, replicate=replicate)
  print(f"####################\n\nResponse from {model}")
  for i, chunk in enumerate(response):
    if i < 5: # NOTE: LIMITING CHUNKS FOR THIS DEMO
      print((chunk['choices'][0]['delta']))

```

    ####################
    
    Response from gpt-3.5-turbo
    {
      "role": "assistant",
      "content": ""
    }
    {
      "content": "I"
    }
    {
      "content": "'m"
    }
    {
      "content": " sorry"
    }
    {
      "content": ","
    }
    ####################
    
    Response from claude-2
    {'role': 'assistant', 'content': ' Unfortunately'}
    {'role': 'assistant', 'content': ' I'}
    {'role': 'assistant', 'content': ' don'}
    {'role': 'assistant', 'content': "'t"}
    {'role': 'assistant', 'content': ' have'}
    ####################
    
    Response from stability-ai/stablelm-tuned-alpha-7b:c49dae362cbaecd2ceabb5bd34fdb68413c4ff775111fea065d259d577757beb
    {'role': 'assistant', 'content': "I'm"}
    {'role': 'assistant', 'content': ' sorry,'}
    {'role': 'assistant', 'content': ' I'}
    {'role': 'assistant', 'content': ' cannot'}
    {'role': 'assistant', 'content': ' answer'}
    ####################
    
    Response from a16z-infra/llama-2-7b-chat:4f0b260b6a13eb53a6b1891f089d57c08f41003ae79458be5011303d81a394dc
    {'role': 'assistant', 'content': ''}
    {'role': 'assistant', 'content': ' Hello'}
    {'role': 'assistant', 'content': '!'}
    {'role': 'assistant', 'content': ' I'}
    {'role': 'assistant', 'content': "'"}
    ####################
    
    Response from replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b
    {'role': 'assistant', 'content': 'Comment:'}
    {'role': 'assistant', 'content': 'Hi! '}
    {'role': 'assistant', 'content': 'How '}
    {'role': 'assistant', 'content': 'are '}
    {'role': 'assistant', 'content': 'you '}
    ####################
    
    Response from command-nightly
    {'role': 'assistant', 'content': ' Hello'}
    {'role': 'assistant', 'content': '!'}
    {'role': 'assistant', 'content': ' '}
    {'role': 'assistant', 'content': ' I'}
    {'role': 'assistant', 'content': "'m"}
    


```python

```
