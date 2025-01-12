# Use liteLLM to call Falcon, Wizard, MPT 7B using OpenAI chatGPT Input/output

* Falcon 7B: https://app.baseten.co/explore/falcon_7b
* Wizard LM: https://app.baseten.co/explore/wizardlm
* MPT 7B Base: https://app.baseten.co/explore/mpt_7b_instruct


## Call all baseten llm models using OpenAI chatGPT Input/Output using liteLLM
Example call
```python
model = "q841o8w" # baseten model version ID
response = completion(model=model, messages=messages, custom_llm_provider="baseten")
```


```python
!pip install litellm==0.1.399
!pip install baseten urllib3
```


```python
import os
import litellm
from litellm import completion
```

## Setup


```python
os.environ['BASETEN_API_KEY'] = "" #@param
messages = [{ "content": "what does Baseten do? ","role": "user"}]
```

## Calling Falcon 7B: https://app.baseten.co/explore/falcon_7b
### Pass Your Baseten model `Version ID` as `model`


```python
model = "qvv0xeq"
response = completion(model=model, messages=messages, custom_llm_provider="baseten")
response
```

    [32mINFO[0m API key set.
    INFO:baseten:API key set.
    




    {'choices': [{'finish_reason': 'stop',
       'index': 0,
       'message': {'role': 'assistant',
        'content': "what does Baseten do? \nI'm sorry, I cannot provide a specific answer as"}}],
     'created': 1692135883.699066,
     'model': 'qvv0xeq'}



## Calling Wizard LM https://app.baseten.co/explore/wizardlm
### Pass Your Baseten model `Version ID` as `model`


```python
model = "q841o8w"
response = completion(model=model, messages=messages, custom_llm_provider="baseten")
response
```

    [32mINFO[0m API key set.
    INFO:baseten:API key set.
    




    {'choices': [{'finish_reason': 'stop',
       'index': 0,
       'message': {'role': 'assistant',
        'content': 'As an AI language model, I do not have personal beliefs or practices, but based on the information available online, Baseten is a popular name for a traditional Ethiopian dish made with injera, a spongy flatbread, and wat, a spicy stew made with meat or vegetables. It is typically served for breakfast or dinner and is a staple in Ethiopian cuisine. The name Baseten is also used to refer to a traditional Ethiopian coffee ceremony, where coffee is brewed and served in a special ceremony with music and food.'}}],
     'created': 1692135900.2806294,
     'model': 'q841o8w'}



## Calling mosaicml/mpt-7b https://app.baseten.co/explore/mpt_7b_instruct
### Pass Your Baseten model `Version ID` as `model`


```python
model = "31dxrj3"
response = completion(model=model, messages=messages, custom_llm_provider="baseten")
response
```

    [32mINFO[0m API key set.
    INFO:baseten:API key set.
    




    {'choices': [{'finish_reason': 'stop',
       'index': 0,
       'message': {'role': 'assistant',
        'content': "\n===================\n\nIt's a tool to build a local version of a game on your own machine to host\non your website.\n\nIt's used to make game demos and show them on Twitter, Tumblr, and Facebook.\n\n\n\n## What's built\n\n- A directory of all your game directories, named with a version name and build number, with images linked to.\n- Includes HTML to include in another site.\n- Includes images for your icons and"}}],
     'created': 1692135914.7472186,
     'model': '31dxrj3'}


