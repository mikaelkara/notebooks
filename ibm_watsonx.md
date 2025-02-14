---
sidebar_label: IBM watsonx.ai
---
# ChatWatsonx

>ChatWatsonx is a wrapper for IBM [watsonx.ai](https://www.ibm.com/products/watsonx-ai) foundation models.

The aim of these examples is to show how to communicate with `watsonx.ai` models using `LangChain` LLMs API.

## Overview

### Integration details
| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/ibm/) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatWatsonx](https://python.langchain.com/api_reference/ibm/chat_models/langchain_ibm.chat_models.ChatWatsonx.html#langchain_ibm.chat_models.ChatWatsonx) | [langchain-ibm](https://python.langchain.com/api_reference/ibm/index.html) | ❌ | ❌ | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-ibm?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-ibm?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling/) | [Structured output](/docs/how_to/structured_output/) | JSON mode | Image input | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | 

## Setup

To access IBM watsonx.ai models you'll need to create an IBM watsonx.ai account, get an API key, and install the `langchain-ibm` integration package.

### Credentials

The cell below defines the credentials required to work with watsonx Foundation Model inferencing.

**Action:** Provide the IBM Cloud user API key. For details, see
[Managing user API keys](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui).


```python
import os
from getpass import getpass

watsonx_api_key = getpass()
os.environ["WATSONX_APIKEY"] = watsonx_api_key
```

Additionally you are able to pass additional secrets as an environment variable. 


```python
import os

os.environ["WATSONX_URL"] = "your service instance url"
os.environ["WATSONX_TOKEN"] = "your token for accessing the CPD cluster"
os.environ["WATSONX_PASSWORD"] = "your password for accessing the CPD cluster"
os.environ["WATSONX_USERNAME"] = "your username for accessing the CPD cluster"
os.environ["WATSONX_INSTANCE_ID"] = "your instance_id for accessing the CPD cluster"
```

### Installation

The LangChain IBM integration lives in the `langchain-ibm` package:


```python
!pip install -qU langchain-ibm
```

## Instantiation

You might need to adjust model `parameters` for different models or tasks. For details, refer to [Available TextChatParameters](https://ibm.github.io/watsonx-ai-python-sdk/fm_schema.html#ibm_watsonx_ai.foundation_models.schema.TextChatParameters).


```python
parameters = {
    "temperature": 0.9,
    "max_tokens": 200,
}
```

Initialize the `WatsonxLLM` class with the previously set parameters.


**Note**: 

- To provide context for the API call, you must pass the `project_id` or `space_id`. To get your project or space ID, open your project or space, go to the **Manage** tab, and click **General**. For more information see: [Project documentation](https://www.ibm.com/docs/en/watsonx-as-a-service?topic=projects) or [Deployment space documentation](https://www.ibm.com/docs/en/watsonx/saas?topic=spaces-creating-deployment).
- Depending on the region of your provisioned service instance, use one of the urls listed in [watsonx.ai API Authentication](https://ibm.github.io/watsonx-ai-python-sdk/setup_cloud.html#authentication).

In this example, we’ll use the `project_id` and Dallas URL.


You need to specify the `model_id` that will be used for inferencing. You can find the list of all the available models in [Supported chat models](https://ibm.github.io/watsonx-ai-python-sdk/fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_chat_model_specs).


```python
from langchain_ibm import ChatWatsonx

chat = ChatWatsonx(
    model_id="ibm/granite-34b-code-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)
```

Alternatively, you can use Cloud Pak for Data credentials. For details, see [watsonx.ai software setup](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).  


```python
chat = ChatWatsonx(
    model_id="ibm/granite-34b-code-instruct",
    url="PASTE YOUR URL HERE",
    username="PASTE YOUR USERNAME HERE",
    password="PASTE YOUR PASSWORD HERE",
    instance_id="openshift",
    version="4.8",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)
```

Instead of `model_id`, you can also pass the `deployment_id` of the previously tuned model. The entire model tuning workflow is described in [Working with TuneExperiment and PromptTuner](https://ibm.github.io/watsonx-ai-python-sdk/pt_working_with_class_and_prompt_tuner.html).


```python
chat = ChatWatsonx(
    deployment_id="PASTE YOUR DEPLOYMENT_ID HERE",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)
```

## Invocation

To obtain completions, you can call the model directly using a string prompt.


```python
# Invocation

messages = [
    ("system", "You are a helpful assistant that translates English to French."),
    (
        "human",
        "I love you for listening to Rock.",
    ),
]

chat.invoke(messages)
```




    AIMessage(content="J'adore que tu escois de écouter de la rock ! ", additional_kwargs={}, response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 34, 'total_tokens': 53}, 'model_name': 'ibm/granite-34b-code-instruct', 'system_fingerprint': '', 'finish_reason': 'stop'}, id='chat-ef888fc41f0d4b37903b622250ff7528', usage_metadata={'input_tokens': 34, 'output_tokens': 19, 'total_tokens': 53})




```python
# Invocation multiple chat
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

system_message = SystemMessage(
    content="You are a helpful assistant which telling short-info about provided topic."
)
human_message = HumanMessage(content="horse")

chat.invoke([system_message, human_message])
```




    AIMessage(content='horses are quadrupedal mammals that are members of the family Equidae. They are typically farm animals, competing in horse racing and other forms of equine competition. With over 200 breeds, horses are diverse in their physical appearance and behavior. They are intelligent, social animals that are often used for transportation, food, and entertainment.', additional_kwargs={}, response_metadata={'token_usage': {'completion_tokens': 89, 'prompt_tokens': 29, 'total_tokens': 118}, 'model_name': 'ibm/granite-34b-code-instruct', 'system_fingerprint': '', 'finish_reason': 'stop'}, id='chat-9a6e28abb3d448aaa4f83b677a9fd653', usage_metadata={'input_tokens': 29, 'output_tokens': 89, 'total_tokens': 118})



## Chaining
Create `ChatPromptTemplate` objects which will be responsible for creating a random question.


```python
from langchain_core.prompts import ChatPromptTemplate

system = (
    "You are a helpful assistant that translates {input_language} to {output_language}."
)
human = "{input}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
```

Provide a inputs and run the chain.


```python
chain = prompt | chat
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love Python",
    }
)
```




    AIMessage(content='Ich liebe Python.', additional_kwargs={}, response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 28, 'total_tokens': 35}, 'model_name': 'ibm/granite-34b-code-instruct', 'system_fingerprint': '', 'finish_reason': 'stop'}, id='chat-fef871190b6047a7a3e68c58b3810c33', usage_metadata={'input_tokens': 28, 'output_tokens': 7, 'total_tokens': 35})



## Streaming the Model output 

You can stream the model output.


```python
system_message = SystemMessage(
    content="You are a helpful assistant which telling short-info about provided topic."
)
human_message = HumanMessage(content="moon")

for chunk in chat.stream([system_message, human_message]):
    print(chunk.content, end="")
```

    The Moon is the fifth largest moon in the solar system and the largest relative to its host planet. It is the fifth brightest object in Earth's night sky after the Sun, the stars, the Milky Way, and the Moon itself. It orbits around the Earth at an average distance of 238,855 miles (384,400 kilometers). The Moon's gravity is about one-sixthth of Earth's and thus allows for the formation of tides on Earth. The Moon is thought to have formed around 4.5 billion years ago from debris from a collision between Earth and a Mars-sized body named Theia. The Moon is effectively immutable, with its current characteristics remaining from formation. Aside from Earth, the Moon is the only other natural satellite of Earth. The most widely accepted theory is that it formed from the debris of a collision

## Batch the Model output 

You can batch the model output.


```python
message_1 = [
    SystemMessage(
        content="You are a helpful assistant which telling short-info about provided topic."
    ),
    HumanMessage(content="cat"),
]
message_2 = [
    SystemMessage(
        content="You are a helpful assistant which telling short-info about provided topic."
    ),
    HumanMessage(content="dog"),
]

chat.batch([message_1, message_2])
```




    [AIMessage(content='The cat is a popular domesticated carnivorous mammal that belongs to the family Felidae. Cats arefriendly, intelligent, and independent animals that are well-known for their playful behavior, agility, and ability to hunt prey. cats come in a wide range of breeds, each with their own unique physical and behavioral characteristics. They are kept as pets worldwide due to their affectionate nature and companionship. Cats are important members of the household and are often involved in everything from childcare to entertainment.', additional_kwargs={}, response_metadata={'token_usage': {'completion_tokens': 127, 'prompt_tokens': 28, 'total_tokens': 155}, 'model_name': 'ibm/granite-34b-code-instruct', 'system_fingerprint': '', 'finish_reason': 'stop'}, id='chat-fa452af0a0fa4a668b6a704aecd7d718', usage_metadata={'input_tokens': 28, 'output_tokens': 127, 'total_tokens': 155}),
     AIMessage(content='Dogs are domesticated animals that belong to the Canidae family, also known as wolves. They are one of the most popular pets worldwide, known for their loyalty and affection towards their owners. Dogs come in various breeds, each with unique characteristics, and are trained for different purposes such as hunting, herding, or guarding. They require a lot of exercise and mental stimulation to stay healthy and happy, and they need proper training and socialization to be well-behaved. Dogs are also known for their playful and energetic nature, making them great companions for people of all ages.', additional_kwargs={}, response_metadata={'token_usage': {'completion_tokens': 144, 'prompt_tokens': 28, 'total_tokens': 172}, 'model_name': 'ibm/granite-34b-code-instruct', 'system_fingerprint': '', 'finish_reason': 'stop'}, id='chat-cae7663c50cf4f3499726821cc2f0ec7', usage_metadata={'input_tokens': 28, 'output_tokens': 144, 'total_tokens': 172})]



## Tool calling

### ChatWatsonx.bind_tools()

Please note that `ChatWatsonx.bind_tools` is on beta state, so we recommend using `mistralai/mistral-large` model.


```python
from langchain_ibm import ChatWatsonx

chat = ChatWatsonx(
    model_id="mistralai/mistral-large",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
    params=parameters,
)
```


```python
from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = chat.bind_tools([GetWeather])
```


```python
ai_msg = llm_with_tools.invoke(
    "Which city is hotter today: LA or NY?",
)
ai_msg
```




    AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'chatcmpl-tool-6c06a19bbe824d78a322eb193dbde12d', 'type': 'function', 'function': {'name': 'GetWeather', 'arguments': '{"location": "Los Angeles, CA"}'}}, {'id': 'chatcmpl-tool-493542e46f1141bfbfeb5deae6c9e086', 'type': 'function', 'function': {'name': 'GetWeather', 'arguments': '{"location": "New York, NY"}'}}]}, response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 95, 'total_tokens': 141}, 'model_name': 'mistralai/mistral-large', 'system_fingerprint': '', 'finish_reason': 'tool_calls'}, id='chat-027f2bdb217e4238909cb26d3e8a8fbf', tool_calls=[{'name': 'GetWeather', 'args': {'location': 'Los Angeles, CA'}, 'id': 'chatcmpl-tool-6c06a19bbe824d78a322eb193dbde12d', 'type': 'tool_call'}, {'name': 'GetWeather', 'args': {'location': 'New York, NY'}, 'id': 'chatcmpl-tool-493542e46f1141bfbfeb5deae6c9e086', 'type': 'tool_call'}], usage_metadata={'input_tokens': 95, 'output_tokens': 46, 'total_tokens': 141})



### AIMessage.tool_calls
Notice that the AIMessage has a `tool_calls` attribute. This contains in a standardized ToolCall format that is model-provider agnostic.


```python
ai_msg.tool_calls
```




    [{'name': 'GetWeather',
      'args': {'location': 'Los Angeles, CA'},
      'id': 'chatcmpl-tool-6c06a19bbe824d78a322eb193dbde12d',
      'type': 'tool_call'},
     {'name': 'GetWeather',
      'args': {'location': 'New York, NY'},
      'id': 'chatcmpl-tool-493542e46f1141bfbfeb5deae6c9e086',
      'type': 'tool_call'}]



## API reference

For detailed documentation of all `ChatWatsonx` features and configurations head to the [API reference](https://python.langchain.com/api_reference/ibm/chat_models/langchain_ibm.chat_models.ChatWatsonx.html).
