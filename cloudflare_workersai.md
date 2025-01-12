---
sidebar_label: Cloudflare Workers AI
---
# ChatCloudflareWorkersAI

This will help you getting started with CloudflareWorkersAI [chat models](/docs/concepts/chat_models). For detailed documentation of all available Cloudflare WorkersAI models head to the [API reference](https://developers.cloudflare.com/workers-ai/).


## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/cloudflare_workersai) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| ChatCloudflareWorkersAI | langchain-community| ❌ | ❌ | ✅ | ❌  | ❌ |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 

## Setup

- To access Cloudflare Workers AI models you'll need to create a Cloudflare account, get an account number and API key, and install the `langchain-community` package.


### Credentials


Head to [this document](https://developers.cloudflare.com/workers-ai/get-started/rest-api/) to sign up to Cloudflare Workers AI and generate an API key.

If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:


```python
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
```

### Installation

The LangChain ChatCloudflareWorkersAI integration lives in the `langchain-community` package:


```python
%pip install -qU langchain-community
```

## Instantiation

Now we can instantiate our model object and generate chat completions:


```python
from langchain_community.chat_models.cloudflare_workersai import ChatCloudflareWorkersAI

llm = ChatCloudflareWorkersAI(
    account_id="my_account_id",
    api_token="my_api_token",
    model="@hf/nousresearch/hermes-2-pro-mistral-7b",
)
```

## Invocation


```python
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg
```

    2024-11-07 15:55:14 - INFO - Sending prompt to Cloudflare Workers AI: {'prompt': 'role: system, content: You are a helpful assistant that translates English to French. Translate the user sentence.\nrole: user, content: I love programming.', 'tools': None}
    




    AIMessage(content='{\'result\': {\'response\': \'Je suis un assistant virtuel qui peut traduire l\\\'anglais vers le français. La phrase que vous avez dite est : "J\\\'aime programmer." En français, cela se traduit par : "J\\\'adore programmer."\'}, \'success\': True, \'errors\': [], \'messages\': []}', additional_kwargs={}, response_metadata={}, id='run-838fd398-8594-4ca5-9055-03c72993caf6-0')




```python
print(ai_msg.content)
```

    {'result': {'response': 'Je suis un assistant virtuel qui peut traduire l\'anglais vers le français. La phrase que vous avez dite est : "J\'aime programmer." En français, cela se traduit par : "J\'adore programmer."'}, 'success': True, 'errors': [], 'messages': []}
    

## Chaining

We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:


```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)
```

    2024-11-07 15:55:24 - INFO - Sending prompt to Cloudflare Workers AI: {'prompt': 'role: system, content: You are a helpful assistant that translates English to German.\nrole: user, content: I love programming.', 'tools': None}
    




    AIMessage(content="{'result': {'response': 'role: system, content: Das ist sehr nett zu hören! Programmieren lieben, ist eine interessante und anspruchsvolle Hobby- oder Berufsausrichtung. Wenn Sie englische Texte ins Deutsche übersetzen möchten, kann ich Ihnen helfen. Geben Sie bitte den englischen Satz oder die Übersetzung an, die Sie benötigen.'}, 'success': True, 'errors': [], 'messages': []}", additional_kwargs={}, response_metadata={}, id='run-0d3be9a6-3d74-4dde-b49a-4479d6af00ef-0')



## API reference

For detailed documentation on `ChatCloudflareWorkersAI` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.cloudflare_workersai.html).
