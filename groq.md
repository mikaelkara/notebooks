---
sidebar_label: Groq
---
# ChatGroq

This will help you getting started with Groq [chat models](../../concepts/chat_models.mdx). For detailed documentation of all ChatGroq features and configurations head to the [API reference](https://python.langchain.com/api_reference/groq/chat_models/langchain_groq.chat_models.ChatGroq.html). For a list of all Groq models, visit this [link](https://console.groq.com/docs/models).

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/groq) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatGroq](https://python.langchain.com/api_reference/groq/chat_models/langchain_groq.chat_models.ChatGroq.html) | [langchain-groq](https://python.langchain.com/api_reference/groq/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-groq?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-groq?style=flat-square&label=%20) |

### Model features
| [Tool calling](../../how_to/tool_calling.ipynb) | [Structured output](../../how_to/structured_output.ipynb) | JSON mode | [Image input](../../how_to/multimodal_inputs.ipynb) | Audio input | Video input | [Token-level streaming](../../how_to/chat_streaming.ipynb) | Native async | [Token usage](../../how_to/chat_token_usage_tracking.ipynb) | [Logprobs](../../how_to/logprobs.ipynb) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | 

## Setup

To access Groq models you'll need to create a Groq account, get an API key, and install the `langchain-groq` integration package.

### Credentials

Head to the [Groq console](https://console.groq.com/keys) to sign up to Groq and generate an API key. Once you've done this set the GROQ_API_KEY environment variable:


```python
import getpass
import os

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
```

If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:


```python
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"
```

### Installation

The LangChain Groq integration lives in the `langchain-groq` package:


```python
%pip install -qU langchain-groq
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.0[0m[39;49m -> [0m[32;49m24.1.2[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.
    

## Instantiation

Now we can instantiate our model object and generate chat completions:


```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
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




    AIMessage(content='I enjoy programming. (The French translation is: "J\'aime programmer.")\n\nNote: I chose to translate "I love programming" as "J\'aime programmer" instead of "Je suis amoureux de programmer" because the latter has a romantic connotation that is not present in the original English sentence.', response_metadata={'token_usage': {'completion_tokens': 73, 'prompt_tokens': 31, 'total_tokens': 104, 'completion_time': 0.1140625, 'prompt_time': 0.003352463, 'queue_time': None, 'total_time': 0.117414963}, 'model_name': 'mixtral-8x7b-32768', 'system_fingerprint': 'fp_c5f20b5bb1', 'finish_reason': 'stop', 'logprobs': None}, id='run-64433c19-eadf-42fc-801e-3071e3c40160-0', usage_metadata={'input_tokens': 31, 'output_tokens': 73, 'total_tokens': 104})




```python
print(ai_msg.content)
```

    I enjoy programming. (The French translation is: "J'aime programmer.")
    
    Note: I chose to translate "I love programming" as "J'aime programmer" instead of "Je suis amoureux de programmer" because the latter has a romantic connotation that is not present in the original English sentence.
    

## Chaining

We can [chain](../../how_to/sequence.ipynb) our model with a prompt template like so:


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




    AIMessage(content='That\'s great! I can help you translate English phrases related to programming into German.\n\n"I love programming" can be translated as "Ich liebe Programmieren" in German.\n\nHere are some more programming-related phrases translated into German:\n\n* "Programming language" = "Programmiersprache"\n* "Code" = "Code"\n* "Variable" = "Variable"\n* "Function" = "Funktion"\n* "Array" = "Array"\n* "Object-oriented programming" = "Objektorientierte Programmierung"\n* "Algorithm" = "Algorithmus"\n* "Data structure" = "Datenstruktur"\n* "Debugging" = "Fehlersuche"\n* "Compile" = "Kompilieren"\n* "Link" = "Verknüpfen"\n* "Run" = "Ausführen"\n* "Test" = "Testen"\n* "Deploy" = "Bereitstellen"\n* "Version control" = "Versionskontrolle"\n* "Open source" = "Open Source"\n* "Software development" = "Softwareentwicklung"\n* "Agile methodology" = "Agile Methodik"\n* "DevOps" = "DevOps"\n* "Cloud computing" = "Cloud Computing"\n\nI hope this helps! Let me know if you have any other questions or if you need further translations.', response_metadata={'token_usage': {'completion_tokens': 331, 'prompt_tokens': 25, 'total_tokens': 356, 'completion_time': 0.520006542, 'prompt_time': 0.00250165, 'queue_time': None, 'total_time': 0.522508192}, 'model_name': 'mixtral-8x7b-32768', 'system_fingerprint': 'fp_c5f20b5bb1', 'finish_reason': 'stop', 'logprobs': None}, id='run-74207fb7-85d3-417d-b2b9-621116b75d41-0', usage_metadata={'input_tokens': 25, 'output_tokens': 331, 'total_tokens': 356})



## API reference

For detailed documentation of all ChatGroq features and configurations head to the API reference: https://python.langchain.com/api_reference/groq/chat_models/langchain_groq.chat_models.ChatGroq.html
