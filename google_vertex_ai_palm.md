---
sidebar_label: Google Cloud Vertex AI
---
# ChatVertexAI

This page provides a quick overview for getting started with VertexAI [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatVertexAI features and configurations head to the [API reference](https://python.langchain.com/api_reference/google_vertexai/chat_models/langchain_google_vertexai.chat_models.ChatVertexAI.html).

ChatVertexAI exposes all foundational models available in Google Cloud, like `gemini-1.5-pro`, `gemini-1.5-flash`, etc. For a full and updated list of available models visit [VertexAI documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/overview).

:::info Google Cloud VertexAI vs Google PaLM

The Google Cloud VertexAI integration is separate from the [Google PaLM integration](/docs/integrations/chat/google_generative_ai/). Google has chosen to offer an enterprise version of PaLM through GCP, and this supports the models made available through there. 

:::

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/google_vertex_ai) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatVertexAI](https://python.langchain.com/api_reference/google_vertexai/chat_models/langchain_google_vertexai.chat_models.ChatVertexAI.html) | [langchain-google-vertexai](https://python.langchain.com/api_reference/google_vertexai/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-google-vertexai?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-google-vertexai?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 

## Setup

To access VertexAI models you'll need to create a Google Cloud Platform account, set up credentials, and install the `langchain-google-vertexai` integration package.

### Credentials

To use the integration you must:
- Have credentials configured for your environment (gcloud, workload identity, etc...)
- Store the path to a service account JSON file as the GOOGLE_APPLICATION_CREDENTIALS environment variable

This codebase uses the `google.auth` library which first looks for the application credentials variable mentioned above, and then looks for system-level auth.

For more information, see: 
- https://cloud.google.com/docs/authentication/application-default-credentials#GAC
- https://googleapis.dev/python/google-auth/latest/reference/google.auth.html#module-google.auth

If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:


```python
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"
```

### Installation

The LangChain VertexAI integration lives in the `langchain-google-vertexai` package:


```python
%pip install -qU langchain-google-vertexai
```

    Note: you may need to restart the kernel to use updated packages.
    

## Instantiation

Now we can instantiate our model object and generate chat completions:


```python
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(
    model="gemini-1.5-flash-001",
    temperature=0,
    max_tokens=None,
    max_retries=6,
    stop=None,
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




    AIMessage(content="J'adore programmer. \n", response_metadata={'is_blocked': False, 'safety_ratings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}], 'usage_metadata': {'prompt_token_count': 20, 'candidates_token_count': 7, 'total_token_count': 27}}, id='run-7032733c-d05c-4f0c-a17a-6c575fdd1ae0-0', usage_metadata={'input_tokens': 20, 'output_tokens': 7, 'total_tokens': 27})




```python
print(ai_msg.content)
```

    J'adore programmer. 
    
    

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




    AIMessage(content='Ich liebe Programmieren. \n', response_metadata={'is_blocked': False, 'safety_ratings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}], 'usage_metadata': {'prompt_token_count': 15, 'candidates_token_count': 8, 'total_token_count': 23}}, id='run-c71955fd-8dc1-422b-88a7-853accf4811b-0', usage_metadata={'input_tokens': 15, 'output_tokens': 8, 'total_tokens': 23})



## API reference

For detailed documentation of all ChatVertexAI features and configurations, like how to send multimodal inputs and configure safety settings, head to the API reference: https://python.langchain.com/api_reference/google_vertexai/chat_models/langchain_google_vertexai.chat_models.ChatVertexAI.html
