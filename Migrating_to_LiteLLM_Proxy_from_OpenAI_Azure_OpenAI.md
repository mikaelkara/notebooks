# Migrating to LiteLLM Proxy from OpenAI/Azure OpenAI

Covers:

*   /chat/completion
*   /embedding


These are **selected examples**. LiteLLM Proxy is **OpenAI-Compatible**, it works with any project that calls OpenAI. Just change the `base_url`, `api_key` and `model`.

For more examples, [go here](https://docs.litellm.ai/docs/proxy/user_keys)

To pass provider-specific args, [go here](https://docs.litellm.ai/docs/completion/provider_specific_params#proxy-usage)

To drop unsupported params (E.g. frequency_penalty for bedrock with librechat), [go here](https://docs.litellm.ai/docs/completion/drop_params#openai-proxy-usage)


## /chat/completion



### OpenAI Python SDK


```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_body={ # pass in any provider-specific param, if not supported by openai, https://docs.litellm.ai/docs/completion/input#provider-specific-params
        "metadata": { # 👈 use for logging additional params (e.g. to langfuse)
            "generation_name": "ishaan-generation-openai-client",
            "generation_id": "openai-client-gen-id22",
            "trace_id": "openai-client-trace-id22",
            "trace_user_id": "openai-client-user-id2"
        }
    }
)

print(response)
```

## Function Calling


```python
from openai import OpenAI
client = OpenAI(
    api_key="sk-1234", # [OPTIONAL] set if you set one on proxy, else set ""
    base_url="http://0.0.0.0:4000",
)

tools = [
  {
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
          },
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
      },
    }
  }
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
completion = client.chat.completions.create(
  model="gpt-4o", # use 'model_name' from config.yaml
  messages=messages,
  tools=tools,
  tool_choice="auto"
)

print(completion)

```

### Azure OpenAI Python SDK


```python
import openai
client = openai.AzureOpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_body={ # pass in any provider-specific param, if not supported by openai, https://docs.litellm.ai/docs/completion/input#provider-specific-params
        "metadata": { # 👈 use for logging additional params (e.g. to langfuse)
            "generation_name": "ishaan-generation-openai-client",
            "generation_id": "openai-client-gen-id22",
            "trace_id": "openai-client-trace-id22",
            "trace_user_id": "openai-client-user-id2"
        }
    }
)

print(response)
```

### Langchain Python


```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
import os

os.environ["OPENAI_API_KEY"] = "anything"

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000",
    model = "gpt-3.5-turbo",
    temperature=0.1,
    extra_body={
        "metadata": {
            "generation_name": "ishaan-generation-langchain-client",
            "generation_id": "langchain-client-gen-id22",
            "trace_id": "langchain-client-trace-id22",
            "trace_user_id": "langchain-client-user-id2"
        }
    }
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)
```

### Curl



```
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
    -H 'Content-Type: application/json' \
    -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ],
    "metadata": {
        "generation_name": "ishaan-test-generation",
        "generation_id": "gen-id22",
        "trace_id": "trace-id22",
        "trace_user_id": "user-id2"
    }
}'
```



### LlamaIndex


```python
import os, dotenv

from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

llm = AzureOpenAI(
    engine="azure-gpt-3.5",               # model_name on litellm proxy
    temperature=0.0,
    azure_endpoint="http://0.0.0.0:4000", # litellm proxy endpoint
    api_key="sk-1234",                    # litellm proxy API Key
    api_version="2023-07-01-preview",
)

embed_model = AzureOpenAIEmbedding(
    deployment_name="azure-embedding-model",
    azure_endpoint="http://0.0.0.0:4000",
    api_key="sk-1234",
    api_version="2023-07-01-preview",
)


documents = SimpleDirectoryReader("llama_index_data").load_data()
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

```

### Langchain JS


```python
import { ChatOpenAI } from "@langchain/openai";


const model = new ChatOpenAI({
  modelName: "gpt-4",
  openAIApiKey: "sk-1234",
  modelKwargs: {"metadata": "hello world"} // 👈 PASS Additional params here
}, {
  basePath: "http://0.0.0.0:4000",
});

const message = await model.invoke("Hi there!");

console.log(message);

```

### OpenAI JS


```python
const { OpenAI } = require('openai');

const openai = new OpenAI({
  apiKey: "sk-1234", // This is the default and can be omitted
  baseURL: "http://0.0.0.0:4000"
});

async function main() {
  const chatCompletion = await openai.chat.completions.create({
    messages: [{ role: 'user', content: 'Say this is a test' }],
    model: 'gpt-3.5-turbo',
  }, {"metadata": {
            "generation_name": "ishaan-generation-openaijs-client",
            "generation_id": "openaijs-client-gen-id22",
            "trace_id": "openaijs-client-trace-id22",
            "trace_user_id": "openaijs-client-user-id2"
        }});
}

main();

```

### Anthropic SDK


```python
import os

from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:4000", # proxy endpoint
    api_key="sk-s4xN1IiLTCytwtZFJaYQrA", # litellm proxy virtual key
)

message = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Hello, Claude",
        }
    ],
    model="claude-3-opus-20240229",
)
print(message.content)
```

## /embeddings

### OpenAI Python SDK


```python
import openai
from openai import OpenAI

# set base_url to your proxy server
# set api_key to send to proxy server
client = OpenAI(api_key="<proxy-api-key>", base_url="http://0.0.0.0:4000")

response = client.embeddings.create(
    input=["hello from litellm"],
    model="text-embedding-ada-002"
)

print(response)

```

### Langchain Embeddings


```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="sagemaker-embeddings", openai_api_base="http://0.0.0.0:4000", openai_api_key="temp-key")


text = "This is a test document."

query_result = embeddings.embed_query(text)

print(f"SAGEMAKER EMBEDDINGS")
print(query_result[:5])

embeddings = OpenAIEmbeddings(model="bedrock-embeddings", openai_api_base="http://0.0.0.0:4000", openai_api_key="temp-key")

text = "This is a test document."

query_result = embeddings.embed_query(text)

print(f"BEDROCK EMBEDDINGS")
print(query_result[:5])

embeddings = OpenAIEmbeddings(model="bedrock-titan-embeddings", openai_api_base="http://0.0.0.0:4000", openai_api_key="temp-key")

text = "This is a test document."

query_result = embeddings.embed_query(text)

print(f"TITAN EMBEDDINGS")
print(query_result[:5])
```

### Curl Request



```curl
curl -X POST 'http://0.0.0.0:4000/embeddings' \
  -H 'Content-Type: application/json' \
  -d ' {
  "model": "text-embedding-ada-002",
  "input": ["write a litellm poem"]
  }'
```


