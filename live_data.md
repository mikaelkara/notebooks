## This demo app shows:
* How to use LlamaIndex, an open source library to help you build custom data augmented LLM applications
* How to ask Llama 3 questions about recent live data via the Tavily live search API

The LangChain package is used to facilitate the call to Llama 3 hosted on OctoAI

**Note** We will be using OctoAI to run the examples here. You will need to first sign into [OctoAI](https://octoai.cloud/) with your Github or Google account, then create a free API token [here](https://octo.ai/docs/getting-started/how-to-create-an-octoai-access-token) that you can use for a while (a month or $10 in OctoAI credits, whichever one runs out first).
After the free trial ends, you will need to enter billing info to continue to use Llama3 hosted on OctoAI.

We start by installing the necessary packages:
- [langchain](https://python.langchain.com/docs/get_started/introduction) which provides RAG capabilities
- [llama-index](https://docs.llamaindex.ai/en/stable/) for data augmentation.


```python
!pip install llama-index 
!pip install llama-index-core
!pip install llama-index-llms-octoai
!pip install llama-index-embeddings-octoai
!pip install octoai-sdk
!pip install tavily-python
!pip install replicate
```

Next we set up the OctoAI token.


```python
from getpass import getpass
import os

OCTOAI_API_TOKEN = getpass()
os.environ["OCTOAI_API_TOKEN"] = OCTOAI_API_TOKEN
```

We then call the Llama 3 model from OctoAI.

We will use the Llama 3 8b instruct model. You can find more on Llama models on the [OctoAI text generation solution page](https://octoai.cloud/text).

At the time of writing this notebook the following Llama models are available on OctoAI:
* meta-llama-3-8b-instruct
* meta-llama-3-70b-instruct
* codellama-7b-instruct
* codellama-13b-instruct
* codellama-34b-instruct
* llama-2-13b-chat
* llama-2-70b-chat
* llamaguard-7b


```python
# use ServiceContext to configure the LLM used and the custom embeddings
from llama_index.core import ServiceContext

# VectorStoreIndex is used to index custom data 
from llama_index.core import VectorStoreIndex

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.octoai import OctoAIEmbedding
from llama_index.llms.octoai import OctoAI

Settings.llm = OctoAI(
    model="meta-llama-3-8b-instruct",
    token=OCTOAI_API_TOKEN,
    temperature=0.0,
    max_tokens=128,
)

Settings.embed_model = OctoAIEmbedding(api_key=OCTOAI_API_TOKEN)
```

Next you will use the [Tavily](https://tavily.com/) search engine to augment the Llama 3's responses. To create a free trial Tavily Search API, sign in with your Google or Github account [here](https://app.tavily.com/sign-in).


```python
from tavily import TavilyClient

TAVILY_API_KEY = getpass()
tavily = TavilyClient(api_key=TAVILY_API_KEY)
```

Do a live web search on "Llama 3 fine-tuning".


```python
response = tavily.search(query="Llama 3 fine-tuning")
context = [{"url": obj["url"], "content": obj["content"]} for obj in response['results']]
```


```python
context
```

Create documents based on the search results, index and save them to a vector store, then create a query engine.


```python
from llama_index.core import Document

documents = [Document(text=ct['content']) for ct in context]
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(streaming=True)
```

You are now ready to ask Llama 3 questions about the live data using the query engine.


```python
response = query_engine.query("give me a summary")
response.print_response_stream()
```


```python
query_engine.query("what's the latest about Llama 3 fine-tuning?").print_response_stream()
```


```python
query_engine.query("tell me more about Llama 3 fine-tuning").print_response_stream()
```
