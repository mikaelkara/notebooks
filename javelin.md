# Javelin AI Gateway Tutorial

This Jupyter Notebook will explore how to interact with the Javelin AI Gateway using the Python SDK. 
The Javelin AI Gateway facilitates the utilization of large language models (LLMs) like OpenAI, Cohere, Anthropic, and others by 
providing a secure and unified endpoint. The gateway itself provides a centralized mechanism to roll out models systematically, 
provide access security, policy & cost guardrails for enterprises, etc., 

For a complete listing of all the features & benefits of Javelin, please visit www.getjavelin.io



## Step 1: Introduction
[The Javelin AI Gateway](https://www.getjavelin.io) is an enterprise-grade API Gateway for AI applications. It integrates robust access security, ensuring secure interactions with large language models. Learn more in the [official documentation](https://docs.getjavelin.io).


## Step 2: Installation
Before we begin, we must install the `javelin_sdk` and set up the Javelin API key as an environment variable. 


```python
pip install 'javelin_sdk'
```

    Requirement already satisfied: javelin_sdk in /usr/local/Caskroom/miniconda/base/lib/python3.11/site-packages (0.1.8)
    Requirement already satisfied: httpx<0.25.0,>=0.24.0 in /usr/local/Caskroom/miniconda/base/lib/python3.11/site-packages (from javelin_sdk) (0.24.1)
    Requirement already satisfied: pydantic<2.0.0,>=1.10.7 in /usr/local/Caskroom/miniconda/base/lib/python3.11/site-packages (from javelin_sdk) (1.10.12)
    Requirement already satisfied: certifi in /usr/local/Caskroom/miniconda/base/lib/python3.11/site-packages (from httpx<0.25.0,>=0.24.0->javelin_sdk) (2023.5.7)
    Requirement already satisfied: httpcore<0.18.0,>=0.15.0 in /usr/local/Caskroom/miniconda/base/lib/python3.11/site-packages (from httpx<0.25.0,>=0.24.0->javelin_sdk) (0.17.3)
    Requirement already satisfied: idna in /usr/local/Caskroom/miniconda/base/lib/python3.11/site-packages (from httpx<0.25.0,>=0.24.0->javelin_sdk) (3.4)
    Requirement already satisfied: sniffio in /usr/local/Caskroom/miniconda/base/lib/python3.11/site-packages (from httpx<0.25.0,>=0.24.0->javelin_sdk) (1.3.0)
    Requirement already satisfied: typing-extensions>=4.2.0 in /usr/local/Caskroom/miniconda/base/lib/python3.11/site-packages (from pydantic<2.0.0,>=1.10.7->javelin_sdk) (4.7.1)
    Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/Caskroom/miniconda/base/lib/python3.11/site-packages (from httpcore<0.18.0,>=0.15.0->httpx<0.25.0,>=0.24.0->javelin_sdk) (0.14.0)
    Requirement already satisfied: anyio<5.0,>=3.0 in /usr/local/Caskroom/miniconda/base/lib/python3.11/site-packages (from httpcore<0.18.0,>=0.15.0->httpx<0.25.0,>=0.24.0->javelin_sdk) (3.7.1)
    Note: you may need to restart the kernel to use updated packages.
    

## Step 3: Completions Example
This section will demonstrate how to interact with the Javelin AI Gateway to get completions from a large language model. Here is a Python script that demonstrates this:
(note) assumes that you have setup a route in the gateway called 'eng_dept03'


```python
from langchain.chains import LLMChain
from langchain_community.llms import JavelinAIGateway
from langchain_core.prompts import PromptTemplate

route_completions = "eng_dept03"

gateway = JavelinAIGateway(
    gateway_uri="http://localhost:8000",  # replace with service URL or host/port of Javelin
    route=route_completions,
    model_name="gpt-3.5-turbo-instruct",
)

prompt = PromptTemplate("Translate the following English text to French: {text}")

llmchain = LLMChain(llm=gateway, prompt=prompt)
result = llmchain.run("podcast player")

print(result)
```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    Cell In[6], line 2
          1 from langchain.chains import LLMChain
    ----> 2 from langchain.llms import JavelinAIGateway
          3 from langchain.prompts import PromptTemplate
          5 route_completions = "eng_dept03"
    

    ImportError: cannot import name 'JavelinAIGateway' from 'langchain.llms' (/usr/local/Caskroom/miniconda/base/lib/python3.11/site-packages/langchain/llms/__init__.py)


# Step 4: Embeddings Example
This section demonstrates how to use the Javelin AI Gateway to obtain embeddings for text queries and documents. Here is a Python script that illustrates this:
(note) assumes that you have setup a route in the gateway called 'embeddings'


```python
from langchain_community.embeddings import JavelinAIGatewayEmbeddings

embeddings = JavelinAIGatewayEmbeddings(
    gateway_uri="http://localhost:8000",  # replace with service URL or host/port of Javelin
    route="embeddings",
)

print(embeddings.embed_query("hello"))
print(embeddings.embed_documents(["hello"]))
```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    Cell In[9], line 1
    ----> 1 from langchain.embeddings import JavelinAIGatewayEmbeddings
          2 from langchain.embeddings.openai import OpenAIEmbeddings
          4 embeddings = JavelinAIGatewayEmbeddings(
          5     gateway_uri="http://localhost:8000", # replace with service URL or host/port of Javelin
          6     route="embeddings",
          7 )
    

    ImportError: cannot import name 'JavelinAIGatewayEmbeddings' from 'langchain.embeddings' (/usr/local/Caskroom/miniconda/base/lib/python3.11/site-packages/langchain/embeddings/__init__.py)


# Step 5: Chat Example
This section illustrates how to interact with the Javelin AI Gateway to facilitate a chat with a large language model. Here is a Python script that demonstrates this:
(note) assumes that you have setup a route in the gateway called 'mychatbot_route'


```python
from langchain_community.chat_models import ChatJavelinAIGateway
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Artificial Intelligence has the power to transform humanity and make the world a better place"
    ),
]

chat = ChatJavelinAIGateway(
    gateway_uri="http://localhost:8000",  # replace with service URL or host/port of Javelin
    route="mychatbot_route",
    model_name="gpt-3.5-turbo",
    params={"temperature": 0.1},
)

print(chat(messages))
```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    Cell In[8], line 1
    ----> 1 from langchain.chat_models import ChatJavelinAIGateway
          2 from langchain.schema import HumanMessage, SystemMessage
          4 messages = [
          5     SystemMessage(
          6         content="You are a helpful assistant that translates English to French."
       (...)
         10     ),
         11 ]
    

    ImportError: cannot import name 'ChatJavelinAIGateway' from 'langchain.chat_models' (/usr/local/Caskroom/miniconda/base/lib/python3.11/site-packages/langchain/chat_models/__init__.py)


Step 6: Conclusion
This tutorial introduced the Javelin AI Gateway and demonstrated how to interact with it using the Python SDK. 
Remember to check the Javelin [Python SDK](https://www.github.com/getjavelin.io/javelin-python) for more examples and to explore the official documentation for additional details.

Happy coding!
