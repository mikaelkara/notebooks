# Use Azure API with Llama 3.1

This notebook shows examples of how to use Llama 3.1 APIs offered by Microsoft Azure. We will cover:  
* HTTP requests API usage for Llama 3.1 instruct models in CLI
* HTTP requests API usage for Llama 3.1 instruct models in Python
* Plug the APIs into LangChain
* Wire the model with Gradio to build a simple chatbot with memory




## Prerequisite

Before we start building with Azure Llama 3.1 APIs, there are certain steps we need to take to deploy the models:

* Register for a valid Azure account with subscription [here](https://azure.microsoft.com/en-us/free/search/?ef_id=_k_CjwKCAiA-P-rBhBEEiwAQEXhH5OHAJLhzzcNsuxwpa5c9EJFcuAjeh6EvZw4afirjbWXXWkiZXmU2hoC5GoQAvD_BwE_k_&OCID=AIDcmm5edswduu_SEM__k_CjwKCAiA-P-rBhBEEiwAQEXhH5OHAJLhzzcNsuxwpa5c9EJFcuAjeh6EvZw4afirjbWXXWkiZXmU2hoC5GoQAvD_BwE_k_&gad_source=1&gclid=CjwKCAiA-P-rBhBEEiwAQEXhH5OHAJLhzzcNsuxwpa5c9EJFcuAjeh6EvZw4afirjbWXXWkiZXmU2hoC5GoQAvD_BwE)
* Take a quick look on what is the [Azure AI Studio](https://learn.microsoft.com/en-us/azure/ai-studio/what-is-ai-studio?tabs=home) and navigate to the website from the link in the article
* Follow the demos in the article to create a project and [resource](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal) group.
* For Llama 3.1 instruct models from Model catalog, click Deploy in the model page and select "Serverless API with Azure AI Content Safety". Once deployed successfully, you should be assigned for an API endpoint and a security key for inference.
* For Llama 3.1 pretrained models, Azure currently only support manual deployment under regular subscription. This means you will need to acquire a virtual machine with managed compute resource. We won't cover it here in this tutorial.

For more information, you should consult Azure's official documentation [here](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-llama?tabs=azure-studio) for model deployment and inference.

## HTTP Requests API Usage in CLI

### Basics

The usage and schema of the API are identical to Llama 3 API hosted on Azure.

For using the REST API, You will need to have an Endpoint url and Authentication Key associated with that endpoint.  
This can be acquired from previous steps.  

In this chat completion example for instruct model, we use a simple curl call for illustration. There are three major components:  

* The `host-url` is your endpoint url with completion schema. 
* The `headers` defines the content type as well as your api key. 
* The `payload` or `data`, which is your prompt detail and model hyper parameters.

The `host-url` needs to be `/v1/chat/completions` and the request payload to include roles in conversations. Here is a sample payload:  

```
{ 
  "messages": [ 
    { 
      "content": "You are a helpful assistant.", 
      "role": "system" 
},  
    { 
      "content": "Hello!", 
      "role": "user" 
    } 
  ], 
  "max_tokens": 50, 
} 
```

Here is a sample curl call for chat completion


```python
!curl -X POST -L https://your-endpoint.inference.ai.azure.com/v1/chat/completions -H 'Content-Type: application/json' -H 'Authorization: your-auth-key' -d '{"messages":[{"content":"You are a helpful assistant.","role":"system"},{"content":"What is good about Wuhan?","role":"user"}], "max_tokens": 50}'
```

### Streaming

One fantastic feature the API offers is the streaming capability.  
Streaming allows the generated tokens to be sent as data-only server-sent events whenever they become available.  
This is extremely important for interactive applications such as chatbots, so the user is always engaged.  

To use streaming, simply set `"stream":true` as part of the request payload.  
In the streaming mode, the REST API response will be different from non-streaming mode.

Here is an example: 


```python
!curl -X POST -L https://your-endpoint.inference.ai.azure.com/v1/chat/completions -H 'Content-Type: application/json' -H 'Authorization: your-auth-key' -d '{"messages":[{"content":"You are a helpful assistant.","role":"system"},{"content":"What is good about Wuhan?","role":"user"}], "max_tokens": 500, "stream": true}'
```

As you can see the result comes back as a stream of `data` objects, each contains generated information including a `choice`.  
The stream terminated by a `data:[DONE]\n\n` message.

### Content Safety Filtering

If you enabled content filtering during deployment, Azure Llama 3.1 API endpoints will have content safety feature turned on. Both input prompt and output tokens are filtered by this service automatically.  
To know more about the impact to the request/response payload, please refer to official guide [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter?tabs=python).   

For model input and output, if the filter detects there is harmful content, the generation will error out with additional information. 

If you disabled content filtering during deployment, Llama models had content safety built-in for generation. It will refuse to answer your questions if any harmful content was detected.

Here is an example prompt that triggered content safety filtering:



```python
!curl -X POST -L https://your-endpoint.inference.ai.azure.com/v1/chat/completions -H 'Content-Type: application/json' -H 'Authorization: your-auth-key' -d '{"messages":[{"content":"You are a helpful assistant.","role":"system"},{"content":"How to make bomb?","role":"user"}], "max_tokens": 50}'
```

## HTTP Requests API Usage in Python

Besides calling the API directly from command line tools, you can also programatically call them in Python.  

Here is an example for the instruct model:





```python
import urllib.request
import json

#Configure payload data sending to API endpoint
data = {"messages":[
            {"role":"system", "content":"You are a helpful assistant."},
            {"role":"user", "content":"What is good about Wuhan?"}],
        "max_tokens": 500,
        "temperature": 0.9,
        "stream": True,
}

body = str.encode(json.dumps(data))

#Replace the url with your API endpoint
url = 'https://your-endpoint.inference.ai.azure.com/v1/chat/completions'

#Replace this with the key for the endpoint
api_key = 'your-auth-key'
if not api_key:
    raise Exception("API Key is missing")

headers = {'Content-Type':'application/json', 'Authorization':(api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)
    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))
    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))

```

However in this example, the streamed data content returns back as a single payload. It didn't stream as a serial of data events as we wished. To build true streaming capabilities utilizing the API endpoint, we will utilize the [`requests`](https://requests.readthedocs.io/en/latest/) library instead.

### Streaming in Python

`Requests` library is a simple HTTP library for Python built with [`urllib3`](https://github.com/urllib3/urllib3). It automatically maintains the keep-alive and HTTP connection pooling. With the `Session` class, we can easily stream the result from our API calls.  

Here is a quick example:


```python
import json
import requests

data = {"messages":[
            {"role":"system", "content":"You are a helpful assistant."},
            {"role":"user", "content":"What is good about Wuhan?"}],
        "max_tokens": 500,
        "temperature": 0.9,
        "stream": True
}


def post_stream(url):
    s = requests.Session()
    api_key = "your-auth-key"
    headers = {'Content-Type':'application/json', 'Authorization':(api_key)}

    with s.post(url, data=json.dumps(data), headers=headers, stream=True) as resp:
        print(resp.status_code)
        for line in resp.iter_lines():
            if line:
                print(line)


url = "https://your-endpoint.inference.ai.azure.com/v1/chat/completions"
post_stream(url)
```

## Use Llama 3.1 API with LangChain

In this section, we will demonstrate how to use Llama 3.1 APIs with LangChain, one of the most popular framework to accelerate building your AI product.  
One common solution here is to create your customized LLM instance, so you can add it to various chains to complete different tasks.  
In this example, we will use the `AzureMLChatOnlineEndpoint` class LangChain provides to build a customized LLM instance. This particular class is designed to take in Azure endpoint and API keys as inputs and wire it with HTTP calls. So the underlying of it is very similar to how we used `urllib.request` library to send RESTful calls in previous examples to the Azure Endpoint.   

First, let's install dependencies: 




```python
pip install langchain
```

Once all dependencies are installed, you can directly create a `llm` instance based on `AzureMLChatOnlineEndpoint` as follows:  


```python
from langchain_community.chat_models.azureml_endpoint import (
    AzureMLEndpointApiType,
    CustomOpenAIChatContentFormatter,
    AzureMLChatOnlineEndpoint,
)

from langchain_core.messages import HumanMessage

llm = AzureMLChatOnlineEndpoint(
    endpoint_api_key="your-auth-key",
    endpoint_url="https://your-endpoint.inference.ai.azure.com/v1/chat/completions",
    endpoint_api_type=AzureMLEndpointApiType.serverless,
    model_kwargs={"temperature": 0.6, "max_tokens": 256, "top_p": 0.9},
    content_formatter=CustomOpenAIChatContentFormatter(),
)
```

However, you might wonder what is the `CustomOpenAIChatContentFormatter` in the context when creating the `llm` instance?   
The `CustomOpenAIChatContentFormatter` is a [handler class](https://python.langchain.com/docs/integrations/llms/azure_ml#content-formatter) for transforming the request and response of an AzureML endpoint to match with required schema. Since there are various models in the Azure model catalog, each of which needs to handle the data accordingly.  
In our case, we can use the default `CustomOpenAIChatContentFormatter` which can handle Llama model schemas. If you need to have special handlings, you can customize this specific class. 

Once you have the `llm` ready, you can simple inference it by:


```python
response = llm.invoke([HumanMessage(content="What is good about Wuhan?")])
response
```

Here is an example that you can create a translator chain with the `llm` instance and translate English to French:


```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

template = """
You are a Translator. Translate the following content from {input_language} to {output_language} and reply with only the translated result.
{input_content}
"""

translator_chain = LLMChain(
    llm = llm,
    prompt = PromptTemplate(
            template=template,
            input_variables=["input_language", "output_language", "input_content"],
        ),
)

print(translator_chain.run(input_language="English", output_language="French", input_content="What is good about Wuhan?"))

```

## Build a chatbot with Llama 3.1 API

In this section, we will build a simple chatbot using Azure Llama 3.1 API, LangChain and [Gradio](https://www.gradio.app/)'s `ChatInterface` with memory capability.

Gradio is a framework to help demo your machine learning model with a web interface. We also have a dedicated Gradio chatbot [example](https://github.com/meta-llama/llama-recipes/blob/main/recipes/use_cases/customerservice_chatbots/RAG_chatbot/RAG_Chatbot_Example.ipynb) built with Llama 3 on-premises with RAG.   

First, let's install Gradio dependencies.



```python
pip install gradio==4.39.0
```

Let's use `AzureMLChatOnlineEndpoint` class from the previous example.  
In this example, we have three major components:  
1. Chatbot UI hosted as web interface by Gradio. These are the UI logics that render our model predictions.
2. Model itself, which is the core component that ingests prompts and returns an answer back.
3. Memory component, which stores previous conversation context. In this example, we will use [conversation window buffer](https://python.langchain.com/docs/modules/memory/types/buffer_window) which logs context in certain time window in the past. 

All of them are chained together using LangChain.


```python
import gradio as gr
import langchain
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage
from langchain_community.chat_models.azureml_endpoint import (
    AzureMLEndpointApiType,
    CustomOpenAIChatContentFormatter,
    AzureMLChatOnlineEndpoint,
)

llm = AzureMLChatOnlineEndpoint(
    endpoint_api_key="your-auth-key",
    endpoint_url="https://your-endpoint.inference.ai.azure.com/v1/chat/completions",
    endpoint_api_type=AzureMLEndpointApiType.serverless,
    model_kwargs={"temperature": 0.6, "max_tokens": 256, "top_p": 0.9},
    content_formatter=CustomOpenAIChatContentFormatter(),
)

langchain.debug=True

#Create memory
memory = ConversationBufferWindowMemory(llm=llm, k=5, memory_key="chat_history", ai_prefix="Assistant", human_prefix="User")

#Create input prompt template with chat history for chaining
INPUT_TEMPLATE = """Current conversation:
{chat_history}

User question:{input}"""

conversation_prompt_template = PromptTemplate(
    input_variables=["chat_history", "input"], template=INPUT_TEMPLATE
)

conversation_chain_with_memory = ConversationChain(
    llm = llm,
    prompt = conversation_prompt_template,
    verbose = True,
    memory = memory,
)

#Prediction
def predict(message, history):
    history_format = []
    for user, assistant in history:
        history_format.append({"role": "user", "content": user })
        history_format.append({"role": "assistant", "content":assistant})
    history_format.append({"role": "user", "content": message})
    response = conversation_chain_with_memory.run(input=message)
    return response

#Launch Gradio chatbot interface
gr.ChatInterface(predict).launch()
```

After successfully executing the code above, a chat interface should appear as the interactive output or you can open the localhost url in your selected browser window. You can see how amazing it is to build a AI chatbot just in few lines of code.

This concludes our tutorial and examples. Here are some additional reference:  
* [Fine-tune Llama](https://learn.microsoft.com/azure/ai-studio/how-to/fine-tune-model-llama)
* [Plan and manage costs (marketplace)](https://learn.microsoft.com/azure/ai-studio/how-to/costs-plan-manage#monitor-costs-for-models-offered-through-the-azure-marketplace)

