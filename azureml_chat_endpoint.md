---
sidebar_label: Azure ML Endpoint
---
# AzureMLChatOnlineEndpoint

>[Azure Machine Learning](https://azure.microsoft.com/en-us/products/machine-learning/) is a platform used to build, train, and deploy machine learning models. Users can explore the types of models to deploy in the Model Catalog, which provides foundational and general purpose models from different providers.
>
>In general, you need to deploy models in order to consume its predictions (inference). In `Azure Machine Learning`, [Online Endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints) are used to deploy these models with a real-time serving. They are based on the ideas of `Endpoints` and `Deployments` which allow you to decouple the interface of your production workload from the implementation that serves it.

This notebook goes over how to use a chat model hosted on an `Azure Machine Learning Endpoint`.


```python
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint
```

## Set up

You must [deploy a model on Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-foundation-models?view=azureml-api-2#deploying-foundation-models-to-endpoints-for-inferencing) or [to Azure AI studio](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-open) and obtain the following parameters:

* `endpoint_url`: The REST endpoint url provided by the endpoint.
* `endpoint_api_type`: Use `endpoint_type='dedicated'` when deploying models to **Dedicated endpoints** (hosted managed infrastructure). Use `endpoint_type='serverless'` when deploying models using the **Pay-as-you-go** offering (model as a service).
* `endpoint_api_key`: The API key provided by the endpoint

## Content Formatter

The `content_formatter` parameter is a handler class for transforming the request and response of an AzureML endpoint to match with required schema. Since there are a wide range of models in the model catalog, each of which may process data differently from one another, a `ContentFormatterBase` class is provided to allow users to transform data to their liking. The following content formatters are provided:

* `CustomOpenAIChatContentFormatter`: Formats request and response data for models like LLaMa2-chat that follow the OpenAI API spec for request and response.

*Note: `langchain.chat_models.azureml_endpoint.LlamaChatContentFormatter` is being deprecated and replaced with `langchain.chat_models.azureml_endpoint.CustomOpenAIChatContentFormatter`.*

You can implement custom content formatters specific for your model deriving from the class `langchain_community.llms.azureml_endpoint.ContentFormatterBase`.

## Examples

The following section contains examples about how to use this class:

### Example: Chat completions with real-time endpoints


```python
from langchain_community.chat_models.azureml_endpoint import (
    AzureMLEndpointApiType,
    CustomOpenAIChatContentFormatter,
)
from langchain_core.messages import HumanMessage

chat = AzureMLChatOnlineEndpoint(
    endpoint_url="https://<your-endpoint>.<your_region>.inference.ml.azure.com/score",
    endpoint_api_type=AzureMLEndpointApiType.dedicated,
    endpoint_api_key="my-api-key",
    content_formatter=CustomOpenAIChatContentFormatter(),
)
response = chat.invoke(
    [HumanMessage(content="Will the Collatz conjecture ever be solved?")]
)
response
```




    AIMessage(content='  The Collatz Conjecture is one of the most famous unsolved problems in mathematics, and it has been the subject of much study and research for many years. While it is impossible to predict with certainty whether the conjecture will ever be solved, there are several reasons why it is considered a challenging and important problem:\n\n1. Simple yet elusive: The Collatz Conjecture is a deceptively simple statement that has proven to be extraordinarily difficult to prove or disprove. Despite its simplicity, the conjecture has eluded some of the brightest minds in mathematics, and it remains one of the most famous open problems in the field.\n2. Wide-ranging implications: The Collatz Conjecture has far-reaching implications for many areas of mathematics, including number theory, algebra, and analysis. A solution to the conjecture could have significant impacts on these fields and potentially lead to new insights and discoveries.\n3. Computational evidence: While the conjecture remains unproven, extensive computational evidence supports its validity. In fact, no counterexample to the conjecture has been found for any starting value up to 2^64 (a number', additional_kwargs={}, example=False)



### Example: Chat completions with pay-as-you-go deployments (model as a service)


```python
chat = AzureMLChatOnlineEndpoint(
    endpoint_url="https://<your-endpoint>.<your_region>.inference.ml.azure.com/v1/chat/completions",
    endpoint_api_type=AzureMLEndpointApiType.serverless,
    endpoint_api_key="my-api-key",
    content_formatter=CustomOpenAIChatContentFormatter,
)
response = chat.invoke(
    [HumanMessage(content="Will the Collatz conjecture ever be solved?")]
)
response
```

If you need to pass additional parameters to the model, use `model_kwargs` argument:


```python
chat = AzureMLChatOnlineEndpoint(
    endpoint_url="https://<your-endpoint>.<your_region>.inference.ml.azure.com/v1/chat/completions",
    endpoint_api_type=AzureMLEndpointApiType.serverless,
    endpoint_api_key="my-api-key",
    content_formatter=CustomOpenAIChatContentFormatter,
    model_kwargs={"temperature": 0.8},
)
```

Parameters can also be passed during invocation:


```python
response = chat.invoke(
    [HumanMessage(content="Will the Collatz conjecture ever be solved?")],
    max_tokens=512,
)
response
```
