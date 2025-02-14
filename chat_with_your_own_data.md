# Azure chat completion models with your own data (preview)

> Note: There is a newer version of the openai library available. See https://github.com/openai/openai-python/discussions/742

This example shows how to use Azure OpenAI service models with your own data. The feature is currently in preview. 

Azure OpenAI on your data enables you to run supported chat models such as GPT-3.5-Turbo and GPT-4 on your data without needing to train or fine-tune models. Running models on your data enables you to chat on top of, and analyze your data with greater accuracy and speed. One of the key benefits of Azure OpenAI on your data is its ability to tailor the content of conversational AI. Because the model has access to, and can reference specific sources to support its responses, answers are not only based on its pretrained knowledge but also on the latest information available in the designated data source. This grounding data also helps the model avoid generating responses based on outdated or incorrect information.

Azure OpenAI on your own data with Azure Cognitive Search provides a customizable, pre-built solution for knowledge retrieval, from which a conversational AI application can be built. To see alternative methods for knowledge retrieval and semantic search, check out the cookbook examples for [vector databases](https://github.com/openai/openai-cookbook/tree/main/examples/vector_databases).

## How it works

[Azure OpenAI on your own data](https://learn.microsoft.com/azure/ai-services/openai/concepts/use-your-data) connects the model with your data, giving it the ability to retrieve and utilize data in a way that enhances the model's output. Together with Azure Cognitive Search, data is retrieved from designated data sources based on the user input and provided conversation history. The data is then augmented and resubmitted as a prompt to the model, giving the model contextual information it can use to generate a response.

See the [Data, privacy, and security for Azure OpenAI Service](https://learn.microsoft.com/legal/cognitive-services/openai/data-privacy?context=%2Fazure%2Fai-services%2Fopenai%2Fcontext%2Fcontext) for more information.

## Prerequisites
To get started, we'll cover a few prequisites. 

To properly access the Azure OpenAI Service, we need to create the proper resources at the [Azure Portal](https://portal.azure.com) (you can check a detailed guide on how to do this in the [Microsoft Docs](https://learn.microsoft.com/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal))

To use your own data with Azure OpenAI models, you will need:

1. Azure OpenAI access and a resource with a chat model deployed (for example, GPT-3 or GPT-4)
2. Azure Cognitive Search resource
3. Azure Blob Storage resource
4. Your documents to be used as data (See [data source options](https://learn.microsoft.com/azure/ai-services/openai/concepts/use-your-data#data-source-options))


For a full walk-through on how to upload your documents to blob storage and create an index using the Azure AI Studio, see this [Quickstart](https://learn.microsoft.com/azure/ai-services/openai/use-your-data-quickstart?pivots=programming-language-studio&tabs=command-line).

## Setup

First, we install the necessary dependencies.


```python
! pip install "openai>=0.28.1,<1.0.0"
! pip install python-dotenv
```

In this example, we'll use `dotenv` to load our environment variables. To connect with Azure OpenAI and the Search index, the following variables should be added to a `.env` file in `KEY=VALUE` format:

* `OPENAI_API_BASE` - the Azure OpenAI endpoint. This can be found under "Keys and Endpoints" for your Azure OpenAI resource in the Azure Portal.
* `OPENAI_API_KEY` - the Azure OpenAI API key. This can be found under "Keys and Endpoints" for your Azure OpenAI resource in the Azure Portal. Omit if using Azure Active Directory authentication (see below `Authentication using Microsoft Active Directory`)
* `SEARCH_ENDPOINT` - the Cognitive Search endpoint. This URL be found on the "Overview" of your Search resource on the Azure Portal.
* `SEARCH_KEY` - the Cognitive Search API key. Found under "Keys" for your Search resource in the Azure Portal.
* `SEARCH_INDEX_NAME` - the name of the index you created with your own data.


```python
import os
import openai
import dotenv

dotenv.load_dotenv()
```


```python
openai.api_base = os.environ["OPENAI_API_BASE"]

# Azure OpenAI on your own data is only supported by the 2023-08-01-preview API version
openai.api_version = "2023-08-01-preview"
```

### Authentication

The Azure OpenAI service supports multiple authentication mechanisms that include API keys and Azure credentials.


```python
use_azure_active_directory = False  # Set this flag to True if you are using Azure Active Directory
```


#### Authentication using API key

To set up the OpenAI SDK to use an *Azure API Key*, we need to set up the `api_type` to `azure` and set `api_key` to a key associated with your endpoint (you can find this key in *"Keys and Endpoints"* under *"Resource Management"* in the [Azure Portal](https://portal.azure.com))


```python
if not use_azure_active_directory:
    openai.api_type = 'azure'
    openai.api_key = os.environ["OPENAI_API_KEY"]
```

#### Authentication using Microsoft Active Directory
Let's now see how we can get a key via Microsoft Active Directory Authentication. See the [documentation](https://learn.microsoft.com/azure/ai-services/openai/how-to/managed-identity) for more information on how to set this up.


```python
! pip install azure-identity
```


```python
from azure.identity import DefaultAzureCredential

if use_azure_active_directory:
    default_credential = DefaultAzureCredential()
    token = default_credential.get_token("https://cognitiveservices.azure.com/.default")

    openai.api_type = "azure_ad"
    openai.api_key = token.token
```

A token is valid for a period of time, after which it will expire. To ensure a valid token is sent with every request, you can refresh an expiring token by hooking into requests.auth:


```python
import typing
import time
import requests

if typing.TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class TokenRefresh(requests.auth.AuthBase):

    def __init__(self, credential: "TokenCredential", scopes: typing.List[str]) -> None:
        self.credential = credential
        self.scopes = scopes
        self.cached_token: typing.Optional[str] = None

    def __call__(self, req):
        if not self.cached_token or self.cached_token.expires_on - time.time() < 300:
            self.cached_token = self.credential.get_token(*self.scopes)
        req.headers["Authorization"] = f"Bearer {self.cached_token.token}"
        return req

```

## Chat completion model with your own data

### Setting the context

In this example, we want our model to base its responses on Azure AI services documentation data. Following the [Quickstart](https://learn.microsoft.com/azure/ai-services/openai/use-your-data-quickstart?tabs=command-line&pivots=programming-language-studio) shared previously, we have added the [markdown](https://github.com/MicrosoftDocs/azure-docs/blob/main/articles/ai-services/cognitive-services-and-machine-learning.md) file for the [Azure AI services and machine learning](https://learn.microsoft.com/azure/ai-services/cognitive-services-and-machine-learning) documentation page to our search index. The model is now ready to answer questions about Azure AI services and machine learning.

### Code

To chat with Azure OpenAI models using your own data with the Python SDK, we must first set up the code to target the chat completions extensions endpoint which is designed to work with your own data. To do this, we've created a convenience function that can be called to set a custom adapter for the library which will target the extensions endpoint for a given deployment ID.


```python
import requests

def setup_byod(deployment_id: str) -> None:
    """Sets up the OpenAI Python SDK to use your own data for the chat endpoint.
    
    :param deployment_id: The deployment ID for the model to use with your own data.

    To remove this configuration, simply set openai.requestssession to None.
    """

    class BringYourOwnDataAdapter(requests.adapters.HTTPAdapter):

        def send(self, request, **kwargs):
            request.url = f"{openai.api_base}/openai/deployments/{deployment_id}/extensions/chat/completions?api-version={openai.api_version}"
            return super().send(request, **kwargs)

    session = requests.Session()

    # Mount a custom adapter which will use the extensions endpoint for any call using the given `deployment_id`
    session.mount(
        prefix=f"{openai.api_base}/openai/deployments/{deployment_id}",
        adapter=BringYourOwnDataAdapter()
    )

    if use_azure_active_directory:
        session.auth = TokenRefresh(default_credential, ["https://cognitiveservices.azure.com/.default"])

    openai.requestssession = session

```

Now we can call the convenience function to configure the SDK with the model we plan to use for our own data.


```python
setup_byod("gpt-4")
```

Providing our search endpoint, key, and index name for the `dataSources` keyword argument, any questions posed to the model will now be grounded in our own data. An additional property, `context`, will be provided to show the data the model referenced to answer the question.


```python
completion = openai.ChatCompletion.create(
    messages=[{"role": "user", "content": "What are the differences between Azure Machine Learning and Azure AI services?"}],
    deployment_id="gpt-4",
    dataSources=[  # camelCase is intentional, as this is the format the API expects
        {
            "type": "AzureCognitiveSearch",
            "parameters": {
                "endpoint": os.environ["SEARCH_ENDPOINT"],
                "key": os.environ["SEARCH_KEY"],
                "indexName": os.environ["SEARCH_INDEX_NAME"],
            }
        }
    ]
)
print(completion)
```

    {
      "id": "65b485bb-b3c9-48da-8b6f-7d3a219f0b40",
      "model": "gpt-4",
      "created": 1693338769,
      "object": "extensions.chat.completion",
      "choices": [
        {
          "index": 0,
          "finish_reason": "stop",
          "message": {
            "role": "assistant",
            "content": "Azure AI services and Azure Machine Learning (AML) both aim to apply artificial intelligence (AI) to enhance business operations, but they target different audiences and offer different capabilities [doc1]. \n\nAzure AI services are designed for developers without machine learning experience and provide pre-trained models to solve general problems such as text analysis, image recognition, and natural language processing [doc5]. These services require general knowledge about your data without needing experience with machine learning or data science and provide REST APIs and language-based SDKs [doc2].\n\nOn the other hand, Azure Machine Learning is tailored for data scientists and involves a longer process of data collection, cleaning, transformation, algorithm selection, model training, and deployment [doc5]. It allows users to create custom solutions for highly specialized and specific problems, requiring familiarity with the subject matter, data, and expertise in data science [doc5].\n\nIn summary, Azure AI services offer pre-trained models for developers without machine learning experience, while Azure Machine Learning is designed for data scientists to create custom solutions for specific problems.",
            "end_turn": true,
            "context": {
              "messages": [
                {
                  "role": "tool",
                  "content": "{\"citations\": [{\"content\": \"<h2 id=\\\"how-are-azure-ai-services-and-azure-machine-learning-aml-similar\\\">How are Azure AI services and Azure Machine Learning (AML) similar?.</h2>\\n<p>Both have the end-goal of applying artificial intelligence (AI) to enhance business operations, though how each provides this in the respective offerings is different..</p>\\n<p>Generally, the audiences are different:</p>\\n<ul>\\n<li>Azure AI services are for developers without machine-learning experience..</li>\\n<li>Azure Machine Learning is tailored for data scientists.\", \"id\": null, \"title\": \"Azure AI services and machine learning\", \"filepath\": \"cognitive-services-and-machine-learning.md\", \"url\": \"https://krpraticstorageacc.blob.core.windows.net/azure-openai/cognitive-services-and-machine-learning.md\", \"metadata\": {\"chunking\": \"orignal document size=1188. Scores=5.689296 and None.Org Highlight count=160.Filtering to chunk no. 2/Highlights=30 of size=137\"}, \"chunk_id\": \"2\"}, {\"content\": \"<td>Convert speech into text and text into natural-sounding speech..Translate from one language to another and enable speaker verification and recognition..</td>\\n</tr>\\n<tr>\\n<td><a href=\\\"https://azure.microsoft.com/services/cognitive-services/directory/vision/\\\">Vision</a></td>\\n<td>Recognize, identify, caption, index, and moderate your pictures, videos, and digital ink content..</td>\\n</tr>\\n<p></tbody>\\n</table></p>\\n<p>Use Azure AI services when you:</p>\\n<ul>\\n<li>Can use a generalized solution..</li>\\n</ul>\\n<p>Use other machine-learning solutions when you:</p>\\n<ul>\\n<li>Need to choose the algorithm and need to train on very specific data..</li>\\n</ul>\\n<h2 id=\\\"what-is-machine-learning\\\">What is machine learning?.</h2>\\n<p>Machine learning is a concept where you bring together data and an algorithm to solve a specific need..Once the data and algorithm are trained, the output is a model that you can use again with different data..The trained model provides insights based on the new data..</p>\\n<p>The process of building a machine learning system requires some knowledge of machine learning or data science..</p>\\n<p>Machine learning is provided using <a href=\\\"/azure/architecture/data-guide/technology-choices/data-science-and-machine-learning?.context=azure%2fmachine-learning%2fstudio%2fcontext%2fml-context\\\">Azure Machine Learning (AML) products and services</a>..</p>\\n<h2 id=\\\"what-is-an-azure-ai-service\\\">What is an Azure AI service?.</h2>\\n<p>An Azure AI service provides part or all of the components in a machine learning solution: data, algorithm, and trained model..These services are meant to require general knowledge about your data without needing experience with machine learning or data science..These services provide both REST API(s) and language-based SDKs..As a result, you need to have programming language knowledge to use the services..</p>\", \"id\": null, \"title\": \"Azure AI services and machine learning\", \"filepath\": \"cognitive-services-and-machine-learning.md\", \"url\": \"https://krpraticstorageacc.blob.core.windows.net/azure-openai/cognitive-services-and-machine-learning.md\", \"metadata\": {\"chunking\": \"orignal document size=1188. Scores=5.689296 and None.Org Highlight count=160.Filtering to chunk no. 1/Highlights=67 of size=506\"}, \"chunk_id\": \"1\"}, {\"content\": \"<hr />\\n<p>title: Azure AI services and Machine Learning\\ntitleSuffix: Azure AI services\\ndescription: Learn where Azure AI services fits in with other Azure offerings for machine learning.\\nservices: cognitive-services\\nmanager: nitinme\\nauthor: aahill\\nms.author: aahi\\nms.service: cognitive-services\\nms.topic: conceptual\\nms.date: 10/28/2021</p>\\n<hr />\\n<h1 id=\\\"azure-ai-services-and-machine-learning\\\">Azure AI services and machine learning</h1>\\n<p>Azure AI services provides machine learning capabilities to solve general problems such as analyzing text for emotional sentiment or analyzing images to recognize objects or faces..You don't need special machine learning or data science knowledge to use these services../what-are-ai-services.md\\\">Azure AI services</a> is a group of services, each supporting different, generalized prediction capabilities..The services are divided into different categories to help you find the right service..</p>\\n<table>\\n<thead>\\n<tr>\\n<th>Service category</th>\\n<th>Purpose</th>\\n</tr>\\n</thead>\\n<tbody>\\n<tr>\\n<td><a href=\\\"https://azure.microsoft.com/services/cognitive-services/directory/decision/\\\">Decision</a></td>\\n<td>Build apps that surface recommendations for informed and efficient decision-making..</td>\\n</tr>\\n<tr>\\n<td><a href=\\\"https://azure.microsoft.com/services/cognitive-services/directory/lang/\\\">Language</a></td>\\n<td>Allow your apps to process natural language with pre-built scripts, evaluate sentiment and learn how to recognize what users want..</td>\\n</tr>\\n<tr>\\n<td><a href=\\\"https://azure.microsoft.com/services/cognitive-services/directory/search/\\\">Search</a></td>\\n<td>Add Bing Search APIs to your apps and harness the ability to comb billions of webpages, images, videos, and news with a single API call..</td>\\n</tr>\\n<tr>\\n<td><a href=\\\"https://azure.microsoft.com/services/cognitive-services/directory/speech/\\\">Speech</a></td>\", \"id\": null, \"title\": \"Azure AI services and machine learning\", \"filepath\": \"cognitive-services-and-machine-learning.md\", \"url\": \"https://krpraticstorageacc.blob.core.windows.net/azure-openai/cognitive-services-and-machine-learning.md\", \"metadata\": {\"chunking\": \"orignal document size=1188. Scores=5.689296 and None.Org Highlight count=160.Filtering to chunk no. 0/Highlights=63 of size=526\"}, \"chunk_id\": \"0\"}, {\"content\": \"<p>How is Azure Cognitive Search related to Azure AI services?</p>\\n<p><a href=\\\"../search/search-what-is-azure-search.md\\\">Azure Cognitive Search</a> is a separate cloud search service that optionally uses Azure AI services to add image and natural language processing to indexing workloads. Azure AI services is exposed in Azure Cognitive Search through <a href=\\\"../search/cognitive-search-predefined-skills.md\\\">built-in skills</a> that wrap individual APIs. You can use a free resource for walkthroughs, but plan on creating and attaching a <a href=\\\"../search/cognitive-search-attach-cognitive-services.md\\\">billable resource</a> for larger volumes.</p>\\n<h2 id=\\\"how-can-you-use-azure-ai-services\\\">How can you use Azure AI services?</h2>\\n<p>Each service provides information about your data. You can combine services together to chain solutions such as converting speech (audio) to text, translating the text into many languages, then using the translated languages to get answers from a knowledge base. While Azure AI services can be used to create intelligent solutions on their own, they can also be combined with traditional machine learning projects to supplement models or accelerate the development process. </p>\\n<p>Azure AI services that provide exported models for other machine learning tools:</p>\\n<table>\\n<thead>\\n<tr>\\n<th>Azure AI service</th>\\n<th>Model information</th>\\n</tr>\\n</thead>\\n<tbody>\\n<tr>\\n<td><a href=\\\"./custom-vision-service/overview.md\\\">Custom Vision</a></td>\\n<td><a href=\\\"./custom-vision-service/export-model-python.md\\\">Export</a> for Tensorflow for Android, CoreML for iOS11, ONNX for Windows ML</td>\\n</tr>\\n</tbody>\\n</table>\\n<h2 id=\\\"learn-more\\\">Learn more</h2>\\n<ul>\\n<li><a href=\\\"/azure/architecture/data-guide/technology-choices/data-science-and-machine-learning\\\">Architecture Guide - What are the machine learning products at Microsoft?</a></li>\\n<li><a href=\\\"../machine-learning/concept-deep-learning-vs-machine-learning.md\\\">Machine learning - Introduction to deep learning vs. machine learning</a></li>\\n</ul>\\n<h2 id=\\\"next-steps\\\">Next steps</h2>\\n<ul>\\n<li>Create your Azure AI services resource in the <a href=\\\"multi-service-resource.md?pivots=azportal\\\">Azure portal</a> or with <a href=\\\"./multi-service-resource.md?pivots=azcli\\\">Azure CLI</a>.</li>\\n<li>Learn how to <a href=\\\"authentication.md\\\">authenticate</a> with your Azure AI service.</li>\\n<li>Use <a href=\\\"diagnostic-logging.md\\\">diagnostic logging</a> for issue identification and debugging. </li>\\n<li>Deploy an Azure AI service in a Docker <a href=\\\"cognitive-services-container-support.md\\\">container</a>.</li>\\n<li>Keep up to date with <a href=\\\"https://azure.microsoft.com/updates/?product=cognitive-services\\\">service updates</a>.</li>\\n</ul>\", \"id\": null, \"title\": \"Azure AI services and machine learning\", \"filepath\": \"cognitive-services-and-machine-learning.md\", \"url\": \"https://krpraticstorageacc.blob.core.windows.net/azure-openai/cognitive-services-and-machine-learning.md\", \"metadata\": {\"chunking\": \"orignal document size=793. Scores=3.3767838 and None.Org Highlight count=69.\"}, \"chunk_id\": \"3\"}, {\"content\": \"<p>How are Azure AI services different from machine learning?.</p>\\n<p>Azure AI services provide a trained model for you..This brings data and an algorithm together, available from a REST API(s) or SDK..An Azure AI service provides answers to general problems such as key phrases in text or item identification in images..</p>\\n<p>Machine learning is a process that generally requires a longer period of time to implement successfully..This time is spent on data collection, cleaning, transformation, algorithm selection, model training, and deployment to get to the same level of functionality provided by an Azure AI service..With machine learning, it is possible to provide answers to highly specialized and/or specific problems..Machine learning problems require familiarity with the specific subject matter and data of the problem under consideration, as well as expertise in data science..</p>\\n<h2 id=\\\"what-kind-of-data-do-you-have\\\">What kind of data do you have?.</h2>\\n<p>Azure AI services, as a group of services, can require none, some, or all custom data for the trained model..</p>\\n<h3 id=\\\"no-additional-training-data-required\\\">No additional training data required</h3>\\n<p>Services that provide a fully-trained model can be treated as a <em>opaque box</em>..You don't need to know how they work or what data was used to train them..</p>\\n<h3 id=\\\"some-or-all-training-data-required\\\">Some or all training data required</h3>\\n<p>Some services allow you to bring your own data, then train a model..This allows you to extend the model using the Service's data and algorithm with your own data..The output matches your needs..When you bring your own data, you may need to tag the data in a way specific to the service..For example, if you are training a model to identify flowers, you can provide a catalog of flower images along with the location of the flower in each image to train the model..These services process significant amounts of model data..</p>\\n<h2 id=\\\"service-requirements-for-the-data-model\\\">Service requirements for the data model</h2>\\n<p>The following data categorizes each service by which kind of data it allows or requires..</p>\\n<table>\\n<thead>\\n<tr>\\n<th>Azure AI service</th>\\n<th>No training data required</th>\\n<th>You provide some or all training data</th>\\n<th>Real-time or near real-time data collection</th>\\n</tr>\\n</thead>\\n<tbody>\\n<tr>\\n<td><a href=\\\"../LUIS/what-is-luis.md\\\">Language Understanding (LUIS)</a></td>\\n<td></td>\\n<td>x</td>\\n<td></td>\\n</tr>\\n<tr>\\n<td><a href=\\\"../personalizer/what-is-personalizer.md\\\">Personalizer</a><sup>1</sup></sup></td>\\n<td>x</td>\\n<td>x</td>\\n<td>x</td>\\n</tr>\\n<tr>\\n<td><a href=\\\"../computer-vision/overview.md\\\">Vision</a></td>\\n<td>x</td>\\n<td></td>\\n<td></td>\\n</tr>\\n</tbody>\\n</table>\\n<p><sup>1</sup> Personalizer only needs training data collected by the service (as it operates in real-time) to evaluate your policy and data..</p>\\n<h2 id=\\\"where-can-you-use-azure-ai-services\\\">Where can you use Azure AI services?.</h2>\\n<p>The services are used in any application that can make REST API(s) or SDK calls..Examples of applications include web sites, bots, virtual or mixed reality, desktop and mobile applications.\", \"id\": null, \"title\": \"Azure AI services and machine learning\", \"filepath\": \"cognitive-services-and-machine-learning.md\", \"url\": \"https://krpraticstorageacc.blob.core.windows.net/azure-openai/cognitive-services-and-machine-learning.md\", \"metadata\": {\"chunking\": \"orignal document size=1734. Scores=3.1447978 and None.Org Highlight count=66.Filtering to highlight size=891\"}, \"chunk_id\": \"4\"}], \"intent\": \"[\\\"What are the differences between Azure Machine Learning and Azure AI services?\\\"]\"}",
                  "end_turn": false
                }
              ]
            }
          }
        }
      ]
    }
    

If you would prefer to stream the response from the model, you can pass the `stream=True` keyword argument:


```python
response = openai.ChatCompletion.create(
    messages=[{"role": "user", "content": "What are the differences between Azure Machine Learning and Azure AI services?"}],
    deployment_id="gpt-4",
    dataSources=[
        {
            "type": "AzureCognitiveSearch",
            "parameters": {
                "endpoint": os.environ["SEARCH_ENDPOINT"],
                "key": os.environ["SEARCH_KEY"],
                "indexName": os.environ["SEARCH_INDEX_NAME"],
            }
        }
    ],
    stream=True,
)

for chunk in response:
    delta = chunk.choices[0].delta

    if "role" in delta:
        print("\n"+ delta.role + ": ", end="", flush=True)
    if "content" in delta:
        print(delta.content, end="", flush=True)
    if "context" in delta:
        print(f"Context: {delta.context}", end="", flush=True)
```

    Context: {
      "messages": [
        {
          "role": "tool",
          "content": "{\"citations\":[{\"content\":\"<h2 id=\\\"how-are-azure-ai-services-and-azure-machine-learning-aml-similar\\\">How are Azure AI services and Azure Machine Learning (AML) similar?.</h2>\\n<p>Both have the end-goal of applying artificial intelligence (AI) to enhance business operations, though how each provides this in the respective offerings is different..</p>\\n<p>Generally, the audiences are different:</p>\\n<ul>\\n<li>Azure AI services are for developers without machine-learning experience..</li>\\n<li>Azure Machine Learning is tailored for data scientists.\",\"id\":null,\"title\":\"Azure AI services and machine learning\",\"filepath\":\"cognitive-services-and-machine-learning.md\",\"url\":\"https://krpraticstorageacc.blob.core.windows.net/azure-openai/cognitive-services-and-machine-learning.md\",\"metadata\":{\"chunking\":\"orignal document size=1188. Scores=5.689296 and None.Org Highlight count=160.Filtering to chunk no. 2/Highlights=30 of size=137\"},\"chunk_id\":\"2\"},{\"content\":\"<td>Convert speech into text and text into natural-sounding speech..Translate from one language to another and enable speaker verification and recognition..</td>\\n</tr>\\n<tr>\\n<td><a href=\\\"https://azure.microsoft.com/services/cognitive-services/directory/vision/\\\">Vision</a></td>\\n<td>Recognize, identify, caption, index, and moderate your pictures, videos, and digital ink content..</td>\\n</tr>\\n<p></tbody>\\n</table></p>\\n<p>Use Azure AI services when you:</p>\\n<ul>\\n<li>Can use a generalized solution..</li>\\n</ul>\\n<p>Use other machine-learning solutions when you:</p>\\n<ul>\\n<li>Need to choose the algorithm and need to train on very specific data..</li>\\n</ul>\\n<h2 id=\\\"what-is-machine-learning\\\">What is machine learning?.</h2>\\n<p>Machine learning is a concept where you bring together data and an algorithm to solve a specific need..Once the data and algorithm are trained, the output is a model that you can use again with different data..The trained model provides insights based on the new data..</p>\\n<p>The process of building a machine learning system requires some knowledge of machine learning or data science..</p>\\n<p>Machine learning is provided using <a href=\\\"/azure/architecture/data-guide/technology-choices/data-science-and-machine-learning?.context=azure%2fmachine-learning%2fstudio%2fcontext%2fml-context\\\">Azure Machine Learning (AML) products and services</a>..</p>\\n<h2 id=\\\"what-is-an-azure-ai-service\\\">What is an Azure AI service?.</h2>\\n<p>An Azure AI service provides part or all of the components in a machine learning solution: data, algorithm, and trained model..These services are meant to require general knowledge about your data without needing experience with machine learning or data science..These services provide both REST API(s) and language-based SDKs..As a result, you need to have programming language knowledge to use the services..</p>\",\"id\":null,\"title\":\"Azure AI services and machine learning\",\"filepath\":\"cognitive-services-and-machine-learning.md\",\"url\":\"https://krpraticstorageacc.blob.core.windows.net/azure-openai/cognitive-services-and-machine-learning.md\",\"metadata\":{\"chunking\":\"orignal document size=1188. Scores=5.689296 and None.Org Highlight count=160.Filtering to chunk no. 1/Highlights=67 of size=506\"},\"chunk_id\":\"1\"},{\"content\":\"<hr />\\n<p>title: Azure AI services and Machine Learning\\ntitleSuffix: Azure AI services\\ndescription: Learn where Azure AI services fits in with other Azure offerings for machine learning.\\nservices: cognitive-services\\nmanager: nitinme\\nauthor: aahill\\nms.author: aahi\\nms.service: cognitive-services\\nms.topic: conceptual\\nms.date: 10/28/2021</p>\\n<hr />\\n<h1 id=\\\"azure-ai-services-and-machine-learning\\\">Azure AI services and machine learning</h1>\\n<p>Azure AI services provides machine learning capabilities to solve general problems such as analyzing text for emotional sentiment or analyzing images to recognize objects or faces..You don't need special machine learning or data science knowledge to use these services../what-are-ai-services.md\\\">Azure AI services</a> is a group of services, each supporting different, generalized prediction capabilities..The services are divided into different categories to help you find the right service..</p>\\n<table>\\n<thead>\\n<tr>\\n<th>Service category</th>\\n<th>Purpose</th>\\n</tr>\\n</thead>\\n<tbody>\\n<tr>\\n<td><a href=\\\"https://azure.microsoft.com/services/cognitive-services/directory/decision/\\\">Decision</a></td>\\n<td>Build apps that surface recommendations for informed and efficient decision-making..</td>\\n</tr>\\n<tr>\\n<td><a href=\\\"https://azure.microsoft.com/services/cognitive-services/directory/lang/\\\">Language</a></td>\\n<td>Allow your apps to process natural language with pre-built scripts, evaluate sentiment and learn how to recognize what users want..</td>\\n</tr>\\n<tr>\\n<td><a href=\\\"https://azure.microsoft.com/services/cognitive-services/directory/search/\\\">Search</a></td>\\n<td>Add Bing Search APIs to your apps and harness the ability to comb billions of webpages, images, videos, and news with a single API call..</td>\\n</tr>\\n<tr>\\n<td><a href=\\\"https://azure.microsoft.com/services/cognitive-services/directory/speech/\\\">Speech</a></td>\",\"id\":null,\"title\":\"Azure AI services and machine learning\",\"filepath\":\"cognitive-services-and-machine-learning.md\",\"url\":\"https://krpraticstorageacc.blob.core.windows.net/azure-openai/cognitive-services-and-machine-learning.md\",\"metadata\":{\"chunking\":\"orignal document size=1188. Scores=5.689296 and None.Org Highlight count=160.Filtering to chunk no. 0/Highlights=63 of size=526\"},\"chunk_id\":\"0\"},{\"content\":\"<p>How is Azure Cognitive Search related to Azure AI services?</p>\\n<p><a href=\\\"../search/search-what-is-azure-search.md\\\">Azure Cognitive Search</a> is a separate cloud search service that optionally uses Azure AI services to add image and natural language processing to indexing workloads. Azure AI services is exposed in Azure Cognitive Search through <a href=\\\"../search/cognitive-search-predefined-skills.md\\\">built-in skills</a> that wrap individual APIs. You can use a free resource for walkthroughs, but plan on creating and attaching a <a href=\\\"../search/cognitive-search-attach-cognitive-services.md\\\">billable resource</a> for larger volumes.</p>\\n<h2 id=\\\"how-can-you-use-azure-ai-services\\\">How can you use Azure AI services?</h2>\\n<p>Each service provides information about your data. You can combine services together to chain solutions such as converting speech (audio) to text, translating the text into many languages, then using the translated languages to get answers from a knowledge base. While Azure AI services can be used to create intelligent solutions on their own, they can also be combined with traditional machine learning projects to supplement models or accelerate the development process. </p>\\n<p>Azure AI services that provide exported models for other machine learning tools:</p>\\n<table>\\n<thead>\\n<tr>\\n<th>Azure AI service</th>\\n<th>Model information</th>\\n</tr>\\n</thead>\\n<tbody>\\n<tr>\\n<td><a href=\\\"./custom-vision-service/overview.md\\\">Custom Vision</a></td>\\n<td><a href=\\\"./custom-vision-service/export-model-python.md\\\">Export</a> for Tensorflow for Android, CoreML for iOS11, ONNX for Windows ML</td>\\n</tr>\\n</tbody>\\n</table>\\n<h2 id=\\\"learn-more\\\">Learn more</h2>\\n<ul>\\n<li><a href=\\\"/azure/architecture/data-guide/technology-choices/data-science-and-machine-learning\\\">Architecture Guide - What are the machine learning products at Microsoft?</a></li>\\n<li><a href=\\\"../machine-learning/concept-deep-learning-vs-machine-learning.md\\\">Machine learning - Introduction to deep learning vs. machine learning</a></li>\\n</ul>\\n<h2 id=\\\"next-steps\\\">Next steps</h2>\\n<ul>\\n<li>Create your Azure AI services resource in the <a href=\\\"multi-service-resource.md?pivots=azportal\\\">Azure portal</a> or with <a href=\\\"./multi-service-resource.md?pivots=azcli\\\">Azure CLI</a>.</li>\\n<li>Learn how to <a href=\\\"authentication.md\\\">authenticate</a> with your Azure AI service.</li>\\n<li>Use <a href=\\\"diagnostic-logging.md\\\">diagnostic logging</a> for issue identification and debugging. </li>\\n<li>Deploy an Azure AI service in a Docker <a href=\\\"cognitive-services-container-support.md\\\">container</a>.</li>\\n<li>Keep up to date with <a href=\\\"https://azure.microsoft.com/updates/?product=cognitive-services\\\">service updates</a>.</li>\\n</ul>\",\"id\":null,\"title\":\"Azure AI services and machine learning\",\"filepath\":\"cognitive-services-and-machine-learning.md\",\"url\":\"https://krpraticstorageacc.blob.core.windows.net/azure-openai/cognitive-services-and-machine-learning.md\",\"metadata\":{\"chunking\":\"orignal document size=793. Scores=3.3767838 and None.Org Highlight count=69.\"},\"chunk_id\":\"3\"},{\"content\":\"<p>How are Azure AI services different from machine learning?.</p>\\n<p>Azure AI services provide a trained model for you..This brings data and an algorithm together, available from a REST API(s) or SDK..An Azure AI service provides answers to general problems such as key phrases in text or item identification in images..</p>\\n<p>Machine learning is a process that generally requires a longer period of time to implement successfully..This time is spent on data collection, cleaning, transformation, algorithm selection, model training, and deployment to get to the same level of functionality provided by an Azure AI service..With machine learning, it is possible to provide answers to highly specialized and/or specific problems..Machine learning problems require familiarity with the specific subject matter and data of the problem under consideration, as well as expertise in data science..</p>\\n<h2 id=\\\"what-kind-of-data-do-you-have\\\">What kind of data do you have?.</h2>\\n<p>Azure AI services, as a group of services, can require none, some, or all custom data for the trained model..</p>\\n<h3 id=\\\"no-additional-training-data-required\\\">No additional training data required</h3>\\n<p>Services that provide a fully-trained model can be treated as a <em>opaque box</em>..You don't need to know how they work or what data was used to train them..</p>\\n<h3 id=\\\"some-or-all-training-data-required\\\">Some or all training data required</h3>\\n<p>Some services allow you to bring your own data, then train a model..This allows you to extend the model using the Service's data and algorithm with your own data..The output matches your needs..When you bring your own data, you may need to tag the data in a way specific to the service..For example, if you are training a model to identify flowers, you can provide a catalog of flower images along with the location of the flower in each image to train the model..These services process significant amounts of model data..</p>\\n<h2 id=\\\"service-requirements-for-the-data-model\\\">Service requirements for the data model</h2>\\n<p>The following data categorizes each service by which kind of data it allows or requires..</p>\\n<table>\\n<thead>\\n<tr>\\n<th>Azure AI service</th>\\n<th>No training data required</th>\\n<th>You provide some or all training data</th>\\n<th>Real-time or near real-time data collection</th>\\n</tr>\\n</thead>\\n<tbody>\\n<tr>\\n<td><a href=\\\"../LUIS/what-is-luis.md\\\">Language Understanding (LUIS)</a></td>\\n<td></td>\\n<td>x</td>\\n<td></td>\\n</tr>\\n<tr>\\n<td><a href=\\\"../personalizer/what-is-personalizer.md\\\">Personalizer</a><sup>1</sup></sup></td>\\n<td>x</td>\\n<td>x</td>\\n<td>x</td>\\n</tr>\\n<tr>\\n<td><a href=\\\"../computer-vision/overview.md\\\">Vision</a></td>\\n<td>x</td>\\n<td></td>\\n<td></td>\\n</tr>\\n</tbody>\\n</table>\\n<p><sup>1</sup> Personalizer only needs training data collected by the service (as it operates in real-time) to evaluate your policy and data..</p>\\n<h2 id=\\\"where-can-you-use-azure-ai-services\\\">Where can you use Azure AI services?.</h2>\\n<p>The services are used in any application that can make REST API(s) or SDK calls..Examples of applications include web sites, bots, virtual or mixed reality, desktop and mobile applications.\",\"id\":null,\"title\":\"Azure AI services and machine learning\",\"filepath\":\"cognitive-services-and-machine-learning.md\",\"url\":\"https://krpraticstorageacc.blob.core.windows.net/azure-openai/cognitive-services-and-machine-learning.md\",\"metadata\":{\"chunking\":\"orignal document size=1734. Scores=3.1447978 and None.Org Highlight count=66.Filtering to highlight size=891\"},\"chunk_id\":\"4\"}],\"intent\":\"[\\\"What are the differences between Azure Machine Learning and Azure AI services?\\\"]\"}",
          "end_turn": false
        }
      ]
    }
    assistant: Azure AI services and Azure Machine Learning (AML) both aim to apply artificial intelligence (AI) to enhance business operations, but they target different audiences and offer different capabilities [doc1]. 
    
    Azure AI services are designed for developers without machine learning experience and provide pre-trained models to solve general problems such as text analysis, image recognition, and natural language processing [doc5]. These services require general knowledge about your data without needing experience with machine learning or data science and provide REST APIs and language-based SDKs [doc2].
    
    On the other hand, Azure Machine Learning is tailored for data scientists and offers a platform to build, train, and deploy custom machine learning models [doc1]. It requires knowledge of machine learning or data science and allows users to choose the algorithm and train on very specific data [doc2].
    
    In summary, Azure AI services offer pre-trained models for developers without machine learning experience, while Azure Machine Learning is a platform for data scientists to build and deploy custom machine learning models.
