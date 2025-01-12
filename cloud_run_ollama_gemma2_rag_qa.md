```
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Cloud Run GPU Inference: Gemma 2 RAG Q&A with Ollama and LangChain

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/open-models/serving/cloud_run_ollama_gemma2_rag_qa.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fopen-models%2Fserving%2Fcloud_run_ollama_gemma2_rag_qa.ipynb">
      <img width="32px" src="https://cloud.google.com/ml-engine/images/colab-enterprise-logo-32px.png" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/open-models/serving/cloud_run_ollama_gemma2_rag_qa.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/open-models/serving/cloud_run_ollama_gemma2_rag_qa.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
| Author(s) | [Elia Secchi](https://github.com/eliasecchig/) |

## Overview



> **[Cloud Run](https://cloud.google.com/run)**:
It's a serverless platform by Google Cloud for running containerized applications. It automatically scales and manages infrastructure, supporting various programming languages. Cloud Run now offers GPU acceleration for AI/ML workloads.

> **Note:** GPU support in Cloud Run is a guarded feature. Before running this notebook, make sure your Google Cloud project is enabled. You can do that by visiting this page [g.co/cloudrun/gpu](https://g.co/cloudrun/gpu).


> **[Ollama](ollama.com)**: is an open-source tool for easily running and deploying large language models locally. It offers simple management and usage of LLMs on personal computers or servers.

This notebook showcase how to deploy [Google Gemma 2](https://blog.google/technology/developers/google-gemma-2/) in Cloud Run, with the objective to build a simple RAG Q&A application.

By the end of this notebook, you will learn how to:

1. Deploy Google Gemma 2 on Cloud Run using Ollama
2. Implement a Retrieval-Augmented Generation (RAG) application with Gemma 2 and Ollama
3. Build a custom container with Ollama to deploy any Large Language Model (LLM) of your choice



### Required roles

To get the permissions that you need to complete the tutorial, ask your administrator to grant you the following IAM roles on your project:

1. Artifact Registry Administrator (`roles/artifactregistry.admin`)
2. Cloud Build Editor (`roles/cloudbuild.builds.editor`)
3. Cloud Run Admin (`roles/run.developer`)
4. Service Account User (`roles/iam.serviceAccountUser`)
5. Service Usage Consumer (`roles/serviceusage.serviceUsageConsumer`)
6. Storage Admin (`roles/storage.admin`)



For more information about granting roles, see [Manage access](https://cloud.google.com/iam/docs/granting-changing-revoking-access).

![cloud_run_gemma_ollama.png](https://storage.googleapis.com/github-repo/generative-ai/open-models/serving/cloud_run_gemma_ollama.png)

## Get started

### Install Vertex AI SDK and other required packages


```
%pip install --upgrade --user --quiet google-cloud-aiplatform langchain-community langchainhub langchain_google_vertexai
```

### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.


```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>

### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.


```
!gcloud auth login --update-adc --quiet
```

### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
# Use the environment variable if the user doesn't provide Project ID.
import os

import vertexai

PROJECT_ID = "[your-project-id]"  # @param {type:"string", isTemplate: true}
if PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Fetch your Google Cloud project number


```
PROJECT_NUMBER = get_ipython().getoutput('gcloud projects describe $PROJECT_ID --format="value(projectNumber)"')[0]
```

## Deploy Ollama with Cloud Run

## Build your container

For deploying Gemma 2 in Cloud Run, create a container that packages the Ollama server and the Gemma 2 model.

To build the container, you can use [Cloud Build](https://cloud.google.com/build), a serverless CI/CD platform which allows developers to easily build software.

> For optimal startup time and improved scalability, it's recommended to store model weights for Gemma 2 (9B) and similarly sized models directly in the container image.
However, consider the storage requirements of larger models as they might be impractical to store in the container image. Refer to [Best practices: AI inference on Cloud Run with GPUs](https://cloud.google.com/run/docs/configuring/services/gpu-best-practices#loading-storing-models-tradeoff) for an overview of the trade-offs.

### Create Artifact Registry repository

To build a container you will need to first create a repository in Google Cloud Artifact Registry:


```
AR_REPOSITORY_NAME = "cr-gpu-repo"
```


```
!gcloud artifacts repositories create $AR_REPOSITORY_NAME \
      --repository-format=docker \
      --location=$LOCATION \
      --project=$PROJECT_ID
```

### Create a Dockerfile

You will then need to create a Dockerfile which defines the build steps of the container.

You can customize the model used by modifying the `MODEL_NAME` variable. 
Explore the [Ollama library](https://ollama.com/library) for a comprehensive list of available models.


```
MODEL_NAME = "gemma2:9b"
```


```
dockerfile_content = f"""
FROM ollama/ollama

# Set the host and port to listen on
ENV OLLAMA_HOST 0.0.0.0:8080

# Set the directory to store model weight files
ENV OLLAMA_MODELS /models

# Reduce the verbosity of the logs
ENV OLLAMA_DEBUG false

# Do not unload model weights from the GPU
ENV OLLAMA_KEEP_ALIVE -1

# Choose the model to load. Ollama defaults to 4-bit quantized weights
ENV MODEL {MODEL_NAME}

# Start the ollama server and download the model weights
RUN ollama serve & sleep 5 && ollama pull $MODEL

# At startup time we start the server and run a dummy request
# to request the model to be loaded in the GPU memory
ENTRYPOINT ["/bin/sh"]
CMD ["-c", "ollama serve  & (ollama run $MODEL 'Say one word' &) && wait"]
"""

# Write the Dockerfile
with open("Dockerfile", "w") as f:
    f.write(dockerfile_content)
```

### Trigger Cloud Build

You are now ready to trigger the container build process!
We will use the `gcloud builds submit` command, using a `e2-highcpu-32` machine to optimize build time. We use e2-highcpu-32 machines because multiple cores allow for parallel downloads, significantly speeding up the build process.

Cloud Build pricing is based on build minutes consumed. See [the pricing page](https://cloud.google.com/build/pricing) for details

The operation will take ~10 minutes for completion.


```
CONTAINER_URI = (
    f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/{AR_REPOSITORY_NAME}/ollama-gemma-2"
)
```


```
!gcloud builds submit --tag $CONTAINER_URI --project $PROJECT_ID --machine-type e2-highcpu-32
```

You can now use the container you just built to deploy a new Cloud Run service!

### Deploy container in Cloud Run

You are now ready for deployment! Cloud Run offers multiple deployment methods, including Console, gcloud CLI, Cloud Code, Terraform, YAML, and Client Libraries. Explore all the options in the [official documentation](https://cloud.google.com/run/docs/deploying#service).

For quick prototyping, you can start with the gcloud CLI `gcloud run deploy` command. This convenient command-line tool provides a straightforward way to get your container running on Cloud Run. Learn more about its features and usage in the [gcloud CLI reference](https://cloud.google.com/sdk/gcloud/reference/run/deploy).


```
SERVICE_NAME = "ollama-gemma-2"  # @param {type:"string"}
```


```
!gcloud beta run deploy $SERVICE_NAME \
    --project $PROJECT_ID \
    --region $LOCATION \
    --image $CONTAINER_URI \
    --concurrency 4 \
    --cpu 8 \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --max-instances 7 \
    --memory 32Gi \
    --no-allow-unauthenticated \
    --no-cpu-throttling \
    --timeout=600
```

*Expect a slower initial deployment as the container image is being pulled for the first time.*

### Setting concurrency for optimal performance

In Cloud Run, [concurrency](https://cloud.google.com/run/docs/about-concurrency) defines the maximum number of requests that can be processed simultaneously by a given instance.

For this sample we set a `concurrency` value equal to 4.

As part of your use case you might need to experiment with different concurrency settings to find the best latency vs throughput tradeoff.

Refer to the following documentation pages to know more about performance optimizations:
- [Setting concurrency for optimal performance in Cloud Run](https://cloud.google.com/run/docs/tutorials/gpu-gemma2-with-ollama#set-concurrency-for-performance)
- [GPU performance best practices](https://cloud.google.com/run/docs/configuring/services/gpu-best-practices)

## Invoking Gemma 2 in Cloud Run

We are now ready to send some requests to Gemma!

### Fetch identity token

Once deployed to Cloud Run, to invoke Gemma 2, we will need to fetch an Identity token to perform authentication. See the relative documentation to discover more about [authentication in Cloud Run](https://cloud.google.com/run/docs/authenticating/overview).

In the appendix of this sample, you'll find a helper function that supports the automatic refresh of the [Identity Token](https://cloud.google.com/docs/authentication/token-types#id), which expires every hour by default.


```
ID_TOKEN = get_ipython().getoutput('gcloud auth print-identity-token -q')[0]
```

### Setup the Service URL


```
SERVICE_URL = f"https://{SERVICE_NAME}-{PROJECT_NUMBER}.{LOCATION}.run.app"  # type: ignore
```

## Invoking Gemma

You are ready to test the model you just deployed! The [Ollama API docs](https://github.com/ollama/ollama/blob/main/docs/api.md) are a great resource to learn more about the different endpoints and how to interact with your model.

#### Invoke through CURL request
You can invoke Gemma and Cloud Run in many ways. For example, you can send an HTTP CURL request to Cloud Run:


```
ENDPOINT_URL = f"{SERVICE_URL}/api/generate"
```


```bash
%%bash -s "$ENDPOINT_URL" "$ID_TOKEN" "$MODEL_NAME" 
ENDPOINT_URL=$1
ID_TOKEN=$2
MODEL_NAME=$3

curl -s -X POST "${ENDPOINT_URL}" \
-H "Authorization: Bearer ${ID_TOKEN}" \
-H "Content-Type: application/json" \
-d '{ "model": "'${MODEL_NAME}'", "prompt": "Hi", "max_tokens": 100, "stream": false}'
```

#### Invoke with a Python POST Request

You can also invoke the model using a POST request with Python's popular `requests` library.  [Learn more about the `requests` library here.](https://requests.readthedocs.io/en/latest/) 


```
import requests

headers = {"Authorization": f"Bearer {ID_TOKEN}", "Content-Type": "application/json"}  # type: ignore

data = {
    "model": MODEL_NAME,
    "prompt": "Hi, I am using python!",
    "max_tokens": 100,
    "stream": False,
}

response = requests.post(ENDPOINT_URL, headers=headers, json=data)

print(response.text)
```

#### Invoke Ollama with Python integrations

Popular Generative AI orchestration frameworks like [LangChain](https://www.langchain.com) and [LlamaIndex](https://www.llamaindex.ai/) offer direct integration with Ollama:
- [LangChain integration](https://python.langchain.com/v0.2/docs/integrations/llms/ollama/)
- [LlamaIndex integration](https://docs.llamaindex.ai/en/stable/api_reference/llms/ollama/)

As part of this sample, we will be using the LangChain integration to perform different calls and build a sample RAG chain.

### Import libraries


```
import google.auth
from langchain.schema import BaseMessage, Document
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
```


```
llm = ChatOllama(
    model=MODEL_NAME,
    base_url=SERVICE_URL,
    num_predict=300,
    headers={"Authorization": f"Bearer {ID_TOKEN}"},  # type: ignore
)
```


```
# You can perform a synchronous invocation through the `.invoke` method

llm.invoke("Hi!")
```

Or invoke through the generation of a stream through the `.stream` **method**


```
# You can also generate a stream through the `.stream` method

for m in llm.stream("Hi!"):
    print(m)
```

## RAG Q&A Chain with Gemma 2 and Cloud Run

We can leverage the LangChain integration to create a sample RAG application with Gemma, Cloud Run, [Vertex AI Embedding](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings) for generating embeddings and [FAISS vector store](https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/) for document retrieval.

Through RAG, we will ask Gemma 2 to answer questions about the [Cloud Run documentation page](https://cloud.google.com/run/docs/overview/what-is-cloud-run)

### Setup embedding model and retriever

We are ready to setup our embedding model and retriever.


```
credentials, _ = google.auth.default(quota_project_id=PROJECT_ID)
embeddings = VertexAIEmbeddings(
    project=PROJECT_ID, model_name="text-embedding-004", credentials=credentials
)
```


```
loader = WebBaseLoader("https://cloud.google.com/run/docs/overview/what-is-cloud-run")
docs = loader.load()
documents = CharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(
    docs
)

vector = SKLearnVectorStore.from_documents(documents, embeddings)
retriever = vector.as_retriever()
```

### RAG Chain Definition

We will define now our RAG Chain.

The RAG chain works as follows:

1. The user's query and conversation history are passed to the `query_rewrite_chain` to generate a rewritten query optimized for semantic search.
2. The rewritten query is used by the `retriever` to fetch relevant documents.
3. The retrieved documents are formatted into a single string.
4. The formatted documents, along with the original user messages, are passed to the LLM with instructions to generate an answer based on the provided context.
5. The LLM's response is parsed and returned as the final answer.


```
answer_generation_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question answering-tasks. "
            "Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. "
            "{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
query_rewrite_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite a query to a semantic search engine using the current conversation. "
            "Provide only the rewritten query as output.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
```


```
query_rewrite_chain = query_rewrite_template | llm


def extract_query(messages: list[BaseMessage]) -> str:
    return query_rewrite_chain.invoke(messages).content


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": extract_query | retriever | format_docs,
        "messages": RunnablePassthrough(),
    }
    | answer_generation_template
    | llm
    | StrOutputParser()
)
```

### Testing the RAG Chain


```
rag_chain.invoke([("human", "What features does Cloud Run offer?")])
```

Now, let's use a specific question from the documentation to explore how RAG addresses potential gaps in the model's knowledge.


```
QUESTION = "List all the different Cloud Run integrations"
```

First, we'll ask the LLM directly:


```
print(llm.invoke(QUESTION).content)
```

Then, we'll ask the same question using the RAG chain:


```
print(rag_chain.invoke([("human", QUESTION)]))
```

We can notice how RAG chain provides a more accurate and comprehensive answer than the LLM by leveraging the [source documentation](https://cloud.google.com/run/docs/overview/what-is-cloud-run). 

## Conclusion
Congratulations. Now you know how to deploy an open model to Cloud Run powered by a GPU! Specifically, you deployed a Gemma 2 model to Cloud Run with a GPU, as part of a RAG application powered by LangChain. You were able to ask answers from Gemma 2 about a documentation page.

For more information about your identity tokens expiring and how to refresh your tokens, see the next section below "Appendix: Handling Identity Token Expiration".

To clean up the resources you created in this section, see the section at the bottom "Cleaning up".

## Appendix: Handling Identity Token Expiration

When deploying a Generative AI application Google Cloud Run, you'll often need to authenticate your requests using Identity Tokens.

These tokens will expire hourly, requiring a mechanism for automatic refresh to ensure uninterrupted operation.

The following helper classes provide an example of how to deal with  token refresh. 
It leverages:
1. The `google.auth` library to handle the authentication process and automatically refresh the token when necessary
2. ChatOllama's [auth parameter](https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.ollama.ChatOllama.html#langchain_community.chat_models.ollama.ChatOllama.auth) for passing an authentication callable


See the following resources for more information on authentication:
* [Identity Token Overview](https://cloud.google.com/docs/authentication/token-types#id)
* [Google Cloud Run Authentication Documentation](https://cloud.google.com/run/docs/authenticating/overview)


```
import time

import google.auth
from google.auth.credentials import Credentials
from google.auth.exceptions import DefaultCredentialsError
import google.auth.transport.requests
import google.oauth2.id_token
from requests.auth import AuthBase
from requests.models import PreparedRequest


class GoogleCloudAuth(AuthBase):
    def __init__(self, url: str, token_lifetime: int = 3600):
        self.url: str = url
        self.token: str | None = None
        self.expiry_time: float = 0
        self.token_lifetime: int = token_lifetime
        self.creds: Credentials
        self.creds, _ = google.auth.default()

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        r.headers["Authorization"] = f"Bearer {self.get_token()}"
        return r

    def get_token(self) -> str | None:
        if time.time() >= self.expiry_time:
            self.refresh_token()
        return self.token

    def refresh_token(self) -> None:
        """
        Retrieves an ID token, attempting to use default credentials first,
        and falling back to fetching a service-to-service new token if necessary.
        See more on Cloud Run authentication at this link:
         https://cloud.google.com/run/docs/authenticating/service-to-service
        Args:
            url: The URL to use for the token request.
        """
        auth_req = google.auth.transport.requests.Request()
        try:
            self.token = google.oauth2.id_token.fetch_id_token(auth_req, self.url)
        except DefaultCredentialsError:
            self.creds.refresh(auth_req)
            self.token = self.creds.id_token
        self.expiry_time = time.time() + self.token_lifetime
```


```
llm = ChatOllama(
    auth=GoogleCloudAuth(url=SERVICE_URL),
    model=MODEL_NAME,
    base_url=SERVICE_URL,
    num_predict=300,
)
```


```
llm.invoke("Hi, testing a request")
```

You can now use the `invoke` function as usual, with the token being refreshed automatically every hour.

## Cleaning up
To clean up all Google Cloud resources, you can run the following cell to delete the Cloud Run service you created.


```
# Delete the Cloud Run service deployed above

!gcloud run services delete $SERVICE_NAME --project $PROJECT_ID --region $LOCATION --quiet
```
