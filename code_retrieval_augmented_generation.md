```
# Copyright 2023 Google LLC
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

# Use Retrieval Augmented Generation (RAG) with Gemini API

<table align="left">

  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/code/code_retrieval_augmented_generation.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>

  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fuse-cases%2Fcode%2Fcode_retrieval_augmented_generation.ipynb">
      <img width="32px" src="https://cloud.google.com/ml-engine/images/colab-enterprise-logo-32px.png" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/code/code_retrieval_augmented_generation.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/use-cases/code/code_retrieval_augmented_generation.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Lavi Nigam](https://github.com/lavinigam-gcp), [Polong Lin](https://github.com/polong-lin) |

### Objective

This notebook demonstrates how you augment output from Gemini API by bringing in external knowledge. An example is provided using Code Retrieval Augmented Generation(RAG) pattern using [Google Cloud's Generative AI github repository](https://github.com/GoogleCloudPlatform/generative-ai) as external knowledge. The notebook uses [Gemini API in Vertex AI](https://ai.google.dev/gemini-api), [Embeddings for Text API](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings), FAISS vector store and [LangChain 🦜️🔗](https://python.langchain.com/en/latest/).

### Overview

Here is overview of what we'll go over.

Index Creation:

1. Recursively list the files(.ipynb) in github repo
2. Extract code and markdown from the files
3. Chunk & generate embeddings for each code strings and add initialize the vector store

Runtime:

4. User enters a prompt or asks a question as a prompt
5. Try zero-shot prompt
6. Run prompt using RAG Chain & compare results.To generate response we use **gemini-1.5-pro**

### Cost

This tutorial uses billable components of Google Cloud:

- Gemini API in Vertex AI offered by Google Cloud

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.

**Note:** We are using local vector store(FAISS) for this example however recommend managed highly scalable vector store for production usage such as [Vertex AI Vector Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview) or [AlloyDB for PostgreSQL](https://cloud.google.com/alloydb/docs/ai/work-with-embeddings) or [Cloud SQL for PostgreSQL](https://cloud.google.com/sql/docs/postgres/features)  using pgvector extension.

## Get started

### Install Vertex AI SDK for Python and other required packages



```
%pip install --upgrade --user -q google-cloud-aiplatform \
                                langchain \
                                langchain_google_vertexai \
                                langchain-community \
                                faiss-cpu \
                                nbformat
```

### Restart runtime (Colab only)

To use the newly installed packages, you must restart the runtime on Google Colab.


```
import sys

if "google.colab" in sys.modules:
    import IPython

    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)

Authenticate your environment on Google Colab.



```
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Import libraries


```
import time

from google.cloud import aiplatform
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# LangChain
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
import nbformat
import requests

# Vertex AI
import vertexai

# Print the version of Vertex AI SDK for Python
print(f"Vertex AI SDK version: {aiplatform.__version__}")
```

### Set Google Cloud project information and initialize Vertex AI SDK for Python

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com). Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
# Initialize project
# Define project information
PROJECT_ID = "YOUR_PROJECT_ID"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Code Generation
code_llm = VertexAI(
    model_name="gemini-1.5-pro",
    max_output_tokens=2048,
    temperature=0.1,
    verbose=False,
)
```

Next we need to create a GitHub personal token to be able to list all files in a repository.

- Follow [this link](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) to create GitHub token with repo->public_repo scope and update `GITHUB_TOKEN` variable below.


```
# provide GitHub personal access token
GITHUB_TOKEN = "YOUR_GITHUB_TOKEN"  # @param {type:"string"}
GITHUB_REPO = "GoogleCloudPlatform/generative-ai"  # @param {type:"string"}
```

# Index Creation

We use the Google Cloud Generative AI github repository as the data source. First list all Jupyter Notebook files in the repo and store it in a text file.

You can skip this step(#1) if you have executed it once and generated the output text file.

### 1. Recursively list the files(.ipynb) in the github repository


```
# Crawls a GitHub repository and returns a list of all ipynb files in the repository


def crawl_github_repo(url: str, is_sub_dir: bool, access_token: str = GITHUB_TOKEN):
    ignore_list = ["__init__.py"]

    if not is_sub_dir:
        api_url = f"https://api.github.com/repos/{url}/contents"

    else:
        api_url = url

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {access_token}",
    }

    response = requests.get(api_url, headers=headers)
    response.raise_for_status()  # Check for any request errors

    files = []

    contents = response.json()

    for item in contents:
        if (
            item["type"] == "file"
            and item["name"] not in ignore_list
            and (item["name"].endswith(".py") or item["name"].endswith(".ipynb"))
        ):
            files.append(item["html_url"])
        elif item["type"] == "dir" and not item["name"].startswith("."):
            sub_files = crawl_github_repo(item["url"], True)
            time.sleep(0.1)
            files.extend(sub_files)

    return files
```


```
code_files_urls = crawl_github_repo(GITHUB_REPO, False, GITHUB_TOKEN)

# Write list to a file so you do not have to download each time
with open("code_files_urls.txt", "w") as f:
    for item in code_files_urls:
        f.write(item + "\n")

len(code_files_urls)
```


```
code_files_urls[0:10]
```

### 2. Extract code from the Jupyter notebooks.

You could also include .py file, shell scripts etc.


```
# Extracts the python code from an ipynb file from github


def extract_python_code_from_ipynb(github_url, cell_type="code"):
    raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace(
        "/blob/", "/"
    )

    response = requests.get(raw_url)
    response.raise_for_status()  # Check for any request errors

    notebook_content = response.text

    notebook = nbformat.reads(notebook_content, as_version=nbformat.NO_CONVERT)

    python_code = None

    for cell in notebook.cells:
        if cell.cell_type == cell_type:
            if not python_code:
                python_code = cell.source
            else:
                python_code += "\n" + cell.source

    return python_code


def extract_python_code_from_py(github_url):
    raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace(
        "/blob/", "/"
    )

    response = requests.get(raw_url)
    response.raise_for_status()  # Check for any request errors

    python_code = response.text

    return python_code
```


```
with open("code_files_urls.txt") as f:
    code_files_urls = f.read().splitlines()
len(code_files_urls)
```


```
code_strings = []

for i in range(0, len(code_files_urls)):
    if code_files_urls[i].endswith(".ipynb"):
        content = extract_python_code_from_ipynb(code_files_urls[i], "code")
        doc = Document(
            page_content=content, metadata={"url": code_files_urls[i], "file_index": i}
        )
        code_strings.append(doc)
```

### 3. Chunk & generate embeddings for each code strings & initialize the vector store

We need to split code into usable chunks that the LLM can use for code generation. Therefore it's crucial to use the right chunking approach and chunk size.


```
# Utility functions for Embeddings API with rate limiting


def rate_limit(max_per_minute):
    period = 60 / max_per_minute
    print("Waiting")
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            print(".", end="")
            time.sleep(sleep_time)


class CustomVertexAIEmbeddings(VertexAIEmbeddings):
    requests_per_minute: int
    num_instances_per_batch: int
    model_name: str

    # Overriding embed_documents method
    def embed_documents(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch :],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]
```


```
# Chunk code strings
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)


texts = text_splitter.split_documents(code_strings)
print(len(texts))

# Initialize Embedding API
EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH = 5
embeddings = CustomVertexAIEmbeddings(
    requests_per_minute=EMBEDDING_QPM,
    num_instances_per_batch=EMBEDDING_NUM_BATCH,
    model_name="textembedding-gecko@latest",
)

# Create Index from embedded code chunks
db = FAISS.from_documents(texts, embeddings)

# Init your retriever.
retriever = db.as_retriever(
    search_type="similarity",  # Also test "similarity", "mmr"
    search_kwargs={"k": 5},
)

retriever
```

# Runtime
### 4. User enters a prompt or asks a question as a prompt


```
user_question = "Create a Python function that takes a prompt and predicts using langchain.llms interface with Vertex AI text-bison model"
```


```
# Define prompt templates

# Zero Shot prompt template
prompt_zero_shot = """
    You are a proficient python developer. Respond with the syntactically correct & concise code for to the question below.

    Question:
    {question}

    Output Code :
    """

prompt_prompt_zero_shot = PromptTemplate(
    input_variables=["question"],
    template=prompt_zero_shot,
)


# RAG template
prompt_RAG = """
    You are a proficient python developer. Respond with the syntactically correct code for to the question below. Make sure you follow these rules:
    1. Use context to understand the APIs and how to use it & apply.
    2. Do not add license information to the output code.
    3. Do not include Colab code in the output.
    4. Ensure all the requirements in the question are met.

    Question:
    {question}

    Context:
    {context}

    Helpful Response :
    """

prompt_RAG_template = PromptTemplate(
    template=prompt_RAG, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_llm(
    llm=code_llm,
    prompt=prompt_RAG_template,
    retriever=retriever,
    return_source_documents=True,
)
```

### 5. Try zero-shot prompt


```
response = code_llm.invoke(input=user_question, max_output_tokens=2048, temperature=0.1)
print(response)
```

### 6. Run prompt using RAG Chain & compare results
To generate response we use code-bison however can also use code-gecko and codechat-bison


```
results = qa_chain.invoke(input={"query": user_question})
print(results["result"])
```

### Let's try another prompt


```
user_question = "Create python function that takes text input and returns embeddings using LangChain with Vertex AI textembedding-gecko model"


response = code_llm.invoke(input=user_question, max_output_tokens=2048, temperature=0.1)
print(response)
```


```
results = qa_chain.invoke(input={"query": user_question})
print(results["result"])
```
