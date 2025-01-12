##### Copyright 2024 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Gemini API: Code analysis using LangChain and DeepLake

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/langchain/Code_analysis_using_Gemini_LangChain_and_DeepLake.ipynb"><img src = "https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>
  </td>
</table>

This notebook shows how to use Gemini API with [Langchain](https://python.langchain.com/v0.2/docs/introduction/) and [DeepLake](https://www.deeplake.ai/) for code analysis. The notebook will teach you:
- loading and splitting files
- creating a Deeplake database with embedding information
- setting up a retrieval QA chain

### Load dependencies


```
!pip install -q -U langchain-google-genai deeplake langchain langchain-text-splitters langchain-community
```

    [?25l     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/615.9 kB[0m [31m?[0m eta [36m-:--:--[0m[2K     [91m━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━[0m [32m317.4/615.9 kB[0m [31m9.7 MB/s[0m eta [36m0:00:01[0m[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m615.9/615.9 kB[0m [31m8.8 MB/s[0m eta [36m0:00:00[0m
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m50.4/50.4 kB[0m [31m1.2 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m17.1/17.1 MB[0m [31m78.4 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m37.0 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.3/2.3 MB[0m [31m60.8 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m395.9/395.9 kB[0m [31m23.8 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m76.9/76.9 kB[0m [31m5.0 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m139.2/139.2 kB[0m [31m9.1 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m150.6/150.6 kB[0m [31m9.6 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.5/4.5 MB[0m [31m76.3 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m47.7 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m82.1/82.1 kB[0m [31m5.9 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m12.3/12.3 MB[0m [31m80.1 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m116.3/116.3 kB[0m [31m7.7 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m76.4/76.4 kB[0m [31m4.7 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m77.9/77.9 kB[0m [31m845.4 kB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.3/49.3 kB[0m [31m3.0 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m134.8/134.8 kB[0m [31m10.9 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m141.9/141.9 kB[0m [31m8.5 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m56.8/56.8 kB[0m [31m4.0 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m82.7/82.7 kB[0m [31m6.4 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.3/58.3 kB[0m [31m4.1 MB/s[0m eta [36m0:00:00[0m
    [?25h  Building wheel for deeplake (pyproject.toml) ... [?25l[?25hdone
    


```
from glob import glob
from IPython.display import Markdown, display

from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
```

### Configure your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../../quickstarts/Authentication.ipynb) for an example.



```
import os
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```

## Prepare the files

First, download a [langchain-google](https://github.com/langchain-ai/langchain-google) repository. It is the repository you will analyze in this example.

It contains code integrating Gemini API, VertexAI, and other Google products with langchain.


```
!git clone https://github.com/langchain-ai/langchain-google
```

    Cloning into 'langchain-google'...
    remote: Enumerating objects: 3322, done.[K
    remote: Counting objects: 100% (1028/1028), done.[K
    remote: Compressing objects: 100% (353/353), done.[K
    remote: Total 3322 (delta 849), reused 736 (delta 675), pack-reused 2294 (from 1)[K
    Receiving objects: 100% (3322/3322), 1.78 MiB | 8.74 MiB/s, done.
    Resolving deltas: 100% (2266/2266), done.
    

This example will focus only on the integration of Gemini API with langchain and ignore the rest of the codebase.


```
repo_match = "langchain-google/libs/genai/langchain_google_genai**/*.py"
```

Each file with a matching path will be loaded and split by `RecursiveCharacterTextSplitter`.
In this example, it is specified, that the files are written in Python. It helps split the files without having documents that lack context.


```
docs = []
for file in glob(repo_match, recursive=True):
  loader = TextLoader(file, encoding='utf-8')
  splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=2000, chunk_overlap=0)
  docs.extend(loader.load_and_split(splitter))
```

`Language` Enum provides common separators used in most popular programming languages, it lowers the chances of classes or functions being split in the middle.


```
# common seperators used for Python files
RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
```




    ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']



## Create the database
The data will be loaded into the memory since the database doesn't need to be permanent in this case and is small enough to fit.

The type of storage used is specified by prefix in the path, in this case by `mem://`.

Check out other types of storage [here](https://docs.activeloop.ai/setup/storage-and-creds/storage-options).


```
# define path to database
dataset_path = 'mem://deeplake/langchain_google'
```


```
# define the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
```

Everything needed is ready, and now you can create the database. It should not take longer than a few seconds.


```
db = DeepLake.from_documents(docs, embeddings, dataset_path=dataset_path)
```

    Creating 97 embeddings in 1 batches of size 97:: 100%|██████████| 1/1 [00:02<00:00,  2.41s/it]

    Dataset(path='mem://deeplake/langchain_google', tensors=['text', 'metadata', 'embedding', 'id'])
    
      tensor      htype      shape     dtype  compression
      -------    -------    -------   -------  ------- 
       text       text      (97, 1)     str     None   
     metadata     json      (97, 1)     str     None   
     embedding  embedding  (97, 768)  float32   None   
        id        text      (97, 1)     str     None   
    

    
    

## Question Answering

Set-up the document retriever.


```
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 20 # number of documents to return
```


```
# define the chat model
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash-latest")
```

Now, you can create a chain for Question Answering. In this case, `RetrievalQA` chain will be used.

If you want to use the chat option instead, use `ConversationalRetrievalChain`.


```
qa = RetrievalQA.from_llm(llm, retriever=retriever)
```

The chain is ready to answer your questions.

NOTE: `Markdown` is used for improved formatting of the output.


```
# a helper function for calling retrival chain
def call_qa_chain(prompt):
  response = qa.invoke(prompt)
  display(Markdown(response["result"]))
```


```
call_qa_chain("Show hierarchy for _BaseGoogleGenerativeAI. Do not show content of classes.")
```


```
_BaseGoogleGenerativeAI
    - GoogleGenerativeAI
    - ChatGoogleGenerativeAI
```



```
call_qa_chain("What is the return type of embedding models.")
```


The return type of embedding models is a list of lists of floats. 

Here's a breakdown:

* **List[List[float]]**: This means the model returns a list of embedding vectors, where each vector is represented as a list of floats.

* **Each embedding vector**:  Represents a single piece of text (a document or a query) as a numerical representation. 

* **Floats**: Each element in the embedding vector is a floating-point number, capturing the semantic meaning of the text in a multi-dimensional space. 




```
call_qa_chain("What classes are related to Attributed Question and Answering.")
```


The following classes are related to Attributed Question and Answering (AQA) in the provided context:

* **`GenAIAqa`**: This is the main class representing Google's AQA service. It takes a user's query and a list of passages as input and returns a grounded response, meaning the response is backed by the provided passages.
* **`AqaInput`**: This class defines the input structure for the `GenAIAqa` class. It contains the user's `prompt` and a list of `source_passages` to be used by the AQA model.
* **`AqaOutput`**: This class defines the output structure for the `GenAIAqa` class. It contains the `answer` to the user's query, the `attributed_passages` used to generate the answer, and the `answerable_probability`, which indicates the likelihood that the question can be answered from the provided passages.
* **`_AqaModel`**: This is an internal wrapper class for Google's AQA model. It handles the communication with the Generative AI API and manages parameters like answer style, safety settings, and temperature.
* **`GoogleVectorStore`**: This class provides a way to store and search documents in Google's vector database. It can be used to retrieve relevant passages for AQA, either from an entire corpus or a specific document.
* **`Passage`**: This class represents a single passage of text. It includes the `text` itself and an optional `id`.
* **`GroundedAnswer`**: This dataclass represents a grounded answer, containing the `answer`, the `attributed_passages`, and the `answerable_probability`.

These classes work together to provide a comprehensive AQA solution, allowing users to ask questions and get answers that are grounded in relevant text.




```
call_qa_chain("What are the dependencies of the GenAIAqa class?")
```


The `GenAIAqa` class depends on the following:

* **`google.ai.generativelanguage`:** This is the Google Generative AI Python package, which provides the underlying API for interacting with Google's Generative AI services.
* **`langchain_core`:** This is the core LangChain library, which provides the framework for building and using language models and other components.
* **`_genai_extension`:** This is an internal module within the `langchain-google-genai` package that provides utility functions for interacting with the Google Generative AI API.

In addition to these direct dependencies, the `GenAIAqa` class also indirectly depends on other libraries such as `typing`, `langchain_core.pydantic_v1`, and `langchain_core.runnables`. 



## Summary

Gemini API works great with Langchain. The integration is seamless and provides an easy interface for:
- loading and splitting files
- creating DeepLake database with embeddings
- answering questions based on context from files

## What's next?

This notebook showed only one possible use case for langchain with Gemini API. You can find many more [here](../../examples/langchain) and in particular the one about [parsing large documents](../../examples/langchain/Gemini_LangChain_Summarization_WebLoad.ipynb).
