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

# Getting Started with DeepEval & Gemini API in Vertex AI

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/rag-evaluation/deepeval_with_gemini.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
    <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fuse-cases%2Fretrieval-augmented-generation%2Frag-evaluation%2Fdeepeval_with_gemini.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/rag-evaluation/deepeval_with_gemini.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/use-cases/retrieval-augmented-generation/rag-evaluation/deepeval_with_gemini.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>

| | |
|-|-|
| Author(s) | [Aditya Rane](https://github.com/Adi8885) |

## Overview

### [DeepEval](https://docs.confident-ai.com/docs/getting-started)

DeepEval is an open-source evaluation framework for LLMs. DeepEval makes it extremely easy to build and iterate on LLM (applications) and was built with the following principles in mind:

- Easily "unit test" LLM outputs in a similar way to Pytest.
- Plug-and-use 14+ LLM-evaluated metrics, most with research backing.
- Synthetic dataset generation with state-of-the-art evolution techniques.
- Metrics are simple to customize and covers all use cases.
- Real-time evaluations in production.

### Gemini

Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases. The Gemini API gives you access to the Gemini models.

### Gemini API in Vertex AI

The Gemini API in Vertex AI provides a unified interface for interacting with Gemini models. 

You can interact with the Gemini API using the following methods:

- Use the [Vertex AI Studio](https://cloud.google.com/generative-ai-studio) for quick testing and command generation
- Use cURL commands
- Use the Vertex AI SDK

This notebook focuses on using the **Gemini model with DeepEval**

For more information, see the [Generative AI on Vertex AI](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview) documentation.

### Objectives

In this notebook we will focus on using the Gemini API in Vertex AI with RAGAS
We will use the Gemini Pro (`gemini-1.5-pro`) model for Q&A evaluation.

You will complete the following tasks:

- Install the Vertex AI SDK for Python
- Use the Gemini API in Vertex AI to interact with each model
  - Gemini Pro (`gemini-1.5-pro`) model:
    - Q&A Generation
    - Evaluate Q&A performance with RAGAS

### Costs

This tutorial uses billable components of Google Cloud:

- Vertex AI

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.

## Getting Started

### Install Vertex AI SDK for Python


```
#This notebook was created and tested with below versions 
%pip install --user install deepeval==0.21.51 \
datasets==2.18.0 \
langchain==0.1.14 \
langchain-google-vertexai==1.0.5 \
langchain-chroma==0.1.1 \
chromadb==0.5.0 \
pypdf==4.2.0
```

### Restart current runtime

To use the newly installed packages in this Jupyter runtime, it is recommended to restart the runtime. Run the following cell to restart the current kernel.

The restart process might take a minute or so.


```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

After the restart is complete, continue to the next step.

<div class="alert alert-block alert-warning">
<b>⚠️ Wait for the kernel to finish restarting before you continue. ⚠️</b>
</div>

## Import libraries


```
import itertools

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric

# Base LLM for DeepEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# LangChain package for Vertex AI
from langchain_google_vertexai import (  # type: ignore[import-untyped]
    ChatVertexAI,
    HarmBlockThreshold,
    HarmCategory,
    VertexAIEmbeddings,
)
import vertexai
```


```
# TODO(developer): Update the below lines
PROJECT_ID = "<your_project>"
LOCATION = "<your_region>"

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

## Use Vertex AI models

The [Gemini-1.5-pro](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/overview) models are designed to handle multimodal inputs.


```
# Initialise safety filters for vertex model
safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

generation_config = {"temperature": 0.0, "topk": 1}

# Initialise the ChatVertexAI model
custom_chat_model_gemini = ChatVertexAI(
    model_name="gemini-1.5-pro",
    safety_settings=safety_settings,
    generation_config=generation_config,
    project=PROJECT_ID,
    location=LOCATION,
    response_validation=False,  # Important since deepeval cannot handle validation errors
)
```

The [Vertex AI Embeddings](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings) models are designed to convert text to dense vector representations


```
# Load Embeddings Models
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
```

## Create a local Vector DB
### Load the document


```
# source document
document_uri = "https://arxiv.org/pdf/1706.03762"
```


```
# use PyPDF loaded to read and chunk the input document
loader = PyPDFLoader(document_uri)
docs = loader.load_and_split()

# Verify if pages are loaded correctly
docs[0]
```

### Create local Vector DB


```
# Create an in-memory Vector DB using Chroma
vectordb = Chroma.from_documents(docs, embeddings)
```


```
# Set Vector DB as retriever
retriever = vectordb.as_retriever()
```

### Create Q&A Chain


```
# Create Q&A template for the Gemini Model
template = """Your task is to answer questions related to documents.
Use the following context to answer the question at the end.
{context}

Answers should be crisp.

Question: {question}
Helpful Answer:"""

# Create a prompt template for the q&a chain
PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
)

# Pass prompts to q&a chain
chain_type_kwargs = {"prompt": PROMPT}

# Retriever arguments
retriever.search_kwargs = {"k": 1}
```


```
# Setup a RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=custom_chat_model_gemini,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)
```


```
# Test the chain with a sample question
query = "Who are the authors of paper on Attention is all you need?"
result = qa({"query": query})
result
```

## Evaluation
### Create the evaluation set


```
# Evaluation set with questions and ground_truth
questions = [
    "What architecture is proposed in paper titled Attention is all you need?",
    "Where do primary authors of paper titled Attention is all you need work?",
]
ground_truth = ["Transformers architecture", "Google Brain"]
```

### Run the [Q&A chain](#create-qa-chain) on evaluation dataset 


```
contexts = []
answers = []

# Generate contexts and answers for each question
for query in questions:
    result = qa({"query": query})
    contexts.append(
        [document.page_content for document in result.get("source_documents")]
    )
    answers.append(result.get("result"))
```


```
# Convert into a dataset and prepare for consumption by DeepEval API
dataset = []
for q, a, c, g in itertools.zip_longest(questions, answers, contexts, ground_truth):
    dataset.append({"Question": q, "Answer": g, "Context": c})

# Inspect the dataset
dataset
```

## IMPORTANT : Gemini with DeepEval
> DeepEval is designed to work with OpenAI Models by default. We must write a Wrapper to make it work with Gemini


```
# Base LLM for DeepEval


class GoogleVertexAIDeepEval(DeepEvalBaseLLM):
    """Class to implement Vertex AI for DeepEval"""

    def __init__(self, model):  # pylint: disable=W0231
        self.model = model

    def load_model(self):  # pylint: disable=W0221
        return self.model

    def generate(self, prompt: str) -> str:  # pylint: disable=W0221
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:  # pylint: disable=W0221
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):  # pylint: disable=W0236 , W0221
        return "Vertex AI Model"
```


```
# Initialise the DeepEval wrapper class
google_vertexai_gemini_deepeval = GoogleVertexAIDeepEval(model=custom_chat_model_gemini)
```

### Run the DeepEval Evaluation


```
answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.5, model=google_vertexai_gemini_deepeval, async_mode=False
)
test_cases = []
for record in dataset:
    test_cases.append(
        LLMTestCase(
            input=record["Question"],
            actual_output=record["Answer"],
            retrieval_context=record["Context"],
        )
    )
```


```
# Evaluate test cases in bulk
evaluate(test_cases, [answer_relevancy_metric])
```


```
# measure single instance
answer_relevancy_metric.measure(test_cases[0])
```

### To use DeepEval with Pytest with 


```
%%writefile ./scripts/vertex_llm.py

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
"""Custom Class implementation"""

# Base LLM for DeepEval
from deepeval.models.base_model import DeepEvalBaseLLM

# LangChain package for Vertex AI
from langchain_google_vertexai import ChatVertexAI


class GoogleVertexAIDeepEval(DeepEvalBaseLLM):
    """Class to implement Vertex AI for DeepEval"""

    def __init__(self, model: ChatVertexAI) -> None:  # pylint: disable=W0231
        """Initialise the model"""
        self.model = model

    def load_model(self) -> ChatVertexAI:  # pylint: disable=W0221
        """Loads the model"""
        return self.model

    def generate(self, prompt: str) -> str:  # pylint: disable=W0221
        """Invokes the model"""
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:  # pylint: disable=W0221
        """Invokes the model async"""
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self) -> str:  # pylint: disable=W0236 , W0221
        """Returns the model name"""
        return "Vertex AI Model"

```


```
%%writefile ./scripts/test_chatbot.py

"""Test Script for DeepEval with Gemini"""

import itertools

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# LangChain package for Vertex AI
from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory
import pytest
from vertex_llm import GoogleVertexAIDeepEval  # pylint: disable=E0401

# TODO(developer): Update the below lines
PROJECT_ID = "<your_project>"
LOCATION = "<your_region>"

# Initialize safety filters for Gemini model
safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

# Initialize the ChatVertexAI model
custom_model_gemini = ChatVertexAI(
    model_name="gemini-1.5-pro",
    safety_settings=safety_settings,
    project=PROJECT_ID,
    location=LOCATION,
    response_validation=False,  # Important since deepeval cannot handle validation errors
)

# Initialize the DeepEval wrapper class
google_vertexai_gemini_deepeval = GoogleVertexAIDeepEval(model=custom_model_gemini)

# Evaluation set with questions and ground_truth
questions = [
    "What architecture is proposed in paper titled Attention is all you need?",
    "Where do primary authors of paper titled Attention is all you need work?",
]
ground_truth = ["Transformers architecture", "Google Brain"]

# Convert into a dataset and prepare for consumption by DeepEval API
test_set = []
for q, a in itertools.zip_longest(questions, ground_truth):
    test_set.append({"Question": q, "Answer": a, "Context": None})


@pytest.mark.parametrize("record", test_set)
def test_answer_relevancy(record: dict) -> None:
    """Function to test Answer relevancy"""
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5, model=google_vertexai_gemini_deepeval
    )
    test_case = LLMTestCase(
        input=record["Question"],
        actual_output=record["Answer"],
        retrieval_context=record["Context"],
    )
    assert_test(test_case, [answer_relevancy_metric])

```


```
# run the pytest scripts
!pytest scripts/
```

# Conclusion

In this notebook, you learned:

1. DeepEval - Framework for evaluation .
2. Making DeepEval Work with Gemini API in Vertex AI
3. Integrating DeepEval with Pytest
