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

# Gemini API: Summarize large documents using LangChain

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/langchain/Gemini_LangChain_Summarization_WebLoad.ipynb"><img src="../../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>


## Overview

The [Gemini](https://ai.google.dev/models/gemini) models are a family of generative AI models that allow developers generate content and solve problems. These models are designed and trained to handle both text and images as input.

[LangChain](https://www.langchain.com/) is a framework designed to make integration of Large Language Models (LLM) like Gemini easier for applications.

In this notebook, you'll learn how to create an application to summarize large documents using the Gemini API and LangChain.


## Setup

First, you must install the packages and set the necessary environment variables.

### Installation

Install LangChain's Python library, `langchain` and LangChain's integration package for the Gemini API, `langchain-google-genai`. Installing `langchain-community` allows you to use the `WebBaseLoader` tool shown later in this example.


```
!pip install --quiet langchain-core==0.1.23
!pip install --quiet langchain==0.1.1
!pip install --quiet langchain-google-genai==0.0.6
!pip install --quiet -U langchain-community==0.0.20
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m241.2/241.2 kB[0m [31m2.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m55.4/55.4 kB[0m [31m2.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.0/53.0 kB[0m [31m3.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m802.4/802.4 kB[0m [31m8.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m42.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.2/49.2 kB[0m [31m3.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m302.9/302.9 kB[0m [31m19.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m48.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m53.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m67.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.9/1.9 MB[0m [31m39.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.9/1.9 MB[0m [31m51.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.9/1.9 MB[0m [31m37.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.9/1.9 MB[0m [31m40.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.9/1.9 MB[0m [31m45.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m56.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m56.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m60.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m78.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m80.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m76.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m80.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m69.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m76.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m83.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m146.9/146.9 kB[0m [31m2.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m598.7/598.7 kB[0m [31m6.7 MB/s[0m eta [36m0:00:00[0m
    [?25h


```
from langchain import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
```

## Configure your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.



```
import os
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```

## Summarize text

In this tutorial, you are going to summarize the text from a website using the Gemini model integrated through LangChain.

You'll perform the following steps to achieve the same:
1. Read and parse the website data using LangChain.
2. Chain together the following:
    * A prompt for extracting the required input data from the parsed website data.
    * A prompt for summarizing the text using LangChain.
    * An LLM model (such as the Gemini model) for prompting.

3. Run the created chain to prompt the model for the summary of the website data.

### Read and parse the website data

LangChain provides a wide variety of document loaders. To read the website data as a document, you will use the `WebBaseLoader` from LangChain.

To know more about how to read and parse input data from different sources using the document loaders of LangChain, read LangChain's [document loaders guide](https://python.langchain.com/docs/integrations/document_loaders).


```
loader = WebBaseLoader("https://blog.google/technology/ai/google-gemini-ai/#sundar-note")
docs = loader.load()
```

### Initialize the Gemini model

You must import the `ChatGoogleGenerativeAI` LLM from LangChain to initialize your model.

In this example you will use Gemini 1.5 Flash, (`gemini-1.5-flash-latest`), as it supports text summarization. To know more about this model and the other models availabe, read Google AI's [language documentation](https://ai.google.dev/models/gemini).

You can configure the model parameters such as `temperature` or `top_p`,  by passing the appropriate values when creating the `ChatGoogleGenerativeAI` LLM.  To learn more about the parameters and their uses, read Google AI's [concepts guide](https://ai.google.dev/docs/concepts#model_parameters).


```
from langchain_google_genai import ChatGoogleGenerativeAI

# To configure model parameters use the `generation_config` parameter.
# eg. generation_config = {"temperature": 0.7, "topP": 0.8, "topK": 40}
# If you only want to set a custom temperature for the model use the
# "temperature" parameter directly.

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
```

### Create prompt templates

You'll use LangChain's [`PromptTemplate`](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/) to generate prompts for summarizing the text.

To summarize the text from the website, you will need the following prompts.
1. Prompt to extract the data from the output of `WebBaseLoader`, named `doc_prompt`
2. Prompt for the Gemini model to summarize the extracted text, named `llm_prompt`.

In the `llm_prompt`, the variable `text` will be replaced later by the text from the website.


```
# To extract data from WebBaseLoader
doc_prompt = PromptTemplate.from_template("{page_content}")

# To query Gemini
llm_prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
llm_prompt = PromptTemplate.from_template(llm_prompt_template)

print(llm_prompt)
```

    input_variables=['text'] template='Write a concise summary of the following:\n"{text}"\nCONCISE SUMMARY:'
    

### Create a Stuff documents chain

LangChain provides [Chains](https://python.langchain.com/docs/modules/chains/) for chaining together LLMs with each other or other components for complex applications. You will create a **Stuff documents chain** for this application. A **Stuff documents chain** lets you combine all the documents, insert them into the prompt and pass that prompt to the LLM.

You can create a Stuff documents chain using the [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language).

To learn more about different types of document chains, read LangChain's [chains guide](https://python.langchain.com/docs/modules/chains/document/).


```
# Create Stuff documents chain using LCEL.
# This is called a chain because you are chaining
# together different elements with the LLM.
# In the following example, to create stuff chain,
# you will combine content, prompt, LLM model and
# output parser together like a chain using LCEL.
#
# The chain implements the following pipeline:
# 1. Extract data from documents and save to variable `text`.
# 2. This `text` is then passed to the prompt and input variable
#    in prompt is populated.
# 3. The prompt is then passed to the LLM (Gemini).
# 4. Output from the LLM is passed through an output parser
#    to structure the model response.

stuff_chain = (
    # Extract data from the documents and add to the key `text`.
    {
        "text": lambda docs: "\n\n".join(
            format_document(doc, doc_prompt) for doc in docs
        )
    }
    | llm_prompt         # Prompt for Gemini
    | llm                # Gemini API function
    | StrOutputParser()  # output parser
)
```

### Prompt the model

To generate the summary of the  the website data, pass the documents extracted using the `WebBaseLoader` (`docs`) to `invoke()`.


```
stuff_chain.invoke(docs)
```




    "Google has introduced Gemini, its most capable AI model yet. Gemini is multimodal, meaning it can understand and interact with various forms of information, including text, code, audio, images, and video. It comes in three sizes: Ultra (for complex tasks), Pro (for a wide range of tasks), and Nano (for on-device tasks). Gemini surpasses existing models in performance benchmarks across various domains, including natural language understanding, reasoning, and coding. \n\nGoogle emphasizes Gemini's safety and responsibility features, including comprehensive bias and toxicity evaluation, adversarial testing, and collaboration with external experts.  \n\nGemini is being integrated into various Google products, such as Bard, Pixel, Search, and Ads, and will be available to developers through APIs. \n\nThe release of Gemini marks a significant milestone in AI development, opening up new possibilities for innovation and enhancing human capabilities in various areas. \n"



# Conclusion

That's it. You have successfully created an LLM application to summarize text using LangChain and the Gemini API.
