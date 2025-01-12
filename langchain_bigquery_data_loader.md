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

# How to use the LangChain 🦜️🔗 BigQuery Data Loader

> **NOTE:** This notebook uses the PaLM generative model, which will reach its [discontinuation date in October 2024](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text#model_versions). 

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/language/orchestration/langchain/langchain_bigquery_data_loader.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Flanguage%2Forchestration%2Flangchain%2Flangchain_bigquery_data_loader.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/orchestration/langchain/langchain_bigquery_data_loader.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/bigquery/import?url=https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/orchestration/langchain/langchain_bigquery_data_loader.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/bigquery/v1/32px.svg" alt="BigQuery Studio logo"><br> Open in BigQuery Studio
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/orchestration/langchain/langchain_bigquery_data_loader.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Karl Weinmeister](https://github.com/kweinmeister) |

## Objectives

This notebook provides an introductory understanding of how to use [LangChain](https://langchain.com/) and LangChain's [BigQuery Data Loader](https://python.langchain.com/docs/integrations/document_loaders/google_bigquery). The notebook covers 3 steps:

1. Querying the Vertex AI LLM with LangChain
1. Using the LangChain BigQuery Data Loader
1. Developing a chain that uses the data loader

### Costs

This tutorial uses billable components of Google Cloud:

- Vertex AI
- BigQuery

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing), [BigQuery pricing](https://cloud.google.com/bigquery/pricing),
and use the [Pricing Calculator](https://cloud.google.com/products/calculator/)
to generate a cost estimate based on your projected usage.



```
# Install Vertex AI LLM SDK, BigQuery library, and langchain
%pip install google-cloud-aiplatform google-cloud-bigquery langchain==0.0.332 --upgrade --user
```

**Colab only:** Uncomment the following cell to restart the kernel or use the button to restart the kernel. For Vertex AI Workbench you can restart the terminal using the button on top.



```
# # Automatically restart kernel after installs so that your environment can access the new packages
# import IPython

# app = IPython.Application.instance()
# app.kernel.do_shutdown(True)
```

### Authenticating your notebook environment

- If you are using **Colab** to run this notebook, uncomment the cell below and continue.
- If you are using **Vertex AI Workbench**, check out the setup instructions [here](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env).



```
# from google.colab import auth as google_auth
# google_auth.authenticate_user()
```

### Import libraries


**Colab only:** Uncomment the following cell to initialize the Vertex AI SDK. For Vertex AI Workbench, you don't need to run this.



```
# import vertexai

# PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
# vertexai.init(project=PROJECT_ID, location="us-central1")
```


```
from google.cloud import aiplatform
import google.cloud.bigquery as bq
import langchain
from langchain.document_loaders import BigQueryLoader
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.schema import format_document

# Print LangChain and Vertex AI versions
print(f"LangChain version: {langchain.__version__}")
print(f"Vertex AI SDK version: {aiplatform.__version__}")
```

## Using Vertex AI foundation models with LangChain

Let's start from the beginning, and learn a bit about BigQuery along the way. We'll define a LangChain LLM model. We'll use the [text foundation model](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text) and use a `temperature` setting of 0 for consistent results.


```
llm = VertexAI(model_name="text-bison", temperature=0)

llm("What's BigQuery?")
```

## Using the Data Loader

Let's now learn how to use the document loader. We'll use data from a fictional eCommerce clothing site called [TheLook](https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce), available as a BigQuery public dataset.

Our first goal is to understand the tables in a dataset. Let's query the schema from this dataset to extract the data definition language (DDL). DDL is used to create and modify tables, and can tell us about each column and its type.

Our query is extracting the table name and DDL for each of the tables. We then create a [data loader](https://api.python.langchain.com/en/latest/document_loaders/langchain.document_loaders.bigquery.BigQueryLoader.html), specifying that the table name is a metadata column and the DDL is the content. 



```
# Define our query
query = """
SELECT table_name, ddl
FROM `bigquery-public-data.thelook_ecommerce.INFORMATION_SCHEMA.TABLES`
WHERE table_type = 'BASE TABLE'
ORDER BY table_name;
"""

# Load the data
loader = BigQueryLoader(
    query, metadata_columns=["table_name"], page_content_columns=["ddl"]
)
data = loader.load()
```

## Writing our first chain

Now that we've loaded the documents, let's put them to work.

Our goal is to understand which customers we want to target for an upcoming marketing campaign in Japan. We'll use the [code generation model](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/code-generation) to help us create a query.

We will create a basic chain that "[stuffs](https://python.langchain.com/docs/use_cases/summarization#option-1-stuff)" together all of the table metadata into one prompt. For larger datasets with many more tables, a more sophisticated chaining approach will be needed. That's because there's a limited length to each prompt, i.e. a context window.

For example, you could compress highlights from each individual table's content into smaller documents, and then summarize those using a [map-reduce](https://python.langchain.com/docs/use_cases/summarization#option-2-map-reduce) method. Or, you could iterate over each table, [refining](https://python.langchain.com/docs/use_cases/summarization#option-3-refine) your query as you go.

Here's how to do it. We'll use the [LangChain Expression Language](https://python.langchain.com/docs/expression_language/) (LCEL) to define the chain with 3 steps:

1. We'll combine the page_content from each document (remember, that's the DDL of each table) into a string called content.
1. Create a prompt to find our most valuable customers, passing in content, the combined set of table metadata .
1. Pass the prompt to the LLM.



```
# Use code generation model
llm = VertexAI(model_name="code-bison@latest", max_output_tokens=2048)

# Define the chain
chain = (
    {
        "content": lambda docs: "\n\n".join(
            format_document(doc, PromptTemplate.from_template("{page_content}"))
            for doc in docs
        )
    }
    | PromptTemplate.from_template(
        "Suggest a GoogleSQL query that will help me identify my most valuable customers located in Japan:\n\n{content}"
    )
    | llm
)

# Invoke the chain with the documents, and remove code backticks
result = chain.invoke(data).strip("```")
print(result)
```

Let's now try out our query, and see what it returns!


```
client = bq.Client()
client.query(result).result().to_dataframe()
```

Congratulations, you've now seen how to integrate your BigQuery data into LLM solutions! 🎉
