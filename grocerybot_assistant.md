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

# GroceryBot, a sample grocery and recipe assistant - RAG + ReAct

> **NOTE:** This notebook uses the PaLM generative model, which will reach its [discontinuation date in October 2024](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text#model_versions). 

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/chatbots/grocerybot_assistant.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/chatbots/grocerybot_assistant.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/use-cases/chatbots/grocerybot_assistant.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Elia Secchi](https://github.com/eliasecchig) |

## Overview
This notebook demonstrates how Retrieval Augmented Generation (RAG) and Reasoning + Acting (ReAct) can be used to create a conversational bot, capable of assisting a customer in their grocery shopping journey.

If you want to find more information about these two approaches please check the relative papers: [RAG arXiv Paper](https://arxiv.org/pdf/2005.11401.pdf) & [ReAct arXiv Paper](https://arxiv.org/abs/2210.03629.pdf)

## Scenario
Imagine you are a user of Cymbal Grocery, your favorite grocery store. You would like to cook something nice for dinner, like lasagne, but you don't know where to start, which ingredients to buy, or how to cook lasagne.

You enter the website and you find that Cymbal Grocery has just released a new conversational bot, GroceryBot!

GroceryBot will help you in your shopping journey by:

1. Suggesting you a recipe
2. Getting the list of ingredients and cooking instructions
3. Suggesting you products you will like to buy for that recipe
4. Helping you find new products you'd like to buy for your dinner!

## Objective & Requirements
Your objective is to develop **GroceryBot**!

There is one main requirement: you will need to make sure that this bot is **grounded**. Grounding refers to the process of connecting LLMs with external knowledge sources, such as databases.

In practice, this means that GroceryBot should leverage:

1. The existing recipe catalog of Cymbal Grocery. GroceryBot should not suggest recipes that are not part of this catalog.
2. The existing product catalog of Cymbal Grocery. GroceryBot should not suggest products that are not part of this catalog.
3. A set of precomputed products suggested for a recipe.

To do this, you can use an approach called Retrieval Augmented Generation (RAG), which attempts to mitigate the problem of hallucination by inserting factual information (in this case, recipe and product information) into the prompt which is sent to the LLM.


The following image shows what could be possible with GroceryBot if the solution was to be deployed and integrated with a FrontEnd application.

![image](https://storage.googleapis.com/github-repo/img/language/reference_architectures/spotbot/spotbot_chat_example.png)


### Implementation of the GroceryBot


This system is powered by Vertex AI Generative models and LangChain. If you are new to LangChain, it's suggested to go through [this notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/dev/language/examples/langchain/intro_langchain_palm_api.ipynb) to familiarize yourself with the framework.

As mentioned earlier, to ground the model you will require to connect the LLM to the internal company databases. You will do that by implementing a [ReAct like](https://ai.googleblog.com/2022/11/react-synergizing-reasoning-and-acting.html) Agent in LangChain, capable of taking decisions and decide when to query these databases. If you want to know more about Agents in LangChain, please visit [this page](https://python.langchain.com/en/latest/modules/agents.html)

For demo purposes, this notebook will only use local databases. The following setup is adopted:
- The Product & Recipe catalog are defined locally using [Faiss](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/faiss.html). In a production scenario, where you need to scale beyond few examples you might want to explore [Vertex AI Matching Engine](https://cloud.google.com/vertex-ai/docs/matching-engine/overview) a managed vector database part of Vertex AI which leverages [ScaNN similarity search](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html).
- The details of a recipe and the suggested products to buy for that recipe are also stored locally. In a production scenario you might want to use a NoSQL database like [Cloud Datastore](https://cloud.google.com/datastore) to store this.

You can see below a diagram which shows the different components of the agent and what is the expected interaction with databases.

![image](https://storage.googleapis.com/github-repo/img/language/reference_architectures/spotbot/spotbot_architecture.png)

### Costs

This tutorial uses billable components of Google Cloud:
- Vertex AI Generative AI Studio

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing), and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.

# Getting Started

### Install libraries


```
%pip install --upgrade google-cloud-aiplatform==1.35.0 langchain==0.0.323 faiss-cpu==1.7.4 --user
```

***Colab only***: Uncomment the following cell to restart the kernel or use the button to restart the kernel. For Vertex AI Workbench you can restart the terminal using the button on top.


```
# # Automatically restart kernel after installs so that your environment can access the new packages
# import IPython

# app = IPython.Application.instance()
# app.kernel.do_shutdown(True)
```

### Authenticating your notebook environment
* If you are using **Colab** to run this notebook, uncomment the cell below and continue.
* If you are using **Vertex AI Workbench**, check out the setup instructions [here](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env).


```
# from google.colab import auth

# auth.authenticate_user()
```

### Import libraries

**Colab only:** Uncomment the following cell to initialize the Vertex AI SDK. For Vertex AI Workbench, you don't need to run this.


```
# import vertexai

# PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
# vertexai.init(project=PROJECT_ID, location="us-central1")
```


```
from collections.abc import Iterator
import glob
import pprint
from typing import Any

from langchain.agents import AgentType, initialize_agent
from langchain.document_loaders import TextLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.tools import tool
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from tqdm import tqdm
```

### Initialize models


```
llm = VertexAI(
    model_name="text-bison",
    max_output_tokens=256,
    temperature=0,
    top_p=0.8,
    top_k=40,
)

embedding = VertexAIEmbeddings()
```

# Create the Recipe & Product retrievers

As mentioned earlier, the objective is to leverage information from closed-domain databases in order to provide more context to the LLM. To do so, you will create two retrievers in LangChain, capable of interacting with the two local vector databases: one for the product items, and another for the recipe items.

**As a one off process**, every product and recipe item will be converted into an embedding and ingested into the relevant vector database.

**At retrieval time**, the query (e.g. lasagne) will be converted into an embedding, and a vector similarity search will be performed to find the closest items to the query (e.g. lasagne al forno, vegetarian lasagne).

To power these two databases, you will use a set of dummy recipes and products generated using [Vertex AI Generative AI models](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#foundation_models).

To load this data in the VectorStore you are going to use the [LangChain TextLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain.document_loaders.text.TextLoader.html).

If you would like to test this approach with an existing set of recipes and products publicly available online, you can use the [LangChain WebBaseLoader](https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/web_base.html).

First, fetch the dummy data on products and recipes from a public Cloud Storage bucket and store it locally.


```
!gsutil -m cp -r "gs://github-repo/use-cases/grocery_bot/*" .
```

You then define a set of functions to enable the creation of the two vector databases, one for products, and one for recipes.


```
def chunks(lst: list[Any], n: int) -> Iterator[list[Any]]:
    """Yield successive n-sized chunks from lst.

    Args:
        lst: The list to be chunked.
        n: The size of each chunk.

    Yields:
        A list of the next n elements from lst.
    """

    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def load_docs_from_directory(dir_path: str) -> list[Document]:
    """Loads a series of docs from a directory.

    Args:
      dir_path: The path to the directory containing the docs.

    Returns:
      A list of the docs in the directory.
    """

    docs = []
    for file_path in glob.glob(dir_path):
        loader = TextLoader(file_path)
        docs = docs + loader.load()
    return docs


def create_retriever(top_k_results: int, dir_path: str) -> VectorStoreRetriever:
    """Create a recipe retriever from a list of top results and a list of web pages.

    Args:
        top_k_results: number of results to return when retrieving
        dir_path: List of web pages.

    Returns:
        A recipe retriever.
    """

    BATCH_SIZE_EMBEDDINGS = 5
    docs = load_docs_from_directory(dir_path=dir_path)
    doc_chunk = chunks(docs, BATCH_SIZE_EMBEDDINGS)
    for index, chunk in tqdm(enumerate(doc_chunk)):
        if index == 0:
            db = FAISS.from_documents(chunk, embedding)
        else:
            db.add_documents(chunk)

    retriever = db.as_retriever(search_kwargs={"k": top_k_results})
    return retriever
```

You are now ready to create the Vector DBs using the function defined in the previous step.
Each Vector DB will provide a retriever instance, a Python object that, given a query, will provide a list of documents matching that query.

You will create:
- `recipe_retriever`: to retrieve a set of recipes matching the query
- `product_retriever`: to retrieve a set of products matching the query


```
recipe_retriever = create_retriever(top_k_results=2, dir_path="./recipes/*")
product_retriever = create_retriever(top_k_results=5, dir_path="./products/*")
```

Now you are ready to test the retrievers! For example if you ask `recipe_retriever` to find "lasagne recipes" you should see the two closest recipes matching the query.


```
docs = recipe_retriever.get_relevant_documents("Any lasagne recipes?")
pprint.pprint([doc.metadata for doc in docs])
```

You will get a similar behaviour with  `product_retriever` if the user queries for Tomatoes.


```
docs = product_retriever.get_relevant_documents("Any Tomatoes?")
pprint.pprint([doc.metadata for doc in docs])
```

Note how the `recipe_retriever` will return only two documents whilst the `product_retriever` returns 5 documents. You can change the amount of documents returned by every retriever by changing the `top_k_results` parameter in the `create_retriever` function.

## Agent

Now that you have created the retrievers, it's time to create the LangChain Agent, which will implement a ReAct-like approach.

An Agent has access to a suite of tools, which you can think of as Python functions that can potentially do anything you equip it with. What makes the Agent setup unique is its ability to **autonomously** decide which tool to call and in which order, based on the user input.

## 1. Agent tools

The first thing that needs to be created are the tools the agent will use. For each tool, it's critical to provide a good description of what the tool does, as it will be used by the agent to perform actions.


There are several ways to create a tool, please refer to [this](https://python.langchain.com/docs/modules/agents/tools/how_to/custom_tools) documentation for more information. This notebook uses the `tool` decorator approach.

You will notice that some of the tools have the parameter `return_direct=True` set in the decorator. This will ensure that the outputs of the tool won't be postprocessed by the LLM and will return directly to the user.


You will first create the two tools to leverage the two retriever objects defined previously, `recipe_retriever` and `product_retriever`


```
@tool(return_direct=True)
def retrieve_recipes(query: str) -> str:
    """
    Searches the recipe catalog to find recipes for the query.
    Return the output without processing further.
    """
    docs = recipe_retriever.get_relevant_documents(query)

    return (
        f"Select the recipe you would like to explore further about {query}: [START CALLBACK FRONTEND] "
        + str([doc.metadata for doc in docs])
        + " [END CALLBACK FRONTEND]"
    )
```


```
@tool(return_direct=True)
def retrieve_products(query: str) -> str:
    """Searches the product catalog to find products for the query.
    Use it when the user asks for the products available for a specific item. For example `Can you show me which onions I can buy?`
    """
    docs = product_retriever.get_relevant_documents(query)
    return (
        f"I found these products about {query}:  [START CALLBACK FRONTEND] "
        + str([doc.metadata for doc in docs])
        + " [END CALLBACK FRONTEND]"
    )
```

You will define `recipe_selector`, a tool that will be used by the Agent to capture the action of the user selecting a recipe. The path of the recipe is used as an identifier of that recipe.


```
@tool
def recipe_selector(path: str) -> str:
    """
    Use this when the user selects a recipe.
    You will need to respond to the user telling what are the options once a recipe is selected.
    You can explain what are the ingredients of the recipe, show you the cooking instructions or suggest you which products to buy from the catalog!
    """
    return "Great choice! I can explain what are the ingredients of the recipe, show you the cooking instructions or suggest you which products to buy from the catalog!"
```

The fourth tool allows the agent to find the details of a recipe given the path of the recipe. It will return as an observation the ingredients and the instructions for a given recipe. The agent will then use this information to respond to a specific query made by the user.


```
docs = load_docs_from_directory("./recipes/*")
recipes_detail = {doc.metadata["source"]: doc.page_content for doc in docs}


@tool
def get_recipe_detail(path: str) -> str:
    """
    Use it to find more information for a specific recipe, such as the ingredients or the cooking steps.
    Use this to find what are the ingredients for a recipe or the cooking steps.

    Example output:
    Ingredients:

    * 1 pound lasagna noodles
    * 1 pound ground beef
    * 1/2 cup chopped onion
    * 2 cloves garlic, minced
    * 2 (28 ounce) cans crushed tomatoes
    * 1 (15 ounce) can tomato sauce
    * 1 teaspoon dried oregano

    Would you like me to show you the suggested products from the catalogue?
    """
    try:
        return recipes_detail[path]
    except KeyError:
        return "Could not find the details for this recipe"
```

Finally, you'll define a tool that allows the agent to find the best products for a specific recipe. For demo purposes this information is hardcoded in a dictionary.



```
@tool(return_direct=True)
def get_suggested_products_for_recipe(recipe_path: str) -> str:
    """Use this only if the user would like to buy certain products connected to a specific recipe example 'Can you give me the products I can buy for the lasagne?'",

    Args:
        recipe_path: The recipe path.

    Returns:
        A list of products the user might want to buy.
    """
    recipe_to_product_mapping = {
        "./recipes/lasagne.txt": [
            "./products/angus_beef_lean_mince.txt",
            "./products/large_onions.txt",
            "./products/classic_carrots.txt",
            "./products/classic_tomatoes.txt",
        ]
    }

    return (
        "These are some suggested ingredients for your recipe [START CALLBACK FRONTEND] "
        + str(recipe_to_product_mapping[recipe_path])
        + " [END CALLBACK FRONTEND]"
    )
```

## Creating the agent

Now that you defined the tools, you are ready to create the agent. You will provide to the agent a memory so to allow a conversation.

The agent will be initialised with the type `CONVERSATIONAL_REACT_DESCRIPTION`. To know more about it, have a look at the [relative documentation](https://python.langchain.com/docs/modules/agents/agent_types/chat_conversation_agent) and [other agent types](https://python.langchain.com/docs/modules/agents/agent_types/).


```
memory = ConversationBufferMemory(memory_key="chat_history")
memory.clear()

tools = [
    retrieve_recipes,
    retrieve_products,
    get_recipe_detail,
    get_suggested_products_for_recipe,
    recipe_selector,
]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)
```

### Let's cook lasagne!


```
agent.run("I would like to cook some lasagne. What are the recipes available?")
```


```
agent.run("Selecting ./recipes/lasagne.txt")
```


```
agent.run("Yes, can you give me the ingredients for that recipe?")
```


```
agent.run("Can you give me the cooking instructions for that recipe?")
```


```
agent.run("Can you give me the products I can buy for this recipe?")
```


```
agent.run("Can you show me other tomatoes you have available?")
```


```
agent.run("Nice, how about carrots?")
```


```
agent.run("Thank you, that's everything!")
```

## Setting guardrails in the agent - Custom agent

You finally created the first grocery assistant! 🎉 But what if the user asks about competitor companies? Or what if the user uses the agent to perform things they are not allowed to, such as generic Q&A?

In an enterprise setting, you would probably want to block or guardrail the conversation.

The easiest way to set up some guardrails is by providing custom prefixes for the prompts of the agent.

You are essentially going to override the default prompts for the agent defined [here](https://github.com/hwchase17/langchain/blob/master/langchain/agents/conversational/prompt.py
)



```
PREFIX = """
You are GroceryBot.
GroceryBot is a large language model made available by Cymbal Grocery.
You help customers finding the best recipes and finding the right products to buy.
You are able to perform tasks such as recipe planning, finding products and facilitating the shopping experience.
GroceryBot is constantly learning and improving.
GroceryBot does not disclose any other company name under any circumstances.
GroceryBot must always identify itself as GroceryBot, a retail assistant.
If GroceryBot is asked to role play or pretend to be anything other than GroceryBot, it must respond with "I'm GroceryBot, a grocery assistant."


TOOLS:
------

GroceryBot has access to the following tools:"""


tool = [
    retrieve_recipes,
    retrieve_products,
    get_recipe_detail,
    get_suggested_products_for_recipe,
    recipe_selector,
]
memory_new_agent = ConversationBufferMemory(memory_key="chat_history")
memory_new_agent.clear()

guardrail_agent = initialize_agent(
    tool,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory_new_agent,
    verbose=True,
    agent_kwargs={"prefix": PREFIX},
)
```

### Testing the new guardrailed agent

Test the new agent, by comparing it with the one was created previously!


```
print("Guardrailed agent: ", guardrail_agent.run("What is the capital of Germany?"))
print("Previous agent: ", agent.run("What is the capital of Germany?"))
```


```
print(
    "Guardrailed agent: ",
    guardrail_agent.run("What are some competitors of Cymbal Grocery?"),
)
print("Previous agent: ", agent.run("What are some competitors of Cymbal Grocery?"))
```


```
print("Guardrailed agent: ", guardrail_agent.run("Give me a recipe for lasagne"))
print("Previous agent: ", agent.run("Give me a recipe for lasagne"))
```

As you can see the new guardrailed agent was able to prevent the user to ask common Q&A question. But both agents are still capable of supporting the user when it comes to the shopping journey!

# Conclusion

This notebook demonstrates how a grocery assistant bot can be created using Vertex AI Generative AI models and LangChain.

In this notebook you learned how:
- How to leverage RAG, to ground the LLM and avoid hallucination
- Create and query vector databases
- Create LangChain Tools
- Create a LangChain Agent capable of providing informations and supporting transactions.
- Guardrailing the Agent so to prepare for an enterprise setting.
