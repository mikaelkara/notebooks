# MyScale

>[MyScale](https://docs.myscale.com/en/) is an integrated vector database. You can access your database in SQL and also from here, LangChain.
>`MyScale` can make use of [various data types and functions for filters](https://blog.myscale.com/2023/06/06/why-integrated-database-solution-can-boost-your-llm-apps/#filter-on-anything-without-constraints). It will boost up your LLM app no matter if you are scaling up your data or expand your system to broader application.

In the notebook, we'll demo the `SelfQueryRetriever` wrapped around a `MyScale` vector store with some extra pieces we contributed to LangChain. 

In short, it can be condensed into 4 points:
1. Add `contain` comparator to match the list of any if there is more than one element matched
2. Add `timestamp` data type for datetime match (ISO-format, or YYYY-MM-DD)
3. Add `like` comparator for string pattern search
4. Add arbitrary function capability

## Creating a MyScale vector store
MyScale has already been integrated to LangChain for a while. So you can follow [this notebook](/docs/integrations/vectorstores/myscale) to create your own vectorstore for a self-query retriever.

**Note:** All self-query retrievers requires you to have `lark` installed (`pip install lark`). We use `lark` for grammar definition. Before you proceed to the next step, we also want to remind you that `clickhouse-connect` is also needed to interact with your MyScale backend.


```python
%pip install --upgrade --quiet  lark clickhouse-connect
```

In this tutorial we follow other example's setting and use `OpenAIEmbeddings`. Remember to get an OpenAI API Key for valid access to LLMs.


```python
import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
if "MYSCALE_HOST" not in os.environ:
    os.environ["MYSCALE_HOST"] = getpass.getpass("MyScale URL:")
if "MYSCALE_PORT" not in os.environ:
    os.environ["MYSCALE_PORT"] = getpass.getpass("MyScale Port:")
if "MYSCALE_USERNAME" not in os.environ:
    os.environ["MYSCALE_USERNAME"] = getpass.getpass("MyScale Username:")
if "MYSCALE_PASSWORD" not in os.environ:
    os.environ["MYSCALE_PASSWORD"] = getpass.getpass("MyScale Password:")
```


```python
from langchain_community.vectorstores import MyScale
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
```

## Create some sample data
As you can see, the data we created has some differences compared to other self-query retrievers. We replaced the keyword `year` with `date` which gives you finer control on timestamps. We also changed the type of the keyword `gerne` to a list of strings, where an LLM can use a new `contain` comparator to construct filters. We also provide the `like` comparator and arbitrary function support to filters, which will be introduced in next few cells.

Now let's look at the data first.


```python
docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"date": "1993-07-02", "rating": 7.7, "genre": ["science fiction"]},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"date": "2010-12-30", "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"date": "2006-04-23", "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"date": "2019-08-22", "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"date": "1995-02-11", "genre": ["animated"]},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "date": "1979-09-10",
            "director": "Andrei Tarkovsky",
            "genre": ["science fiction", "adventure"],
            "rating": 9.9,
        },
    ),
]
vectorstore = MyScale.from_documents(
    docs,
    embeddings,
)
```

## Creating our self-querying retriever
Just like other retrievers... simple and nice.


```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genres of the movie. "
        "It only supports equal and contain comparisons. "
        "Here are some examples: genre = [' A '], genre = [' A ', 'B'], contain (genre, 'A')",
        type="list[string]",
    ),
    # If you want to include length of a list, just define it as a new column
    # This will teach the LLM to use it as a column when constructing filter.
    AttributeInfo(
        name="length(genre)",
        description="The length of genres of the movie",
        type="integer",
    ),
    # Now you can define a column as timestamp. By simply set the type to timestamp.
    AttributeInfo(
        name="date",
        description="The date the movie was released",
        type="timestamp",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]
document_content_description = "Brief summary of a movie"
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)
```

## Testing it out with self-query retriever's existing functionalities
And now we can try actually using our retriever!


```python
# This example only specifies a relevant query
retriever.invoke("What are some movies about dinosaurs")
```


```python
# This example only specifies a filter
retriever.invoke("I want to watch a movie rated higher than 8.5")
```


```python
# This example specifies a query and a filter
retriever.invoke("Has Greta Gerwig directed any movies about women")
```


```python
# This example specifies a composite filter
retriever.invoke("What's a highly rated (above 8.5) science fiction film?")
```


```python
# This example specifies a query and composite filter
retriever.invoke(
    "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated"
)
```

# Wait a second... what else?

Self-query retriever with MyScale can do more! Let's find out.


```python
# You can use length(genres) to do anything you want
retriever.invoke("What's a movie that have more than 1 genres?")
```


```python
# Fine-grained datetime? You got it already.
retriever.invoke("What's a movie that release after feb 1995?")
```


```python
# Don't know what your exact filter should be? Use string pattern match!
retriever.invoke("What's a movie whose name is like Andrei?")
```


```python
# Contain works for lists: so you can match a list with contain comparator!
retriever.invoke("What's a movie who has genres science fiction and adventure?")
```

## Filter k

We can also use the self query retriever to specify `k`: the number of documents to fetch.

We can do this by passing `enable_limit=True` to the constructor.


```python
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True,
)
```


```python
# This example only specifies a relevant query
retriever.invoke("what are two movies about dinosaurs")
```
