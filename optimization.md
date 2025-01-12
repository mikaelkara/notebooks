# Optimization

This notebook goes over how to optimize chains using LangChain and [LangSmith](https://smith.langchain.com).

## Set up

We will set an environment variable for LangSmith, and load the relevant data


```python
import os

os.environ["LANGCHAIN_PROJECT"] = "movie-qa"
```


```python
import pandas as pd
```


```python
df = pd.read_csv("data/imdb_top_1000.csv")
```


```python
df["Released_Year"] = df["Released_Year"].astype(int, errors="ignore")
```

## Create the initial retrieval chain

We will use a self-query retriever


```python
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
```


```python
records = df.to_dict("records")
documents = [Document(page_content=d["Overview"], metadata=d) for d in records]
```


```python
vectorstore = Chroma.from_documents(documents, embeddings)
```


```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

metadata_field_info = [
    AttributeInfo(
        name="Released_Year",
        description="The year the movie was released",
        type="int",
    ),
    AttributeInfo(
        name="Series_Title",
        description="The title of the movie",
        type="str",
    ),
    AttributeInfo(
        name="Genre",
        description="The genre of the movie",
        type="string",
    ),
    AttributeInfo(
        name="IMDB_Rating", description="A 1-10 rating for the movie", type="float"
    ),
]
document_content_description = "Brief summary of a movie"
llm = ChatOpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)
```


```python
from langchain_core.runnables import RunnablePassthrough
```


```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
```


```python
prompt = ChatPromptTemplate.from_template(
    """Answer the user's question based on the below information:

Information:

{info}

Question: {question}"""
)
generator = (prompt | ChatOpenAI() | StrOutputParser()).with_config(
    run_name="generator"
)
```


```python
chain = (
    RunnablePassthrough.assign(info=(lambda x: x["question"]) | retriever) | generator
)
```

## Run examples

Run examples through the chain. This can either be manually, or using a list of examples, or production traffic


```python
chain.invoke({"question": "what is a horror movie released in early 2000s"})
```




    'One of the horror movies released in the early 2000s is "The Ring" (2002), directed by Gore Verbinski.'



## Annotate

Now, go to LangSmitha and annotate those examples as correct or incorrect

## Create Dataset

We can now create a dataset from those runs.

What we will do is find the runs marked as correct, then grab the sub-chains from them. Specifically, the query generator sub chain and the final generation step


```python
from langsmith import Client

client = Client()
```


```python
runs = list(
    client.list_runs(
        project_name="movie-qa",
        execution_order=1,
        filter="and(eq(feedback_key, 'correctness'), eq(feedback_score, 1))",
    )
)

len(runs)
```




    14




```python
gen_runs = []
query_runs = []
for r in runs:
    gen_runs.extend(
        list(
            client.list_runs(
                project_name="movie-qa",
                filter="eq(name, 'generator')",
                trace_id=r.trace_id,
            )
        )
    )
    query_runs.extend(
        list(
            client.list_runs(
                project_name="movie-qa",
                filter="eq(name, 'query_constructor')",
                trace_id=r.trace_id,
            )
        )
    )
```


```python
runs[0].inputs
```




    {'question': 'what is a high school comedy released in early 2000s'}




```python
runs[0].outputs
```




    {'output': 'One high school comedy released in the early 2000s is "Mean Girls" starring Lindsay Lohan, Rachel McAdams, and Tina Fey.'}




```python
query_runs[0].inputs
```




    {'query': 'what is a high school comedy released in early 2000s'}




```python
query_runs[0].outputs
```




    {'output': {'query': 'high school comedy',
      'filter': {'operator': 'and',
       'arguments': [{'comparator': 'eq', 'attribute': 'Genre', 'value': 'comedy'},
        {'operator': 'and',
         'arguments': [{'comparator': 'gte',
           'attribute': 'Released_Year',
           'value': 2000},
          {'comparator': 'lt', 'attribute': 'Released_Year', 'value': 2010}]}]}}}




```python
gen_runs[0].inputs
```




    {'question': 'what is a high school comedy released in early 2000s',
     'info': []}




```python
gen_runs[0].outputs
```




    {'output': 'One high school comedy released in the early 2000s is "Mean Girls" starring Lindsay Lohan, Rachel McAdams, and Tina Fey.'}



## Create datasets

We can now create datasets for the query generation and final generation step.
We do this so that (1) we can inspect the datapoints, (2) we can edit them if needed, (3) we can add to them over time


```python
client.create_dataset("movie-query_constructor")

inputs = [r.inputs for r in query_runs]
outputs = [r.outputs for r in query_runs]

client.create_examples(
    inputs=inputs, outputs=outputs, dataset_name="movie-query_constructor"
)
```


```python
client.create_dataset("movie-generator")

inputs = [r.inputs for r in gen_runs]
outputs = [r.outputs for r in gen_runs]

client.create_examples(inputs=inputs, outputs=outputs, dataset_name="movie-generator")
```

## Use as few shot examples

We can now pull down a dataset and use them as few shot examples in a future chain


```python
examples = list(client.list_examples(dataset_name="movie-query_constructor"))
```


```python
import json


def filter_to_string(_filter):
    if "operator" in _filter:
        args = [filter_to_string(f) for f in _filter["arguments"]]
        return f"{_filter['operator']}({','.join(args)})"
    else:
        comparator = _filter["comparator"]
        attribute = json.dumps(_filter["attribute"])
        value = json.dumps(_filter["value"])
        return f"{comparator}({attribute}, {value})"
```


```python
model_examples = []

for e in examples:
    if "filter" in e.outputs["output"]:
        string_filter = filter_to_string(e.outputs["output"]["filter"])
    else:
        string_filter = "NO_FILTER"
    model_examples.append(
        (
            e.inputs["query"],
            {"query": e.outputs["output"]["query"], "filter": string_filter},
        )
    )
```


```python
retriever1 = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
    chain_kwargs={"examples": model_examples},
)
```


```python
chain1 = (
    RunnablePassthrough.assign(info=(lambda x: x["question"]) | retriever1) | generator
)
```


```python
chain1.invoke(
    {"question": "what are good action movies made before 2000 but after 1997?"}
)
```




    '1. "Saving Private Ryan" (1998) - Directed by Steven Spielberg, this war film follows a group of soldiers during World War II as they search for a missing paratrooper.\n\n2. "The Matrix" (1999) - Directed by the Wachowskis, this science fiction action film follows a computer hacker who discovers the truth about the reality he lives in.\n\n3. "Lethal Weapon 4" (1998) - Directed by Richard Donner, this action-comedy film follows two mismatched detectives as they investigate a Chinese immigrant smuggling ring.\n\n4. "The Fifth Element" (1997) - Directed by Luc Besson, this science fiction action film follows a cab driver who must protect a mysterious woman who holds the key to saving the world.\n\n5. "The Rock" (1996) - Directed by Michael Bay, this action thriller follows a group of rogue military men who take over Alcatraz and threaten to launch missiles at San Francisco.'




```python

```
