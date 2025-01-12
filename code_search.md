## Code Search with Vector Embeddings and Qdrant

*Authored by: [Qdrant Team](https://qdrant.tech/)*

In this notebook, we demonstrate how you can use vector embeddings to navigate a codebase, and find relevant code snippets. We'll search codebases using natural semantic queries, and search for code based on a similar logic.

You can check out the [live deployment](https://code-search.qdrant.tech/) of this approach which exposes the Qdrant codebase for search with a web interface.

### The approach

We need two models to accomplish our goal.

- General usage neural encoder for Natural Language Processing (NLP), in our case [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). We'll call this NLP model.

- Specialized embeddings for code-to-code similarity search. We'll use the [jinaai/jina-embeddings-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code) model for the task. It supports English and 30 widely used programming languages with a 8192 sequence length. Let's call this code model.

To prepare our code for the NLP model, we need to preprocess the code to a format that closely resembles natural language. The code model supports a variety of standard programming languages, so there is no need to preprocess the snippets. We can use the code as is.

## Installing Dependencies

Let's install the packages we'll work with.

- [inflection](https://pypi.org/project/inflection/) - A string transformation library. It singularizes and pluralizes English words, and transforms CamelCase to underscored string.
- [fastembed](https://pypi.org/project/fastembed/) - A CPU-first, lightweight library for generating vector embeddings. [GPU support is available](https://github.com/qdrant/fastembed#%EF%B8%8F-fastembed-on-a-gpu).
- [qdrant-client](https://pypi.org/project/qdrant-client/) - Official Python library to interface with the Qdrant server.


```python
%pip install inflection qdrant-client fastembed
```

### Data preparation

Chunking the application sources into smaller parts is a non-trivial task. In general, functions, class methods, structs, enums, and all the other language-specific constructs are good candidates for chunks. They are big enough to contain some meaningful information, but small enough to be processed by embedding models with a limited context window. You can also use docstrings, comments, and other metadata can be used to enrich the chunks with additional information.

<div style="text-align:center"><img src="https://huggingface.co/datasets/Anush008/cookbook-images/resolve/main/data-chunking.png" /></div>

Text-based search is based on function signatures, but code search may return smaller pieces, such as loops. So, if we receive a particular function signature from the NLP model and part of its implementation from the code model, we merge the results.

### Parsing the Codebaase

We'll use the [Qdrant codebase](https://github.com/qdrant/qdrant) for this demo.
While this codebase uses Rust, you can use this approach with any other language. You can use an [Language Server Protocol (LSP)](https://microsoft.github.io/language-server-protocol/) tool to build a graph of the codebase, and then extract chunks. We did our work with the [rust-analyzer](https://rust-analyzer.github.io/). We exported the parsed codebase into the [LSIF](https://microsoft.github.io/language-server-protocol/specifications/lsif/0.4.0/specification/) format, a standard for code intelligence data. Next, we used the LSIF data to navigate the codebase and extract the chunks.

You can use the same approach for other languages. There are [plenty of implementations](https://microsoft.github.io/language-server-protocol/implementors/servers/) available.

We will then export the chunks into JSON documents with not only the code itself, but also context with the location of the code in the project.

You can examine the Qdrant structures, parsed in JSON, in the [structures.jsonl file](https://storage.googleapis.com/tutorial-attachments/code-search/structures.jsonl) in our Google Cloud Storage bucket. Download it and use it as a source of data for our code search.


```python
!wget https://storage.googleapis.com/tutorial-attachments/code-search/structures.jsonl
```

Next, load the file and parse the lines into a list of dictionaries:


```python
import json

structures = []
with open("structures.jsonl", "r") as fp:
    for i, row in enumerate(fp):
        entry = json.loads(row)
        structures.append(entry)
```

Let's see how one entry looks like.


```python
structures[0]
```

```python
{'name': 'InvertedIndexRam',
 'signature': '# [doc = " Inverted flatten index from dimension id to posting list"] # [derive (Debug , Clone , PartialEq)] pub struct InvertedIndexRam { # [doc = " Posting lists for each dimension flattened (dimension id -> posting list)"] # [doc = " Gaps are filled with empty posting lists"] pub postings : Vec < PostingList > , # [doc = " Number of unique indexed vectors"] # [doc = " pre-computed on build and upsert to avoid having to traverse the posting lists."] pub vector_count : usize , }',
 'code_type': 'Struct',
 'docstring': '= " Inverted flatten index from dimension id to posting list"',
 'line': 15,
 'line_from': 13,
 'line_to': 22,
 'context': {'module': 'inverted_index',
  'file_path': 'lib/sparse/src/index/inverted_index/inverted_index_ram.rs',
  'file_name': 'inverted_index_ram.rs',
  'struct_name': None,
  'snippet': '/// Inverted flatten index from dimension id to posting list\n#[derive(Debug, Clone, PartialEq)]\npub struct InvertedIndexRam {\n    /// Posting lists for each dimension flattened (dimension id -> posting list)\n    /// Gaps are filled with empty posting lists\n    pub postings: Vec<PostingList>,\n    /// Number of unique indexed vectors\n    /// pre-computed on build and upsert to avoid having to traverse the posting lists.\n    pub vector_count: usize,\n}\n'}}
  ```

### Code to natural language conversion

Each programming language has its own syntax which is not a part of the natural language. Thus, a general-purpose model probably does not understand the code as is. We can, however, normalize the data by removing code specifics and including additional context, such as module, class, function, and file name. We take the following steps:

1. Extract the signature of the function, method, or other code construct.
2. Divide camel case and snake case names into separate words.
3. Take the docstring, comments, and other important metadata.
4. Build a sentence from the extracted data using a predefined template.
5. Remove the special characters and replace them with spaces.

We can now define the `textify` function that uses the `inflection` library to carry out our conversions:


```python
import inflection
import re

from typing import Dict, Any


def textify(chunk: Dict[str, Any]) -> str:
    # Get rid of all the camel case / snake case
    # - inflection.underscore changes the camel case to snake case
    # - inflection.humanize converts the snake case to human readable form
    name = inflection.humanize(inflection.underscore(chunk["name"]))
    signature = inflection.humanize(inflection.underscore(chunk["signature"]))

    # Check if docstring is provided
    docstring = ""
    if chunk["docstring"]:
        docstring = f"that does {chunk['docstring']} "

    # Extract the location of that snippet of code
    context = (
        f"module {chunk['context']['module']} " f"file {chunk['context']['file_name']}"
    )
    if chunk["context"]["struct_name"]:
        struct_name = inflection.humanize(
            inflection.underscore(chunk["context"]["struct_name"])
        )
        context = f"defined in struct {struct_name} {context}"

    # Combine all the bits and pieces together
    text_representation = (
        f"{chunk['code_type']} {name} "
        f"{docstring}"
        f"defined as {signature} "
        f"{context}"
    )

    # Remove any special characters and concatenate the tokens
    tokens = re.split(r"\W", text_representation)
    tokens = filter(lambda x: x, tokens)
    return " ".join(tokens)
```

Now we can use `textify` to convert all chunks into text representations:


```python
text_representations = list(map(textify, structures))
```

Let's see how one of our representations looks like:


```python
text_representations[1000]
```

```python
'Function Hnsw discover precision that does Checks discovery search precision when using hnsw index this is different from the tests in defined as Fn hnsw discover precision module integration file hnsw_discover_test rs'
```

### Natural language embeddings




```python
from fastembed import TextEmbedding

batch_size = 5

nlp_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2", threads=0)
nlp_embeddings = nlp_model.embed(text_representations, batch_size=batch_size)
```

### Code Embeddings


```python
code_snippets = [structure["context"]["snippet"] for structure in structures]

code_model = TextEmbedding("jinaai/jina-embeddings-v2-base-code")

code_embeddings = code_model.embed(code_snippets, batch_size=batch_size)
```

### Building Qdrant collection

Qdrant supports multiple modes of deployment. Including in-memory for prototyping, Docker and Qdrant Cloud. You can refer to the [installation instructions](https://qdrant.tech/documentation/guides/installation/) for more information.

We'll continue the tutorial using an in-memory instance.

> [!TIP]
> In-memory can only be used for quick-prototyping and tests. It is a Python implementation of the Qdrant server methods.

Let's create a collection to store our vectors.


```python
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "qdrant-sources"

client = QdrantClient(":memory:")  # Use in-memory storage
# client = QdrantClient("http://locahost:6333")  # For Qdrant server

client.create_collection(
    COLLECTION_NAME,
    vectors_config={
        "text": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
        "code": models.VectorParams(
            size=768,
            distance=models.Distance.COSINE,
        ),
    },
)
```

Our newly created collection is ready to accept the data. Let’s upload the embeddings:


```python
from tqdm import tqdm

points = []
total = len(structures)
print("Number of points to upload: ", total)

for id, (text_embedding, code_embedding, structure) in tqdm(enumerate(zip(nlp_embeddings, code_embeddings, structures)), total=total):
    # FastEmbed returns generators. Embeddings are computed as consumed.
    points.append(
        models.PointStruct(
            id=id,
            vector={
                "text": text_embedding,
                "code": code_embedding,
            },
            payload=structure,
        )
    )

    # Upload points in batches
    if len(points) >= batch_size:
        client.upload_points(COLLECTION_NAME, points=points, wait=True)
        points = []

# Ensure any remaining points are uploaded
if points:
    client.upload_points(COLLECTION_NAME, points=points)

print(f"Total points in collection: {client.count(COLLECTION_NAME).count}")
```

The uploaded points are immediately available for search. Next, query the collection to find relevant code snippets.

### Querying the codebase

We use one of the models to search the collection via Qdrant's new [Query API](https://qdrant.tech/blog/qdrant-1.10.x/). Start with text embeddings. Run the following query “How do I count points in a collection?”. Review the results.


```python
query = "How do I count points in a collection?"

hits = client.query_points(
    COLLECTION_NAME,
    query=next(nlp_model.query_embed(query)).tolist(),
    using="text",
    limit=3,
).points
```

Now, review the results. The following table lists the module, the file name
and score. Each line includes a link to the signature.

| module             | file_name           | score      | signature                                                                                                                                                                                                                                                                                 |
|--------------------|---------------------|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| operations                | types.rs        | 0.5493385 | [`pub struct CountRequestInternal`](https://github.com/qdrant/qdrant/blob/4aac02315bb3ca461a29484094cf6d19025fce99/lib/collection/src/operations/types.rs#L794)                          |
| map_index         | types.rs            | 0.49973965  | [`fn get_points_with_value_count`](https://github.com/qdrant/qdrant/blob/4aac02315bb3ca461a29484094cf6d19025fce99/lib/segment/src/index/field_index/map_index/mod.rs#L89)                       |
| map_index | mutable_map_index.rs | 0.49941066  | [`pub fn get_points_with_value_count`](https://github.com/qdrant/qdrant/blob/4aac02315bb3ca461a29484094cf6d19025fce99/lib/segment/src/index/field_index/map_index/mutable_map_index.rs#L143) |

It seems we were able to find some relevant code structures. Let's try the same with the code embeddings:


```python
hits = client.query_points(
    COLLECTION_NAME,
    query=next(code_model.query_embed(query)).tolist(),
    using="code",
    limit=3,
).points
```

Output:

| module        | file_name                  | score      | signature                                                                                                                                                                                                                                                                   |
|---------------|----------------------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| field_index   | geo_index.rs               | 0.7217579 | [`fn count_indexed_points`](https://github.com/qdrant/qdrant/blob/4aac02315bb3ca461a29484094cf6d19025fce99/lib/segment/src/index/field_index/geo_index/mod.rs#L319)         |
| numeric_index | mod.rs                     | 0.7113214  | [`fn count_indexed_points`](https://github.com/qdrant/qdrant/blob/4aac02315bb3ca461a29484094cf6d19025fce99/lib/segment/src/index/field_index/numeric_index/mod.rs#L317) |
| full_text_index     | text_index.rs                     | 0.6993165  | [`fn count_indexed_points`](https://github.com/qdrant/qdrant/blob/4aac02315bb3ca461a29484094cf6d19025fce99/lib/segment/src/index/field_index/full_text_index/text_index.rs#L179)     |

While the scores retrieved by different models are not comparable, but we can
see that the results are different. Code and text embeddings can capture
different aspects of the codebase. We can use both models to query the collection
and then combine the results to get the most relevant code snippets.


```python
from qdrant_client import models

hits = client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        models.Prefetch(
            query=next(nlp_model.query_embed(query)).tolist(),
            using="text",
            limit=5,
        ),
        models.Prefetch(
            query=next(code_model.query_embed(query)).tolist(),
            using="code",
            limit=5,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF)
).points
```


```python
for hit in hits:
    print(
        "| ",
        hit.payload["context"]["module"], " | ",
        hit.payload["context"]["file_path"], " | ",
        hit.score, " | `",
        hit.payload["signature"], "` |"
    )
```

    |  operations  |  lib/collection/src/operations/types.rs  |  0.5  | ` # [doc = " Count Request"] # [doc = " Counts the number of points which satisfy the given filter."] # [doc = " If filter is not provided, the count of all points in the collection will be returned."] # [derive (Debug , Deserialize , Serialize , JsonSchema , Validate)] # [serde (rename_all = "snake_case")] pub struct CountRequestInternal { # [doc = " Look only for points which satisfies this conditions"] # [validate] pub filter : Option < Filter > , # [doc = " If true, count exact number of points. If false, count approximate number of points faster."] # [doc = " Approximate count might be unreliable during the indexing process. Default: true"] # [serde (default = "default_exact_count")] pub exact : bool , } ` |
    |  field_index  |  lib/segment/src/index/field_index/geo_index.rs  |  0.5  | ` fn count_indexed_points (& self) -> usize ` |
    |  map_index  |  lib/segment/src/index/field_index/map_index/mod.rs  |  0.33333334  | ` fn get_points_with_value_count < Q > (& self , value : & Q) -> Option < usize > where Q : ? Sized , N : std :: borrow :: Borrow < Q > , Q : Hash + Eq , ` |
    |  numeric_index  |  lib/segment/src/index/field_index/numeric_index/mod.rs  |  0.33333334  | ` fn count_indexed_points (& self) -> usize ` |
    |  fixtures  |  lib/segment/src/fixtures/payload_context_fixture.rs  |  0.25  | ` fn total_point_count (& self) -> usize ` |
    |  map_index  |  lib/segment/src/index/field_index/map_index/mutable_map_index.rs  |  0.25  | ` fn get_points_with_value_count < Q > (& self , value : & Q) -> Option < usize > where Q : ? Sized , N : std :: borrow :: Borrow < Q > , Q : Hash + Eq , ` |
    |  id_tracker  |  lib/segment/src/id_tracker/simple_id_tracker.rs  |  0.2  | ` fn total_point_count (& self) -> usize ` |
    |  map_index  |  lib/segment/src/index/field_index/map_index/mod.rs  |  0.2  | ` fn count_indexed_points (& self) -> usize ` |
    |  map_index  |  lib/segment/src/index/field_index/map_index/mod.rs  |  0.16666667  | ` fn count_indexed_points (& self) -> usize ` |
    |  field_index  |  lib/segment/src/index/field_index/stat_tools.rs  |  0.16666667  | ` fn number_of_selected_points (points : usize , values : usize) -> usize ` |
    

This is one example of how you can fuse the results from different models.
In a real-world scenario, you might run some reranking and deduplication, as well as additional processing of the results.

### Grouping the results

You can improve the search results, by grouping them by payload properties.
In our case, we can group the results by the module. If we use code embeddings,
we can see multiple results from the `map_index` module. Let's group the
results and assume a single result per module:


```python
results = client.query_points_groups(
    COLLECTION_NAME,
    query=next(code_model.query_embed(query)).tolist(),
    using="code",
    group_by="context.module",
    limit=5,
    group_size=1,
)
```


```python
for group in results.groups:
    for hit in group.hits:
        print(
            "| ",
            hit.payload["context"]["module"], " | ",
            hit.payload["context"]["file_name"], " | ",
            hit.score, " | `",
            hit.payload["signature"], "` |"
        )
```

    |  field_index  |  geo_index.rs  |  0.7217579  | ` fn count_indexed_points (& self) -> usize ` |
    |  numeric_index  |  mod.rs  |  0.7113214  | ` fn count_indexed_points (& self) -> usize ` |
    |  fixtures  |  payload_context_fixture.rs  |  0.6993165  | ` fn total_point_count (& self) -> usize ` |
    |  map_index  |  mod.rs  |  0.68385994  | ` fn count_indexed_points (& self) -> usize ` |
    |  full_text_index  |  text_index.rs  |  0.6660142  | ` fn count_indexed_points (& self) -> usize ` |
    

That concludes our tutorial. Thanks for taking the time to get here. We've just begun exploring what's possible with vector embeddings and how to improve it. Feel free to experiment your way; you could build something very cool! Do share it with us 🙏 We are [here](https://qdrant.tech/contact-us/).
