# Google Cloud Vertex AI Reranker

> The [Vertex Search Ranking API](https://cloud.google.com/generative-ai-app-builder/docs/ranking) is one of the standalone APIs in [Vertex AI Agent Builder](https://cloud.google.com/generative-ai-app-builder/docs/builder-apis). It takes a list of documents and reranks those documents based on how relevant the documents are to a query. Compared to embeddings, which look only at the semantic similarity of a document and a query, the ranking API can give you precise scores for how well a document answers a given query. The ranking API can be used to improve the quality of search results after retrieving an initial set of candidate documents.

>The ranking API is stateless so there's no need to index documents before calling the API. All you need to do is pass in the query and documents. This makes the API well suited for reranking documents from any document retrievers.

>For more information, see [Rank and rerank documents](https://cloud.google.com/generative-ai-app-builder/docs/ranking).


```python
%pip install --upgrade --quiet langchain langchain-community langchain-google-community langchain-google-community[vertexaisearch] langchain-google-vertexai langchain-chroma langchain-text-splitters beautifulsoup4
```

### Setup


```python
PROJECT_ID = ""
REGION = ""
RANKING_LOCATION_ID = "global"  # @param {type:"string"}

# Initialize GCP project for Vertex AI
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=REGION)
```

### Load and Prepare data

For this example, we will be using the [Google Wiki page](https://en.wikipedia.org/wiki/Google)to demonstrate how the Vertex Ranking API works.

We use a standard pipeline of `load -> split -> embed data`.

The embeddings are created using the [Vertex Embeddings API](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings#supported_models) model - `textembedding-gecko@003`


```python
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

vectordb = None

# Load wiki page
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Google")
data = loader.load()

# Split doc into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=5)
splits = text_splitter.split_documents(data)

print(f"Your {len(data)} documents have been split into {len(splits)} chunks")

if vectordb is not None:  # delete existing vectordb if it already exists
    vectordb.delete_collection()

embedding = VertexAIEmbeddings(model_name="textembedding-gecko@003")
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
```

    Your 1 documents have been split into 266 chunks
    


```python
import pandas as pd
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_google_community.vertex_rank import VertexAIRank

# Instantiate the VertexAIReranker with the SDK manager
reranker = VertexAIRank(
    project_id=PROJECT_ID,
    location_id=RANKING_LOCATION_ID,
    ranking_config="default_ranking_config",
    title_field="source",
    top_n=5,
)

basic_retriever = vectordb.as_retriever(search_kwargs={"k": 5})  # fetch top 5 documents

# Create the ContextualCompressionRetriever with the VertexAIRanker as a Reranker
retriever_with_reranker = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=basic_retriever
)
```

### Testing out the Vertex Ranking API

Let's query both the `basic_retriever` and `retriever_with_reranker` with the same query and compare the retrieved documents.

The Ranking API takes in the input from the `basic_retriever` and passes it to the Ranking API.

The ranking API is used to improve the quality of the ranking and determine a score that indicates the relevance of each record to the query.

You can see the difference between the Unranked and the Ranked Documents. The Ranking API moves the most semantically relevant documents to the top of the context window of the LLM thus helping it form a better answer with reasoning.


```python
import pandas as pd

# Use the basic_retriever and the retriever_with_reranker to get relevant documents
query = "how did the name google originate?"
retrieved_docs = basic_retriever.invoke(query)
reranked_docs = retriever_with_reranker.invoke(query)

# Create two lists of results for unranked and ranked docs
unranked_docs_content = [docs.page_content for docs in retrieved_docs]
ranked_docs_content = [docs.page_content for docs in reranked_docs]

# Create a comparison DataFrame using the padded lists
comparison_df = pd.DataFrame(
    {
        "Unranked Documents": unranked_docs_content,
        "Ranked Documents": ranked_docs_content,
    }
)

comparison_df
```





  <div id="df-43c4f5f2-c31d-4664-85dd-60cad39bd5fa" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unranked Documents</th>
      <th>Ranked Documents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>^ a b Brin, Sergey; Page, Lawrence (1998). "The anatomy of a large-scale hypertextual Web search engine" (PDF). Computer Networks and ISDN Systems. 30 (1–7): 107–117. CiteSeerX 10.1.1.115.5930. doi:10.1016/S0169-7552(98)00110-X. ISSN 0169-7552. S2CID 7587743. Archived (PDF) from the original on September 27, 2015. Retrieved April 7, 2019.\n\n^ "About: RankDex". Archived from the original on January 20, 2012. Retrieved September 29, 2010., RankDex\n\n^ "Method for node ranking in a linked database". Google Patents. Archived from the original on October 15, 2015. Retrieved October 19, 2015.\n\n^ Koller, David (January 2004). "Origin of the name "Google"". Stanford University. Archived from the original on June 27, 2012.</td>
      <td>The name "Google" originated from a misspelling of "googol",[211][212] which refers to the number represented by a 1 followed by one-hundred zeros. Page and Brin write in their original paper on PageRank:[33] "We chose our system name, Google, because it is a common spelling of googol, or 10100[,] and fits well with our goal of building very large-scale search engines." Having found its way increasingly into everyday language, the verb "google" was added to the Merriam Webster Collegiate Dictionary and the Oxford English Dictionary in 2006, meaning "to use the Google search engine to obtain information on the Internet."[213][214] Google's mission statement, from the outset, was "to organize the world's information and make it universally accessible and useful",[215] and its unofficial</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Eventually, they changed the name to Google; the name of the search engine was a misspelling of the word googol,[21][36][37] a very large number written 10100 (1 followed by 100 zeros), picked to signify that the search engine was intended to provide large quantities of information.[38]</td>
      <td>Eventually, they changed the name to Google; the name of the search engine was a misspelling of the word googol,[21][36][37] a very large number written 10100 (1 followed by 100 zeros), picked to signify that the search engine was intended to provide large quantities of information.[38]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The name "Google" originated from a misspelling of "googol",[211][212] which refers to the number represented by a 1 followed by one-hundred zeros. Page and Brin write in their original paper on PageRank:[33] "We chose our system name, Google, because it is a common spelling of googol, or 10100[,] and fits well with our goal of building very large-scale search engines." Having found its way increasingly into everyday language, the verb "google" was added to the Merriam Webster Collegiate Dictionary and the Oxford English Dictionary in 2006, meaning "to use the Google search engine to obtain information on the Internet."[213][214] Google's mission statement, from the outset, was "to organize the world's information and make it universally accessible and useful",[215] and its unofficial</td>
      <td>^ Meijer, Bart (January 3, 2019). "Google shifted $23 billion to tax haven Bermuda in 2017: filing". Reuters. Archived from the original on January 3, 2019. Retrieved January 3, 2019. Google moved 19.9 billion euros ($22.7 billion) through a Dutch shell company to Bermuda in 2017, as part of an arrangement that allows it to reduce its foreign tax bill\n\n^ Hamburger, Tom; Gold, Matea (April 13, 2014). "Google, once disdainful of lobbying, now a master of Washington influence". The Washington Post. Archived from the original on October 27, 2017. Retrieved August 22, 2017.\n\n^ Koller, David (January 2004). "Origin of the name, "Google."". Stanford University. Archived from the original on June 27, 2012. Retrieved May 28, 2006.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>^ Meijer, Bart (January 3, 2019). "Google shifted $23 billion to tax haven Bermuda in 2017: filing". Reuters. Archived from the original on January 3, 2019. Retrieved January 3, 2019. Google moved 19.9 billion euros ($22.7 billion) through a Dutch shell company to Bermuda in 2017, as part of an arrangement that allows it to reduce its foreign tax bill\n\n^ Hamburger, Tom; Gold, Matea (April 13, 2014). "Google, once disdainful of lobbying, now a master of Washington influence". The Washington Post. Archived from the original on October 27, 2017. Retrieved August 22, 2017.\n\n^ Koller, David (January 2004). "Origin of the name, "Google."". Stanford University. Archived from the original on June 27, 2012. Retrieved May 28, 2006.</td>
      <td>^ a b Brin, Sergey; Page, Lawrence (1998). "The anatomy of a large-scale hypertextual Web search engine" (PDF). Computer Networks and ISDN Systems. 30 (1–7): 107–117. CiteSeerX 10.1.1.115.5930. doi:10.1016/S0169-7552(98)00110-X. ISSN 0169-7552. S2CID 7587743. Archived (PDF) from the original on September 27, 2015. Retrieved April 7, 2019.\n\n^ "About: RankDex". Archived from the original on January 20, 2012. Retrieved September 29, 2010., RankDex\n\n^ "Method for node ranking in a linked database". Google Patents. Archived from the original on October 15, 2015. Retrieved October 19, 2015.\n\n^ Koller, David (January 2004). "Origin of the name "Google"". Stanford University. Archived from the original on June 27, 2012.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>^ Swant, Marty. "The World's Valuable Brands". Forbes. Archived from the original on October 18, 2020. Retrieved January 19, 2022.\n\n^ "Best Global Brands". Interbrand. Archived from the original on February 1, 2022. Retrieved March 7, 2011.\n\n^ a b c d "How we started and where we are today – Google". about.google. Archived from the original on April 22, 2020. Retrieved April 24, 2021.\n\n^ Brezina, Corona (2013). Sergey Brin, Larry Page, Eric Schmidt, and Google (1st ed.). New York: Rosen Publishing Group. p. 18. ISBN 978-1-4488-6911-4. LCCN 2011039480.\n\n^ a b c "Our history in depth". Google Company. Archived from the original on April 1, 2012. Retrieved July 15, 2017.</td>
      <td>^ Swant, Marty. "The World's Valuable Brands". Forbes. Archived from the original on October 18, 2020. Retrieved January 19, 2022.\n\n^ "Best Global Brands". Interbrand. Archived from the original on February 1, 2022. Retrieved March 7, 2011.\n\n^ a b c d "How we started and where we are today – Google". about.google. Archived from the original on April 22, 2020. Retrieved April 24, 2021.\n\n^ Brezina, Corona (2013). Sergey Brin, Larry Page, Eric Schmidt, and Google (1st ed.). New York: Rosen Publishing Group. p. 18. ISBN 978-1-4488-6911-4. LCCN 2011039480.\n\n^ a b c "Our history in depth". Google Company. Archived from the original on April 1, 2012. Retrieved July 15, 2017.</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-43c4f5f2-c31d-4664-85dd-60cad39bd5fa')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-43c4f5f2-c31d-4664-85dd-60cad39bd5fa button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-43c4f5f2-c31d-4664-85dd-60cad39bd5fa');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-fff80078-f146-44f5-9eff-d91c9305c276">
  <button class="colab-df-quickchart" onclick="quickchart('df-fff80078-f146-44f5-9eff-d91c9305c276')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-fff80078-f146-44f5-9eff-d91c9305c276 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_7648ee4a-f747-429c-820f-e03d3c59f765">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('comparison_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_7648ee4a-f747-429c-820f-e03d3c59f765 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('comparison_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>




Let's inspect a couple of reranked documents. We observe that the retriever still returns the relevant Langchain type [documents](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) but as part of the metadata field, we also recieve the `relevance_score` from the Ranking API.


```python
for i in range(2):
    print(f"Document {i}")
    print(reranked_docs[i])
    print("----------------------------------------------------------\n")
```



<style>
  pre {
      white-space: pre-wrap;
  }
</style>



    Document 0
    page_content='The name "Google" originated from a misspelling of "googol",[211][212] which refers to the number represented by a 1 followed by one-hundred zeros. Page and Brin write in their original paper on PageRank:[33] "We chose our system name, Google, because it is a common spelling of googol, or 10100[,] and fits well with our goal of building very large-scale search engines." Having found its way increasingly into everyday language, the verb "google" was added to the Merriam Webster Collegiate Dictionary and the Oxford English Dictionary in 2006, meaning "to use the Google search engine to obtain information on the Internet."[213][214] Google\'s mission statement, from the outset, was "to organize the world\'s information and make it universally accessible and useful",[215] and its unofficial' metadata={'id': '2', 'relevance_score': 0.9800000190734863, 'source': 'https://en.wikipedia.org/wiki/Google'}
    ----------------------------------------------------------
    
    Document 1
    page_content='Eventually, they changed the name to Google; the name of the search engine was a misspelling of the word googol,[21][36][37] a very large number written 10100 (1 followed by 100 zeros), picked to signify that the search engine was intended to provide large quantities of information.[38]' metadata={'id': '1', 'relevance_score': 0.75, 'source': 'https://en.wikipedia.org/wiki/Google'}
    ----------------------------------------------------------
    
    

### Putting it all together

This shows an example of a complete RAG chain with a simple prompt template on how you can perform reranking using the Vertex Ranking API.




```python
from langchain.chains import LLMChain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_vertexai import VertexAI

llm = VertexAI(model_name="gemini-1.0-pro-002")

# Instantiate the VertexAIReranker with the SDK manager
reranker = VertexAIRank(
    project_id=PROJECT_ID,
    location_id=RANKING_LOCATION_ID,
    ranking_config="default_ranking_config",
    title_field="source",  # metadata field key from your existing documents
    top_n=5,
)

# value of k can be set to a higher value as well for tweaking performance
# eg: # of docs: basic_retriever(100) -> reranker(5)
basic_retriever = vectordb.as_retriever(search_kwargs={"k": 5})  # fetch top 5 documents

# Create the ContextualCompressionRetriever with the VertexAIRanker as a Reranker
retriever_with_reranker = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=basic_retriever
)

template = """
<context>
{context}
</context>

Question:
{query}

Don't give information outside the context or repeat your findings.
Answer:
"""
prompt = PromptTemplate.from_template(template)

reranker_setup_and_retrieval = RunnableParallel(
    {"context": retriever_with_reranker, "query": RunnablePassthrough()}
)

chain = reranker_setup_and_retrieval | prompt | llm
```



<style>
  pre {
      white-space: pre-wrap;
  }
</style>




```python
query = "how did the name google originate?"
```



<style>
  pre {
      white-space: pre-wrap;
  }
</style>




```python
chain.invoke(query)
```



<style>
  pre {
      white-space: pre-wrap;
  }
</style>






    'The name "Google" originated as a misspelling of the word "googol," a mathematical term for the number 1 followed by 100 zeros. Larry Page and Sergey Brin, the founders of Google, chose the name because it reflected their goal of building a search engine that could handle massive amounts of information. \n'


