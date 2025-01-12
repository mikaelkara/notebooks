# Retrieval Augmented Generation (Using Open Source Vector Store) - Procurement Contract Analyst - Palm2 & LangChain

<table align="left">

  <td>
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/search/retrieval-augmented-generation/examples/contract_analysis.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Colab logo"> Run in Colab
    </a>
  </td>
  <td>
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/search/retrieval-augmented-generation/examples/contract_analysis.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo">
      View on GitHub
    </a>
  </td>
  <td>
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/search/retrieval-augmented-generation/examples/contract_analysis.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo">
      Open in Vertex AI Workbench
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Guru Rangavittal](https://github.com/guruvittal) |

## Installation & Authentication

Install LangChain, Vertex AI LLM SDK, ChromaDB, and related libraries.



```
%pip install -q google-cloud-aiplatform==1.36.0 langchain==0.0.327 unstructured chromadb==0.4.15 --upgrade --user
```

### Restart current runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.


```
# Restart kernel after installs so that your environment can access the new packages

import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```




    {'status': 'ok', 'restart': True}



<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>


**Authenticate**
Within the Colab a simple user authentication is adequate.



```
import sys

if "google.colab" in sys.modules:
    from google.colab import auth as google_auth

    google_auth.authenticate_user()
```

## Get Libraries & Classes


**Reference Libraries**

In this section, we will identify all the library classes that will be referenced in the code.



```
from google.cloud import aiplatform
from langchain.document_loaders import GCSDirectoryLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI

# Chroma DB as Vector Store Database
from langchain.vectorstores import Chroma

# Using Vertex AI
import vertexai

print(f"Vertex AI SDK version: {aiplatform.__version__}")
```

    Vertex AI SDK version: 1.35.0
    

## Initialize Vertex AI

**We will need a project id and location where the Vertex AI compute and embedding will be hosted**



```
PROJECT_ID = "PROJECT_ID"  # @param {type:"string"}

LOCATION = "us-central1"  # @param {type:"string"}

# Initialize Vertex AI SDK
vertexai.init(project=PROJECT_ID, location=LOCATION)
```

## Ingest the Contracts to build the context for the LLM

_Load all the Procurement Contract Documents_



```
loader = GCSDirectoryLoader(
    project_name=PROJECT_ID, bucket="contractunderstandingatticusdataset"
)
documents = loader.load()
```

_Split documents into chunks as needed by the token limit of the LLM and let there be an overlap between the chunks_



```
# split the documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(f"# of documents = {len(docs)}")
```

    # of documents = 2150
    

## Structuring the ingested documents in a vector space using a Vector Database


_Create an embedding vector engine for all the text in the contract documents that have been ingested_



```
# Define Text Embeddings model
embedding = VertexAIEmbeddings()


embedding
```




    VertexEmbeddings(model_name='textembedding-gecko@001', model=<class 'vertexai.language_models._language_models.TextEmbeddingModel'>, requests_per_minute=590)



_Create a vector store and store the embeddings in the vector store_



```
contracts_vector_db = Chroma.from_documents(docs, embedding)
```

## Obtain handle to the retriever

We will use the native retriever provided by Chroma DB to perform similarity search within the contracts document vector store among the different document chunks so as to return that document chunk which has the lowest vectoral "distance" with the incoming user query.



```
# Expose index to the retriever
retriever = contracts_vector_db.as_retriever(
    search_type="similarity", search_kwargs={"k": 2}
)
```

## Define a Retrieval QA Chain to use retriever



```
# Create chain to answer questions
from langchain.chains import RetrievalQA

llm = VertexAI(
    model_name="text-bison-32k",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

# Uses LLM to synthesize results from the search index.
# We use Vertex AI PaLM Text API for LLM
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)
```

## Leverage LLM to search from retriever


_Example:_



```
query = "Who all entered into agreement with Sagebrush?"
result = qa({"query": query})
print(result)
```

    Waiting
    {'query': 'Who all entered into agreement with Sagebrush?', 'result': ' Allison Transmission Holdings, Inc.', 'source_documents': [Document(page_content='Each party cooperated and participated in the drafting and preparation of this Agreement and the documents referred to herein, and any and all drafts relating thereto exchanged among the parties shall be deemed the work product of all of the parties and may not be construed against any party by reason of its drafting or preparation. Accordingly, any rule of law or any legal decision that would require interpretation of any ambiguities in this Agreement against any party that drafted or prepared it is of no application and is hereby expressly waived by each of the parties hereto, and any controversy over interpretations of this Agreement shall be decided without regards to events of drafting or preparation.\n\n[Signature Pages Follow]   7\n\nIN WITNESS WHEREOF, each of the parties hereto has executed this COOPERATION AGREEMENT or caused the same to be executed by its duly authorized representative as of the date first above written. Allison Transmission Holdings, Inc.', metadata={'source': 'gs://contractunderstandingatticusdataset/ALLISONTRANSMISSIONHOLDINGSINC_12_15_2014-EX-99.1-COOPERATION AGREEMENT.txt'}), Document(page_content='Each party cooperated and participated in the drafting and preparation of this Agreement and the documents referred to herein, and any and all drafts relating thereto exchanged among the parties shall be deemed the work product of all of the parties and may not be construed against any party by reason of its drafting or preparation. Accordingly, any rule of law or any legal decision that would require interpretation of any ambiguities in this Agreement against any party that drafted or prepared it is of no application and is hereby expressly waived by each of the parties hereto, and any controversy over interpretations of this Agreement shall be decided without regards to events of drafting or preparation.\n\n[Signature Pages Follow]   7\n\nIN WITNESS WHEREOF, each of the parties hereto has executed this COOPERATION AGREEMENT or caused the same to be executed by its duly authorized representative as of the date first above written. Allison Transmission Holdings, Inc.', metadata={'source': 'gs://contractunderstandingatticusdataset/ALLISONTRANSMISSIONHOLDINGSINC_12_15_2014-EX-99.1-COOPERATION AGREEMENT.txt'})]}
    

## Build a Front End

Enable a simple front end so users can query against contract documents and obtain intelligent answers with grounding information that references the base documents that was used to respond to user query



```
%pip install -q gradio
```


```
from google.cloud import storage
import gradio as gr


def chatbot(input_text):
    result = qa({"query": input_text})

    return (
        result["result"],
        get_public_url(result["source_documents"][0].metadata["source"]),
        result["source_documents"][0].metadata["source"],
    )


def get_public_url(uri):
    """Returns the public URL for a file in Google Cloud Storage."""
    # Split the URI into its components
    components = uri.split("/")

    # Get the bucket name
    bucket_name = components[2]

    # Get the file name
    file_name = components[3]

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    return blob.public_url


print("Launching Gradio")

iface = gr.Interface(
    fn=chatbot,
    inputs=[gr.Textbox(label="Query")],
    examples=[
        "Who are parties to ADMA agreement",
        "What is the agreement between MICOA & Stratton Cheeseman",
        "What is the commission % that Stratton Cheeseman will get from MICOA and how much will they get if MICOA's revenues are $100",
    ],
    title="Contract Analyst",
    outputs=[
        gr.Textbox(label="Response"),
        gr.Textbox(label="URL"),
        gr.Textbox(label="Cloud Storage URI"),
    ],
    theme=gr.themes.Soft,
)

iface.launch(share=False)
```
