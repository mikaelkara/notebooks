# Simple Retrieval Augmenation Generation (RAG) 

<img src="images/simple_rag_flow.png">


```python
import sys 
sys.path.insert(0, "../llm-prompts")

from rag_utils import read_pdf_chunks, extract_matches, print_matches
import os
from anthropic import Anthropic
from tqdm.auto import tqdm
from llm_clnt_factory_api import ClientFactory, get_commpletion
from rag_utils import print_matches, extract_matches
from pinecone import Pinecone, PodSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
```

1. Setup  global variables
2. Load the environment file


```python
TOP_K = 5
INDEX_NAME = "starter-index"
BOLD_BEGIN = "\033[1m"
BOLD_END   =   "\033[0m"
PINECONE_ENVIRONMENT = "gcp-starter"
PDF_DIRECTORY = "pdfs"
VERBOSE = True
CHUNK_SIZE = 500
CHUNK_OVERLAP = 20
DIR_PATH = os.path.join(os.getcwd(),PDF_DIRECTORY)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# create a model for creating embeddings
MODEL = model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
```

Set up Pinecone environment. Use the .env file to load the Pinecone API key
and the environment name, which is "gcp-starter" in this case, for the GCP starter environment community edition of Pinecone is also available for free.


```python
_ = load_dotenv(find_dotenv())
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if pinecone_api_key is None:
    raise ValueError("Please set the PINECONE_API_KEY environment")
pc = Pinecone(api_key=pinecone_api_key,
              environment="gcp-starter",
              spec=PodSpec(environment="gcp-starter")
             )
```

Check if the Pinecone index exisits 


```python
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes() 
]

if INDEX_NAME in existing_indexes:
    print(f"Index {INDEX_NAME} already exists. Deleting it ...")
    pc.delete_index(INDEX_NAME)
```

    Index starter-index already exists. Deleting it ...
    

### Step 1: Create a new index


```python
print(f"Creating a new index {INDEX_NAME}...")
pc.create_index(name=INDEX_NAME,
                metric="cosine",
                dimension=384,
                spec=PodSpec(environment="gcp-starter")
               )
```

    Creating a new index starter-index...
    


```python
# Connect or get a handle to the index
pindex = pc.Index(INDEX_NAME)
```


```python
# read each file in the directory
for filename in tqdm(os.listdir(DIR_PATH)):
    if filename.endswith('.pdf'):
        file_path = os.path.join(DIR_PATH, filename)
        print(f"Processing file: {file_path}")
        for i, chunk in enumerate(read_pdf_chunks(file_path, 
                    CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)):
            # Process each chunk (e.g., create vector embeddings)
            embeddings = model.encode(chunk)

            # create a metadata batch
            c_id = "".join([str(i), '-', filename])
            sample_doc = [
                { "id":  c_id ,
                    "values": embeddings.tolist(),
                    "metadata": {
                        "text": chunk
                    }
                 }
            ]
            if VERBOSE:
                if i % 100 == 0:
                     print(f"Upserting batch id: { c_id}")
                        
                # upsert to Pinecone
                pindex.upsert(sample_doc)
```


      0%|          | 0/1 [00:00<?, ?it/s]


    Processing file: /Users/julesdamji/git-repos/genai-cookbook/rags/pdfs/HAI_AI-Index-Report_2023.pdf
    Upserting batch id: 0-HAI_AI-Index-Report_2023.pdf
    Upserting batch id: 100-HAI_AI-Index-Report_2023.pdf
    Upserting batch id: 200-HAI_AI-Index-Report_2023.pdf
    Upserting batch id: 300-HAI_AI-Index-Report_2023.pdf
    Upserting batch id: 400-HAI_AI-Index-Report_2023.pdf
    Upserting batch id: 500-HAI_AI-Index-Report_2023.pdf
    Upserting batch id: 600-HAI_AI-Index-Report_2023.pdf
    Upserting batch id: 700-HAI_AI-Index-Report_2023.pdf
    Upserting batch id: 800-HAI_AI-Index-Report_2023.pdf
    Upserting batch id: 900-HAI_AI-Index-Report_2023.pdf
    Upserting batch id: 1000-HAI_AI-Index-Report_2023.pdf
    Upserting batch id: 1100-HAI_AI-Index-Report_2023.pdf
    Upserting batch id: 1200-HAI_AI-Index-Report_2023.pdf
    Upserting batch id: 1300-HAI_AI-Index-Report_2023.pdf
    Upserting batch id: 1400-HAI_AI-Index-Report_2023.pdf
    

### Step 2: Search the index 

Get the matching documents for a user query from the Pinecone index. This
step is the retriever bit. We will use this as part of our context and query.


```python
print("Running a semantic search...")
query = "What are the key takeaways for AI in 2023?"
print(f"Query: {query}")
query_embedding = model.encode(query).tolist()
results = pindex.query(vector=query_embedding, top_k=TOP_K,
                        include_values=False, 
                        include_metadata=True)

print(f"Top {TOP_K} results for the query:")
print_matches(results)
```

    Running a semantic search...
    Query: What are the key takeaways for AI in 2023?
    Top 5 results for the query:
    Score  : 0.69
    Matches: Figure 4.3.20advertising and marketing (8.8%) (Figure 4.3.20). 
    Compared to 2018, some of the less prevalent 
    AI-related themes in 2022 included deep learning 
    (4.8%), autonomous vehicles (3.1%), and data 
    storage and management (3.0%).
    Score  : 0.68
    Matches: broader range of societal actors. This year’s AI Index paints a picture of where we are so far with AI, in order to 
    highlight what might await us in the future.
    Jack Clark and Ray Perrault
    Score  : 0.67
    Matches: Table of Contents
     267
    Artificial Intelligence
    Index Report 2023 Chapter 6 PreviewIn the last 10 years, AI governance discussions have accelerated, resulting in numerous policy proposals in various legislative bodies. This 
    section begins by exploring the legislative initiatives related to AI that have been suggested or enacted in different countries and regions,
    Score  : 0.66
    Matches: increased nearly 6.5 times since 2016.When it comes to AI, 
    policymakers have  
    a lot of thoughts.   
    A qualitative analysis of the 
    parliamentary proceedings of a 
    diverse group of nations reveals 
    that policymakers think about AI 
    from a wide range of perspectives. 
    For example, in 2022, legislators in 
    the United Kingdom discussed the 
    risks of AI-led automation; those 
    in Japan considered the necessity 
    of safeguarding human rights in 
    the face of AI; and those in Zambia
    Score  : 0.65
    Matches: generation of text, image, and code unimagined a decade ago, and they outperform the state of the art on many 
    benchmarks, old and new. However, they are prone to hallucination, routinely biased, and can be tricked into 
    serving nefarious aims, highlighting the complicated ethical challenges associated with their deployment.
    Although 2022 was the first year in a decade where private AI investment decreased, AI is still a topic of great
    

### Extract the context from the matching results for an LLM query



```python
context = "".join(extract_matches(results))
```

### Construct our query 
Plus the matches returned from the vector db for the LLM model to finalize
the response

### Step 3: Create an Anthropic client instance

Using our client factory method


```python
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL")
print(f"Using MODEL={MODEL}; base={'Anthropic'}")
```

    Using MODEL=claude-3-opus-20240229; base=Anthropic
    


```python
client_factory = ClientFactory()
client_type = "anthropic"
client_factory.register_client(client_type, Anthropic)
client_kwargs = {"api_key": anthropic_api_key}
```


```python
# create the client
client = client_factory.create_client(client_type, **client_kwargs)
```

### Step 4: Create system and user prompt for the LLM model


```python
system_content = """You are master of all knowledge, and a helpful sage.
                        You must summarize content given to you by drawing from your vast
                        knowledge about history, literature, science, social science, philosophy, religion, economics, 
                        sports, etc. Do not make up any responses. Only provide information that is true and verifiable
                        and use the given context to provide the response.
                     """
    
user_content = f"""What are the key takeaways for AI in 2023 from the HAI_AI Index Report_2023?,
                        given the {context}. Only provide information that is true and verifiable
                        and use the given context to provide the response.
                     """
    
```

### Step 5: Send the query + context to the LLM model
This is the final step in the diagram above where we
take the matching documents that our Pineconde 


```python
response = get_commpletion(client, MODEL, system_content, user_content)
response = response.replace("```", "")
print(f"\n{BOLD_BEGIN}Prompt:{BOLD_END} {user_content}")
print(f"\n{BOLD_BEGIN}Answer:{BOLD_END} {response}")
```

    
    [1mPrompt:[0m What are the key takeaways for AI in 2023 from the HAI_AI Index Report_2023?,
                            given the Figure 4.3.20advertising and marketing (8.8%) (Figure 4.3.20). 
    Compared to 2018, some of the less prevalent 
    AI-related themes in 2022 included deep learning 
    (4.8%), autonomous vehicles (3.1%), and data 
    storage and management (3.0%).broader range of societal actors. This year’s AI Index paints a picture of where we are so far with AI, in order to 
    highlight what might await us in the future.
    Jack Clark and Ray PerraultTable of Contents
     267
    Artificial Intelligence
    Index Report 2023 Chapter 6 PreviewIn the last 10 years, AI governance discussions have accelerated, resulting in numerous policy proposals in various legislative bodies. This 
    section begins by exploring the legislative initiatives related to AI that have been suggested or enacted in different countries and regions,increased nearly 6.5 times since 2016.When it comes to AI, 
    policymakers have  
    a lot of thoughts.   
    A qualitative analysis of the 
    parliamentary proceedings of a 
    diverse group of nations reveals 
    that policymakers think about AI 
    from a wide range of perspectives. 
    For example, in 2022, legislators in 
    the United Kingdom discussed the 
    risks of AI-led automation; those 
    in Japan considered the necessity 
    of safeguarding human rights in 
    the face of AI; and those in Zambiageneration of text, image, and code unimagined a decade ago, and they outperform the state of the art on many 
    benchmarks, old and new. However, they are prone to hallucination, routinely biased, and can be tricked into 
    serving nefarious aims, highlighting the complicated ethical challenges associated with their deployment.
    Although 2022 was the first year in a decade where private AI investment decreased, AI is still a topic of great. Only provide information that is true and verifiable
                            and use the given context to provide the response.
                         
    
    [1mAnswer:[0m Based on the excerpts provided from the HAI AI Index Report 2023, some key takeaways about AI in 2023 include:
    
    1. AI governance discussions and legislative initiatives related to AI have accelerated significantly in the last 10 years across many countries. Parliamentary proceedings show policymakers are thinking about AI from diverse perspectives like automation risks, safeguarding human rights, and economic impacts.
    
    2. Large language models have made remarkable progress, enabling generation of text, image and code at levels unimagined a decade ago. However, they still have issues like hallucination, bias, and potential for misuse, highlighting ethical challenges in deploying them.
    
    3. While 2022 saw the first decrease in private AI investment in a decade, AI remains a major focus of interest and investment. The number of AI-related publications increased nearly 6.5 times since 2016, showing strong ongoing research activity.
    
    4. Compared to 2018, some less prevalent AI topics in 2022 included deep learning, autonomous vehicles, and data storage/management. Advertising and marketing made up 8.8% of AI-related themes in 2022.
    
    So in summary, the report depicts accelerating governance efforts, remarkable technical progress yet ongoing challenges with language models, sustained investment and research interest, and some shifting focus in AI application areas in recent years. The overall picture is of a transformative technology that is advancing rapidly but also raising important policy questions.
    
