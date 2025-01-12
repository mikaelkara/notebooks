# Contextual Retrieval

In this notebook we will showcase how you can implement Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) using LangChain. Contextual Retrieval addresses the conundrum of traditional RAG approaches by prepending chunk-specific explanatory context to each chunk before embedding.

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F2496e7c6fedd7ffaa043895c23a4089638b0c21b-3840x2160.png&w=3840&q=75)


```python
import logging
import os

logging.disable(level=logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

os.environ["AZURE_OPENAI_API_KEY"] = "<YOUR_AZURE_OPENAI_API_KEY>"
os.environ["AZURE_OPENAI_ENDPOINT"] = "<YOUR_AZURE_OPENAI_ENDPOINT>"
os.environ["COHERE_API_KEY"] = "<YOUR_COHERE_API_KEY>"
```


```python
!pip install -q langchain langchain-openai langchain-community faiss-cpu rank_bm25 langchain-cohere 
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.2[0m[39;49m -> [0m[32;49m24.3.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m



```python
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever
from langchain.vectorstores import FAISS
from langchain_cohere import CohereRerank
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
```

## Download Data

We will use `Paul Graham Essay` dataset.


```python
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O './paul_graham_essay.txt'
```

    --2024-11-04 20:18:38--  https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 2606:50c0:8003::154, 2606:50c0:8001::154, 2606:50c0:8002::154, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|2606:50c0:8003::154|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 75042 (73K) [text/plain]
    Saving to: ‘./paul_graham_essay.txt’
    
    ./paul_graham_essay 100%[===================>]  73.28K  --.-KB/s    in 0.04s   
    
    2024-11-04 20:18:38 (2.02 MB/s) - ‘./paul_graham_essay.txt’ saved [75042/75042]
    


## Setup LLM and Embedding model


```python
llm = AzureChatOpenAI(
    deployment_name="gpt-4-32k-0613",
    openai_api_version="2023-08-01-preview",
    temperature=0.0,
)

embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
    api_version="2023-08-01-preview",
)
```

## Load Data


```python
loader = TextLoader("./paul_graham_essay.txt")
documents = loader.load()
WHOLE_DOCUMENT = documents[0].page_content
```

## Prompts for creating context for each chunk

We will use the following prompts to create chunk-specific explanatory context to each chunk before embedding.


```python
prompt_document = PromptTemplate(
    input_variables=["WHOLE_DOCUMENT"], template="{WHOLE_DOCUMENT}"
)
prompt_chunk = PromptTemplate(
    input_variables=["CHUNK_CONTENT"],
    template="Here is the chunk we want to situate within the whole document\n\n{CHUNK_CONTENT}\n\n"
    "Please give a short succinct context to situate this chunk within the overall document for "
    "the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.",
)
```

## Retrievers


```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever


def split_text(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=200)
    doc_chunks = text_splitter.create_documents(texts)
    for i, doc in enumerate(doc_chunks):
        # Append a new Document object with the appropriate doc_id
        doc.metadata = {"doc_id": f"doc_{i}"}
    return doc_chunks


def create_embedding_retriever(documents_):
    vector_store = FAISS.from_documents(documents_, embedding=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 4})


def create_bm25_retriever(documents_):
    retriever = BM25Retriever.from_documents(documents_, language="english")
    return retriever


# Function to create a combined embedding and BM25 retriever with reranker
class EmbeddingBM25RerankerRetriever:
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        reranker: BaseDocumentCompressor,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker

    def invoke(self, query: str):
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        combined_docs = vector_docs + [
            doc for doc in bm25_docs if doc not in vector_docs
        ]

        reranked_docs = self.reranker.compress_documents(combined_docs, query)
        return reranked_docs
```

### Non-contextual retrievers


```python
chunks = split_text([WHOLE_DOCUMENT])

embedding_retriever = create_embedding_retriever(chunks)

# Define a BM25 retriever
bm25_retriever = create_bm25_retriever(chunks)

reranker = CohereRerank(top_n=3, model="rerank-english-v2.0")

# Create combined retriever
embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
    vector_retriever=embedding_retriever,
    bm25_retriever=bm25_retriever,
    reranker=reranker,
)
```

### Contextual Retrievers


```python
import tqdm as tqdm
from langchain.docstore.document import Document


def create_contextual_chunks(chunks_):
    # uses a llm to add context to each chunk given the prompts defined above
    contextual_documents = []
    for chunk in tqdm.tqdm(chunks_):
        context = prompt_document.format(WHOLE_DOCUMENT=WHOLE_DOCUMENT)
        chunk_context = prompt_chunk.format(CHUNK_CONTENT=chunk)
        llm_response = llm.invoke(context + chunk_context).content
        page_content = f"""Text: {chunk.page_content}\n\n\nContext: {llm_response}"""
        doc = Document(page_content=page_content, metadata=chunk.metadata)
        contextual_documents.append(doc)
    return contextual_documents


contextual_documents = create_contextual_chunks(chunks)
```

    100%|██████████| 21/21 [01:50<00:00,  5.26s/it]
    


```python
print(contextual_documents[1].page_content, "------------", chunks[1].page_content)
```

    Text: I couldn't have put this into words when I was 18. All I knew at the time was that I kept taking philosophy courses and they kept being boring. So I decided to switch to AI.
    
    AI was in the air in the mid 1980s, but there were two things especially that made me want to work on it: a novel by Heinlein called The Moon is a Harsh Mistress, which featured an intelligent computer called Mike, and a PBS documentary that showed Terry Winograd using SHRDLU. I haven't tried rereading The Moon is a Harsh Mistress, so I don't know how well it has aged, but when I read it I was drawn entirely into its world. It seemed only a matter of time before we'd have Mike, and when I saw Winograd using SHRDLU, it seemed like that time would be a few years at most. All you had to do was teach SHRDLU more words.
    
    There weren't any classes in AI at Cornell then, not even graduate classes, so I started trying to teach myself. Which meant learning Lisp, since in those days Lisp was regarded as the language of AI. The commonly used programming languages then were pretty primitive, and programmers' ideas correspondingly so. The default language at Cornell was a Pascal-like language called PL/I, and the situation was similar elsewhere. Learning Lisp expanded my concept of a program so fast that it was years before I started to have a sense of where the new limits were. This was more like it; this was what I had expected college to do. It wasn't happening in a class, like it was supposed to, but that was ok. For the next couple years I was on a roll. I knew what I was going to do.
    
    For my undergraduate thesis, I reverse-engineered SHRDLU. My God did I love working on that program. It was a pleasing bit of code, but what made it even more exciting was my belief — hard to imagine now, but not unique in 1985 — that it was already climbing the lower slopes of intelligence.
    
    I had gotten into a program at Cornell that didn't make you choose a major. You could take whatever classes you liked, and choose whatever you liked to put on your degree. I of course chose "Artificial Intelligence." When I got the actual physical diploma, I was dismayed to find that the quotes had been included, which made them read as scare-quotes. At the time this bothered me, but now it seems amusingly accurate, for reasons I was about to discover.
    
    I applied to 3 grad schools: MIT and Yale, which were renowned for AI at the time, and Harvard, which I'd visited because Rich Draves went there, and was also home to Bill Woods, who'd invented the type of parser I used in my SHRDLU clone. Only Harvard accepted me, so that was where I went.
    
    I don't remember the moment it happened, or if there even was a specific moment, but during the first year of grad school I realized that AI, as practiced at the time, was a hoax. By which I mean the sort of AI in which a program that's told "the dog is sitting on the chair" translates this into some formal representation and adds it to the list of things it knows.
    
    What these programs really showed was that there's a subset of natural language that's a formal language. But a very proper subset. It was clear that there was an unbridgeable gap between what they could do and actually understanding natural language. It was not, in fact, simply a matter of teaching SHRDLU more words. That whole way of doing AI, with explicit data structures representing concepts, was not going to work. Its brokenness did, as so often happens, generate a lot of opportunities to write papers about various band-aids that could be applied to it, but it was never going to get us Mike.
    
    
    Context: This section of the document discusses the author's journey from studying philosophy to switching to AI during his undergraduate years at Cornell University. He talks about his fascination with AI, his self-learning process, and his undergraduate thesis on reverse-engineering SHRDLU. He also discusses his decision to apply to grad schools, his acceptance at Harvard, and his eventual realization that the AI practices of that time were not going to work. ------------ I couldn't have put this into words when I was 18. All I knew at the time was that I kept taking philosophy courses and they kept being boring. So I decided to switch to AI.
    
    AI was in the air in the mid 1980s, but there were two things especially that made me want to work on it: a novel by Heinlein called The Moon is a Harsh Mistress, which featured an intelligent computer called Mike, and a PBS documentary that showed Terry Winograd using SHRDLU. I haven't tried rereading The Moon is a Harsh Mistress, so I don't know how well it has aged, but when I read it I was drawn entirely into its world. It seemed only a matter of time before we'd have Mike, and when I saw Winograd using SHRDLU, it seemed like that time would be a few years at most. All you had to do was teach SHRDLU more words.
    
    There weren't any classes in AI at Cornell then, not even graduate classes, so I started trying to teach myself. Which meant learning Lisp, since in those days Lisp was regarded as the language of AI. The commonly used programming languages then were pretty primitive, and programmers' ideas correspondingly so. The default language at Cornell was a Pascal-like language called PL/I, and the situation was similar elsewhere. Learning Lisp expanded my concept of a program so fast that it was years before I started to have a sense of where the new limits were. This was more like it; this was what I had expected college to do. It wasn't happening in a class, like it was supposed to, but that was ok. For the next couple years I was on a roll. I knew what I was going to do.
    
    For my undergraduate thesis, I reverse-engineered SHRDLU. My God did I love working on that program. It was a pleasing bit of code, but what made it even more exciting was my belief — hard to imagine now, but not unique in 1985 — that it was already climbing the lower slopes of intelligence.
    
    I had gotten into a program at Cornell that didn't make you choose a major. You could take whatever classes you liked, and choose whatever you liked to put on your degree. I of course chose "Artificial Intelligence." When I got the actual physical diploma, I was dismayed to find that the quotes had been included, which made them read as scare-quotes. At the time this bothered me, but now it seems amusingly accurate, for reasons I was about to discover.
    
    I applied to 3 grad schools: MIT and Yale, which were renowned for AI at the time, and Harvard, which I'd visited because Rich Draves went there, and was also home to Bill Woods, who'd invented the type of parser I used in my SHRDLU clone. Only Harvard accepted me, so that was where I went.
    
    I don't remember the moment it happened, or if there even was a specific moment, but during the first year of grad school I realized that AI, as practiced at the time, was a hoax. By which I mean the sort of AI in which a program that's told "the dog is sitting on the chair" translates this into some formal representation and adds it to the list of things it knows.
    
    What these programs really showed was that there's a subset of natural language that's a formal language. But a very proper subset. It was clear that there was an unbridgeable gap between what they could do and actually understanding natural language. It was not, in fact, simply a matter of teaching SHRDLU more words. That whole way of doing AI, with explicit data structures representing concepts, was not going to work. Its brokenness did, as so often happens, generate a lot of opportunities to write papers about various band-aids that could be applied to it, but it was never going to get us Mike.
    


```python
contextual_embedding_retriever = create_embedding_retriever(contextual_documents)

contextual_bm25_retriever = create_bm25_retriever(contextual_documents)

contextual_embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
    vector_retriever=contextual_embedding_retriever,
    bm25_retriever=contextual_bm25_retriever,
    reranker=reranker,
)
```

## Generate Question-Context pairs


```python
import json
import re
import uuid
import warnings
from typing import Dict, List, Tuple

from pydantic import BaseModel
from tqdm import tqdm

# Prompt to generate questions
DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and no prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
"""


class QuestionContextEvalDataset(BaseModel):
    """Embedding QA Dataset.
    Args:
        queries (Dict[str, str]): Dict id -> query.
        corpus (Dict[str, str]): Dict id -> string.
        relevant_docs (Dict[str, List[str]]): Dict query id -> list of doc ids.
    """

    queries: Dict[str, str]  # dict id -> query
    corpus: Dict[str, str]  # dict id -> string
    relevant_docs: Dict[str, List[str]]  # query id -> list of doc ids
    mode: str = "text"

    @property
    def query_docid_pairs(self) -> List[Tuple[str, List[str]]]:
        """Get query, relevant doc ids."""
        return [
            (query, self.relevant_docs[query_id])
            for query_id, query in self.queries.items()
        ]

    def save_json(self, path: str) -> None:
        """Save json."""
        with open(path, "w") as f:
            json.dump(self.dict(), f, indent=4)

    @classmethod
    def from_json(cls, path: str) -> "QuestionContextEvalDataset":
        """Load json."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def generate_question_context_pairs(
    documents: List[Document],
    llm,
    qa_generate_prompt_tmpl: str = DEFAULT_QA_GENERATE_PROMPT_TMPL,
    num_questions_per_chunk: int = 2,
) -> QuestionContextEvalDataset:
    """Generate evaluation dataset using watsonx LLM and a set of chunks with their chunk_ids

    Args:
        documents (List[Document]): chunks of data with chunk_id
        llm: LLM used for generating questions
        qa_generate_prompt_tmpl (str): prompt template used for generating questions
        num_questions_per_chunk (int): number of questions generated per chunk

    Returns:
        List[Documents]: List of langchain document objects with page content and metadata
    """
    doc_dict = {doc.metadata["doc_id"]: doc.page_content for doc in documents}
    queries = {}
    relevant_docs = {}
    for doc_id, text in tqdm(doc_dict.items()):
        query = qa_generate_prompt_tmpl.format(
            context_str=text, num_questions_per_chunk=num_questions_per_chunk
        )
        response = llm.invoke(query).content
        result = re.split(r"\n+", response.strip())
        print(result)
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0][
            :num_questions_per_chunk
        ]

        num_questions_generated = len(questions)
        if num_questions_generated < num_questions_per_chunk:
            warnings.warn(
                f"Fewer questions generated ({num_questions_generated}) "
                f"than requested ({num_questions_per_chunk})."
            )
        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [doc_id]
    # construct dataset
    return QuestionContextEvalDataset(
        queries=queries, corpus=doc_dict, relevant_docs=relevant_docs
    )
```


```python
qa_pairs = generate_question_context_pairs(chunks, llm, num_questions_per_chunk=2)
```

      5%|▍         | 1/21 [00:02<00:59,  2.98s/it]

    ["1. Describe the author's early experiences with programming on the IBM 1401. What were some of the challenges he faced and how did the limitations of the technology at the time influence his programming?", '2. The author initially intended to study philosophy in college but eventually switched to AI. Based on the context, explain the reasons behind this change in his academic direction.']
    

     10%|▉         | 2/21 [00:06<00:58,  3.10s/it]

    ['1. In the context, the author mentions two specific inspirations that led him to pursue AI. Identify these inspirations and explain how they influenced his decision.', '2. The author initially believed that teaching SHRDLU more words would lead to the development of AI. However, he later realized this approach was flawed. Discuss his initial belief and the realization that led him to change his perspective.']
    

     14%|█▍        | 3/21 [00:09<01:00,  3.37s/it]

    ["1. In the context, the author discusses his interest in both computer science and art. Discuss how the author's perspective on the longevity and impact of these two fields influenced his career decisions. Provide specific examples from the text.", '2. The author mentions his book "On Lisp" and his experience of writing it. Based on the context, what challenges did he face while writing this book and how did it contribute to his understanding of Lisp hacking?']
    

     19%|█▉        | 4/21 [00:12<00:54,  3.21s/it]

    ['1. In the context, the author mentions his decision to write his dissertation on the applications of continuations. What reasons does he give for this choice and how does he reflect on this decision in retrospect?', "2. Describe the author's experience at the Accademia di Belli Arti in Florence. How does he portray the teaching and learning environment at the institution, and what activities did the students engage in during their time there?"]
    

     24%|██▍       | 5/21 [00:15<00:49,  3.11s/it]

    ['1. In the context of the document, the author discusses the process of painting still lives and how it differs from painting people. Can you explain this difference and discuss how the author uses this process to create a more realistic representation of the subject?', '2. The author worked at a company called Interleaf, which had incorporated a scripting language into their software. Discuss the challenges the author faced in this job and how it influenced his understanding of programming and software development.']
    

     29%|██▊       | 6/21 [00:18<00:46,  3.11s/it]

    ['1. Based on the author\'s experience at Interleaf, explain the concept of "the low end eats the high end" and how it influenced his later ventures like Viaweb and Y Combinator. ', '2. Discuss the author\'s perspective on the teaching approach at RISD, particularly in the painting department, and how it contrasts with his expectations. What does he mean by "signature style" and how does it relate to the art market?']
    

     33%|███▎      | 7/21 [00:21<00:40,  2.87s/it]

    ["1. In the context, the author mentions his decision to write a book on Lisp. Discuss the author's motivations behind this decision and how it relates to his financial concerns and artistic pursuits.", '2. Analyze the author\'s initial business idea of putting art galleries online. Why did it fail according to the author? How did this failure lead to the realization of building an "internet storefront"?']
    

     38%|███▊      | 8/21 [00:23<00:36,  2.80s/it]

    ['1. "Describe the initial challenges faced by the authors while developing the software for online stores and how they overcame them. Also, explain the significance of their idea of running the software on the server."', '2. "Discuss the role of aesthetics and high production values in the success of an online store as mentioned in the context. How did the author\'s background in art contribute to the development of their online store builder software?"']
    

     43%|████▎     | 9/21 [00:26<00:33,  2.76s/it]

    ['1. In the context of the document, explain the roles and contributions of Robert and Trevor in the development of the ecommerce software. How did their unique perspectives and skills contribute to the project?', "2. Based on the author's experiences and observations, discuss the challenges and learnings they encountered in the early stages of ecommerce, particularly in relation to user acquisition and understanding retail. Use specific examples from the text."]
    

     48%|████▊     | 10/21 [00:30<00:32,  2.98s/it]

    ["1. Discuss the significance of growth rate in the success of a startup, as illustrated in the context of Viaweb's journey. How did the author's understanding of this concept evolve over time?", "2. Analyze the author's transition from running a startup to working at Yahoo. How did this change impact his personal and professional life, and what led to his decision to leave Yahoo in the summer of 1999?"]
    

     52%|█████▏    | 11/21 [00:32<00:29,  2.95s/it]

    ['1. Based on the context, discuss the reasons and circumstances that led the author to leave his job at Yahoo and pursue painting. How did his experiences in California and New York influence his decision to return to the tech industry?', "2. Analyze the author's idea of building a web app for making web apps. How did he envision this idea to be the future of web applications and what challenges did he face in trying to implement this idea?"]
    

     57%|█████▋    | 12/21 [00:35<00:26,  2.93s/it]

    ["1. In the context, the author mentions the creation of a new dialect of Lisp called Arc. Discuss the reasons behind the author's decision to create this new dialect and how it was intended to be used in the development of the Aspra project.", "2. The author discusses a significant shift in the publishing industry due to the advent of the internet. Explain how this shift impacted the author's perspective on writing and publishing, and discuss the implications it had for the generation of essays."]
    

     62%|██████▏   | 13/21 [00:38<00:23,  2.92s/it]

    ["1. Based on the author's experiences, discuss the significance of working on projects that lack prestige and how it can indicate the presence of genuine interest and potential for discovery. Provide examples from the text to support your answer.", "2. Analyze the author's approach to writing essays and giving talks. How does the author use the prospect of public speaking to stimulate creativity and ensure the content is valuable to the audience? Use specific instances from the text in your response."]
    

     67%|██████▋   | 14/21 [00:41<00:20,  2.92s/it]

    ['1. In the context, the author discusses the formation of Y Combinator. Explain how the unique batch model of Y Combinator was discovered and why it was considered distinctive in the investment world during that time.', "2. Based on the context, discuss the author's initial hesitation towards angel investing and how his experiences and collaborations led to the creation of his own investment firm. What were some of the novel approaches they took due to their lack of knowledge about being angel investors?"]
    

     71%|███████▏  | 15/21 [00:44<00:17,  2.90s/it]

    ['1. "Discuss the initial structure and strategy of the Summer Founders Program, including its funding model and the benefits it provided to the participating startups. How did this model contribute to the growth and success of Y Combinator?"', '2. "Explain the evolution of Hacker News from its initial concept as Startup News to its current form. What was the rationale behind the changes made and how did it align with the overall objectives of Y Combinator?"']
    

     76%|███████▌  | 16/21 [00:47<00:14,  2.83s/it]

    ["1. In the context provided, the author compares his stress from Hacker News (HN) to a specific situation. Can you explain this analogy and how it reflects his feelings towards HN's impact on his work at Y Combinator (YC)?", '2. The author mentions a personal event that led to his decision to hand over YC to someone else. What was this event and how did it influence his decision?']
    

     81%|████████  | 17/21 [00:49<00:10,  2.65s/it]

    ['1. Discuss the transition of leadership at YC from the original founders to Sam Altman. What were the reasons behind this change and how was the transition process managed?', '2. Explain the origins and unique characteristics of Lisp as a programming language. How did its initial purpose as a formal model of computation contribute to its power and elegance?']
    

     86%|████████▌ | 18/21 [00:51<00:07,  2.54s/it]

    ['1. "Discuss the challenges faced by Paul Graham in developing the programming language Bel, and how he overcame them. Provide specific examples from the text."', '2. "Explain the significance of McCarthy\'s axiomatic approach in the development of Lisp and Bel. How did the evolution of computer power over time influence this process?"']
    

     90%|█████████ | 19/21 [00:54<00:05,  2.65s/it]

    ["1. In the context, the author mentions a transition from batch processing to microcomputers, skipping a step in the evolution of computers. What was this skipped step and how did it impact the author's perception of microcomputers?", '2. The author discusses his experience living in Florence and walking to the Accademia. Describe the route he took and the various conditions he experienced during his walks. How did this experience contribute to his understanding and appreciation of the city?']
    

     95%|█████████▌| 20/21 [00:57<00:02,  2.64s/it]

    ["1. In the context of the document, explain the significance of the name change from Cambridge Seed to Y Combinator and the choice of the color orange for the logo. What does this reflect about the organization's approach and target audience?", '2. Discuss the author\'s perspective on the term "deal flow" in relation to startups. How does this view align with the purpose of Y Combinator?']
    

    100%|██████████| 21/21 [01:00<00:00,  2.86s/it]

    ["1. In the given context, the author uses the concept of space aliens to differentiate between 'invented' and 'discovered'. Explain this concept in detail and discuss how it applies to the Pythagorean theorem and Lisp in McCarthy's 1960 paper.", '2. The author mentions a significant change in their personal and professional life, which is leaving YC and not working with Jessica anymore. Discuss the metaphor used by the author to describe this change and explain its significance.']
    

    
    

## Evaluate


```python
def compute_hit_rate(expected_ids, retrieved_ids):
    """
    Args:
    expected_ids List[str]: The ground truth doc_id
    retrieved_ids List[str]: The doc_id from retrieved chunks

    Returns:
        float: hit rate as a decimal
    """
    if retrieved_ids is None or expected_ids is None:
        raise ValueError("Retrieved ids and expected ids must be provided")
    is_hit = any(id in expected_ids for id in retrieved_ids)
    return 1.0 if is_hit else 0.0


def compute_mrr(expected_ids, retrieved_ids):
    """
    Args:
    expected_ids List[str]: The ground truth doc_id
    retrieved_ids List[str]: The doc_id from retrieved chunks

    Returns:
        float: MRR score as a decimal
    """
    if retrieved_ids is None or expected_ids is None:
        raise ValueError("Retrieved ids and expected ids must be provided")
    for i, id in enumerate(retrieved_ids):
        if id in expected_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg(expected_ids, retrieved_ids):
    """
    Args:
    expected_ids List[str]: The ground truth doc_id
    retrieved_ids List[str]: The doc_id from retrieved chunks

    Returns:
        float: nDCG score as a decimal
    """
    if retrieved_ids is None or expected_ids is None:
        raise ValueError("Retrieved ids and expected ids must be provided")
    dcg = 0.0
    idcg = 0.0
    for i, id in enumerate(retrieved_ids):
        if id in expected_ids:
            dcg += 1.0 / (i + 1)
        idcg += 1.0 / (i + 1)
    return dcg / idcg
```


```python
import numpy as np
import pandas as pd


def extract_queries(dataset):
    values = []
    for value in dataset.queries.values():
        values.append(value)
    return values


def extract_doc_ids(documents_):
    doc_ids = []
    for doc in documents_:
        doc_ids.append(f"{doc.metadata['doc_id']}")
    return doc_ids


def evaluate(retriever, dataset):
    mrr_result = []
    hit_rate_result = []
    ndcg_result = []

    # Loop over dataset
    for i in tqdm(range(len(dataset.queries))):
        context = retriever.invoke(extract_queries(dataset)[i])

        expected_ids = dataset.relevant_docs[list(dataset.queries.keys())[i]]
        retrieved_ids = extract_doc_ids(context)
        # compute metrics
        mrr = compute_mrr(expected_ids=expected_ids, retrieved_ids=retrieved_ids)
        hit_rate = compute_hit_rate(
            expected_ids=expected_ids, retrieved_ids=retrieved_ids
        )
        ndgc = compute_ndcg(expected_ids=expected_ids, retrieved_ids=retrieved_ids)
        # append results
        mrr_result.append(mrr)
        hit_rate_result.append(hit_rate)
        ndcg_result.append(ndgc)

    array2D = np.array([mrr_result, hit_rate_result, ndcg_result])
    mean_results = np.mean(array2D, axis=1)
    results_df = pd.DataFrame(mean_results)
    results_df.index = ["MRR", "Hit Rate", "nDCG"]
    return results_df
```


```python
embedding_bm25_rerank_results = evaluate(embedding_bm25_retriever_rerank, qa_pairs)
```

    100%|██████████| 42/42 [00:19<00:00,  2.21it/s]
    


```python
contextual_embedding_bm25_rerank_results = evaluate(
    contextual_embedding_bm25_retriever_rerank, qa_pairs
)
```

    100%|██████████| 42/42 [00:17<00:00,  2.36it/s]
    


```python
embedding_retriever_results = evaluate(embedding_retriever, qa_pairs)
```

    100%|██████████| 42/42 [00:02<00:00, 14.96it/s]
    


```python
contextual_embedding_retriever_results = evaluate(
    contextual_embedding_retriever, qa_pairs
)
```

    100%|██████████| 42/42 [00:02<00:00, 15.57it/s]
    


```python
bm25_results = evaluate(bm25_retriever, qa_pairs)
```

    100%|██████████| 42/42 [00:00<00:00, 2934.59it/s]
    


```python
contextual_bm25_results = evaluate(contextual_bm25_retriever, qa_pairs)
```

    100%|██████████| 42/42 [00:00<00:00, 3022.46it/s]
    


```python
def display_results(name, eval_results):
    """Display results from evaluate."""

    metrics = ["MRR", "Hit Rate", "nDCG"]

    columns = {
        "Retrievers": [name],
        **{metric: val for metric, val in zip(metrics, eval_results.values)},
    }

    metric_df = pd.DataFrame(columns)

    return metric_df


pd.concat(
    [
        display_results("Embedding Retriever", embedding_retriever_results),
        display_results("BM25 Retriever", bm25_results),
        display_results(
            "Embedding + BM25 Retriever + Reranker",
            embedding_bm25_rerank_results,
        ),
    ],
    ignore_index=True,
    axis=0,
)
```




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
      <th>Retrievers</th>
      <th>MRR</th>
      <th>Hit Rate</th>
      <th>nDCG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Embedding Retriever</td>
      <td>0.797619</td>
      <td>0.904762</td>
      <td>0.382857</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BM25 Retriever</td>
      <td>0.865079</td>
      <td>0.928571</td>
      <td>0.415238</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Embedding + BM25 Retriever + Reranker</td>
      <td>0.960317</td>
      <td>1.000000</td>
      <td>0.523810</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat(
    [
        display_results(
            "Contextual Embedding Retriever", contextual_embedding_retriever_results
        ),
        display_results("Contextual BM25 Retriever", contextual_bm25_results),
        display_results(
            "Contextual Embedding + BM25 Retriever + Reranker",
            contextual_embedding_bm25_rerank_results,
        ),
    ],
    ignore_index=True,
    axis=0,
)
```




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
      <th>Retrievers</th>
      <th>MRR</th>
      <th>Hit Rate</th>
      <th>nDCG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Contextual Embedding Retriever</td>
      <td>0.785714</td>
      <td>0.904762</td>
      <td>0.377143</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Contextual BM25 Retriever</td>
      <td>0.908730</td>
      <td>0.976190</td>
      <td>0.436190</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Contextual Embedding + BM25 Retriever + Reranker</td>
      <td>0.984127</td>
      <td>1.000000</td>
      <td>0.536797</td>
    </tr>
  </tbody>
</table>
</div>


