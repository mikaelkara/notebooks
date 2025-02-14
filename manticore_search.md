# ManticoreSearch VectorStore

[ManticoreSearch](https://manticoresearch.com/) is an open-source search engine that offers fast, scalable, and user-friendly capabilities. Originating as a fork of [Sphinx Search](http://sphinxsearch.com/), it has evolved to incorporate modern search engine features and improvements. ManticoreSearch distinguishes itself with its robust performance and ease of integration into various applications.

ManticoreSearch has recently introduced [vector search capabilities](https://manual.manticoresearch.com/dev/Searching/KNN), starting with search engine version 6.2 and only with [manticore-columnar-lib](https://github.com/manticoresoftware/columnar) package installed. This feature is a considerable advancement, allowing for the execution of searches based on vector similarity.

As of now, the vector search functionality is only accessible in the developmental (dev) versions of the search engine. Consequently, it is imperative to employ a developmental [manticoresearch-dev](https://pypi.org/project/manticoresearch-dev/) Python client for utilizing this feature effectively.

## Setting up environments

Starting Docker-container with ManticoreSearch and installing manticore-columnar-lib package (optional)


```python
import time

# Start container
containers = !docker ps --filter "name=langchain-manticoresearch-server" -q
if len(containers) == 0:
    !docker run -d -p 9308:9308 --name langchain-manticoresearch-server manticoresearch/manticore:dev
    time.sleep(20)  # Wait for the container to start up

# Get ID of container
container_id = containers[0]

# Install manticore-columnar-lib package as root user
!docker exec -it --user 0 {container_id} apt-get update
!docker exec -it --user 0 {container_id} apt-get install -y manticore-columnar-lib

# Restart container
!docker restart {container_id}
```

    Get:1 http://repo.manticoresearch.com/repository/manticoresearch_jammy_dev jammy InRelease [3525 kB]
    Get:2 http://archive.ubuntu.com/ubuntu jammy InRelease [270 kB]            
    Get:3 http://security.ubuntu.com/ubuntu jammy-security InRelease [110 kB]      
    Get:4 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [119 kB]        
    Get:5 http://security.ubuntu.com/ubuntu jammy-security/universe amd64 Packages [1074 kB]
    Get:6 http://archive.ubuntu.com/ubuntu jammy-backports InRelease [109 kB]      
    Get:7 http://archive.ubuntu.com/ubuntu jammy/universe amd64 Packages [17.5 MB] 
    Get:8 http://security.ubuntu.com/ubuntu jammy-security/main amd64 Packages [1517 kB]
    Get:9 http://security.ubuntu.com/ubuntu jammy-security/restricted amd64 Packages [1889 kB]
    Get:10 http://security.ubuntu.com/ubuntu jammy-security/multiverse amd64 Packages [44.6 kB]
    Get:11 http://archive.ubuntu.com/ubuntu jammy/restricted amd64 Packages [164 kB]
    Get:12 http://archive.ubuntu.com/ubuntu jammy/multiverse amd64 Packages [266 kB]
    Get:13 http://archive.ubuntu.com/ubuntu jammy/main amd64 Packages [1792 kB]    
    Get:14 http://archive.ubuntu.com/ubuntu jammy-updates/multiverse amd64 Packages [50.4 kB]
    Get:15 http://archive.ubuntu.com/ubuntu jammy-updates/restricted amd64 Packages [1927 kB]
    Get:16 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 Packages [1346 kB]
    Get:17 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 Packages [1796 kB]
    Get:18 http://archive.ubuntu.com/ubuntu jammy-backports/universe amd64 Packages [28.1 kB]
    Get:19 http://archive.ubuntu.com/ubuntu jammy-backports/main amd64 Packages [50.4 kB]
    Get:20 http://repo.manticoresearch.com/repository/manticoresearch_jammy_dev jammy/main amd64 Packages [5020 kB]
    Fetched 38.6 MB in 7s (5847 kB/s)                                              
    Reading package lists... Done
    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    The following NEW packages will be installed:
      manticore-columnar-lib
    0 upgraded, 1 newly installed, 0 to remove and 21 not upgraded.
    Need to get 1990 kB of archives.
    After this operation, 10.0 MB of additional disk space will be used.
    Get:1 http://repo.manticoresearch.com/repository/manticoresearch_jammy_dev jammy/main amd64 manticore-columnar-lib amd64 2.2.5-240217-a5342a1 [1990 kB]
    Fetched 1990 kB in 1s (1505 kB/s)                 
    debconf: delaying package configuration, since apt-utils is not installed
    Selecting previously unselected package manticore-columnar-lib.
    (Reading database ... 12260 files and directories currently installed.)
    Preparing to unpack .../manticore-columnar-lib_2.2.5-240217-a5342a1_amd64.deb ...
    Unpacking manticore-columnar-lib (2.2.5-240217-a5342a1) ...
    Setting up manticore-columnar-lib (2.2.5-240217-a5342a1) ...
    a546aec22291


Installing ManticoreSearch python client


```python
%pip install --upgrade --quiet manticoresearch-dev
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.2.1[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.
    

We want to use OpenAIEmbeddings so we have to get the OpenAI API Key.


```python
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import ManticoreSearch, ManticoreSearchSettings
from langchain_text_splitters import CharacterTextSplitter
```


```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../modules/paul_graham_essay.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = GPT4AllEmbeddings()
```

    Created a chunk of size 338, which is longer than the specified 100
    Created a chunk of size 508, which is longer than the specified 100
    Created a chunk of size 277, which is longer than the specified 100
    Created a chunk of size 777, which is longer than the specified 100
    Created a chunk of size 247, which is longer than the specified 100
    Created a chunk of size 228, which is longer than the specified 100
    Created a chunk of size 557, which is longer than the specified 100
    Created a chunk of size 587, which is longer than the specified 100
    Created a chunk of size 173, which is longer than the specified 100
    Created a chunk of size 622, which is longer than the specified 100
    Created a chunk of size 775, which is longer than the specified 100
    Created a chunk of size 292, which is longer than the specified 100
    Created a chunk of size 456, which is longer than the specified 100
    Created a chunk of size 291, which is longer than the specified 100
    Created a chunk of size 367, which is longer than the specified 100
    Created a chunk of size 604, which is longer than the specified 100
    Created a chunk of size 618, which is longer than the specified 100
    Created a chunk of size 340, which is longer than the specified 100
    Created a chunk of size 395, which is longer than the specified 100
    Created a chunk of size 321, which is longer than the specified 100
    Created a chunk of size 453, which is longer than the specified 100
    Created a chunk of size 354, which is longer than the specified 100
    Created a chunk of size 481, which is longer than the specified 100
    Created a chunk of size 233, which is longer than the specified 100
    Created a chunk of size 270, which is longer than the specified 100
    Created a chunk of size 305, which is longer than the specified 100
    Created a chunk of size 520, which is longer than the specified 100
    Created a chunk of size 289, which is longer than the specified 100
    Created a chunk of size 280, which is longer than the specified 100
    Created a chunk of size 417, which is longer than the specified 100
    Created a chunk of size 495, which is longer than the specified 100
    Created a chunk of size 602, which is longer than the specified 100
    Created a chunk of size 1004, which is longer than the specified 100
    Created a chunk of size 272, which is longer than the specified 100
    Created a chunk of size 1203, which is longer than the specified 100
    Created a chunk of size 844, which is longer than the specified 100
    Created a chunk of size 135, which is longer than the specified 100
    Created a chunk of size 306, which is longer than the specified 100
    Created a chunk of size 407, which is longer than the specified 100
    Created a chunk of size 910, which is longer than the specified 100
    Created a chunk of size 398, which is longer than the specified 100
    Created a chunk of size 674, which is longer than the specified 100
    Created a chunk of size 356, which is longer than the specified 100
    Created a chunk of size 474, which is longer than the specified 100
    Created a chunk of size 814, which is longer than the specified 100
    Created a chunk of size 530, which is longer than the specified 100
    Created a chunk of size 469, which is longer than the specified 100
    Created a chunk of size 489, which is longer than the specified 100
    Created a chunk of size 433, which is longer than the specified 100
    Created a chunk of size 603, which is longer than the specified 100
    Created a chunk of size 380, which is longer than the specified 100
    Created a chunk of size 354, which is longer than the specified 100
    Created a chunk of size 391, which is longer than the specified 100
    Created a chunk of size 772, which is longer than the specified 100
    Created a chunk of size 267, which is longer than the specified 100
    Created a chunk of size 571, which is longer than the specified 100
    Created a chunk of size 594, which is longer than the specified 100
    Created a chunk of size 458, which is longer than the specified 100
    Created a chunk of size 386, which is longer than the specified 100
    Created a chunk of size 417, which is longer than the specified 100
    Created a chunk of size 370, which is longer than the specified 100
    Created a chunk of size 402, which is longer than the specified 100
    Created a chunk of size 306, which is longer than the specified 100
    Created a chunk of size 173, which is longer than the specified 100
    Created a chunk of size 628, which is longer than the specified 100
    Created a chunk of size 321, which is longer than the specified 100
    Created a chunk of size 294, which is longer than the specified 100
    Created a chunk of size 689, which is longer than the specified 100
    Created a chunk of size 641, which is longer than the specified 100
    Created a chunk of size 473, which is longer than the specified 100
    Created a chunk of size 414, which is longer than the specified 100
    Created a chunk of size 585, which is longer than the specified 100
    Created a chunk of size 764, which is longer than the specified 100
    Created a chunk of size 502, which is longer than the specified 100
    Created a chunk of size 640, which is longer than the specified 100
    Created a chunk of size 507, which is longer than the specified 100
    Created a chunk of size 564, which is longer than the specified 100
    Created a chunk of size 707, which is longer than the specified 100
    Created a chunk of size 380, which is longer than the specified 100
    Created a chunk of size 615, which is longer than the specified 100
    Created a chunk of size 733, which is longer than the specified 100
    Created a chunk of size 277, which is longer than the specified 100
    Created a chunk of size 497, which is longer than the specified 100
    Created a chunk of size 625, which is longer than the specified 100
    Created a chunk of size 468, which is longer than the specified 100
    Created a chunk of size 289, which is longer than the specified 100
    Created a chunk of size 576, which is longer than the specified 100
    Created a chunk of size 297, which is longer than the specified 100
    Created a chunk of size 534, which is longer than the specified 100
    Created a chunk of size 427, which is longer than the specified 100
    Created a chunk of size 412, which is longer than the specified 100
    Created a chunk of size 381, which is longer than the specified 100
    Created a chunk of size 417, which is longer than the specified 100
    Created a chunk of size 244, which is longer than the specified 100
    Created a chunk of size 307, which is longer than the specified 100
    Created a chunk of size 528, which is longer than the specified 100
    Created a chunk of size 565, which is longer than the specified 100
    Created a chunk of size 487, which is longer than the specified 100
    Created a chunk of size 470, which is longer than the specified 100
    Created a chunk of size 332, which is longer than the specified 100
    Created a chunk of size 552, which is longer than the specified 100
    Created a chunk of size 427, which is longer than the specified 100
    Created a chunk of size 596, which is longer than the specified 100
    Created a chunk of size 192, which is longer than the specified 100
    Created a chunk of size 403, which is longer than the specified 100
    Created a chunk of size 255, which is longer than the specified 100
    Created a chunk of size 1025, which is longer than the specified 100
    Created a chunk of size 438, which is longer than the specified 100
    Created a chunk of size 900, which is longer than the specified 100
    Created a chunk of size 250, which is longer than the specified 100
    Created a chunk of size 614, which is longer than the specified 100
    Created a chunk of size 635, which is longer than the specified 100
    Created a chunk of size 443, which is longer than the specified 100
    Created a chunk of size 478, which is longer than the specified 100
    Created a chunk of size 473, which is longer than the specified 100
    Created a chunk of size 302, which is longer than the specified 100
    Created a chunk of size 549, which is longer than the specified 100
    Created a chunk of size 644, which is longer than the specified 100
    Created a chunk of size 402, which is longer than the specified 100
    Created a chunk of size 489, which is longer than the specified 100
    Created a chunk of size 551, which is longer than the specified 100
    Created a chunk of size 527, which is longer than the specified 100
    Created a chunk of size 563, which is longer than the specified 100
    Created a chunk of size 472, which is longer than the specified 100
    Created a chunk of size 511, which is longer than the specified 100
    Created a chunk of size 419, which is longer than the specified 100
    Created a chunk of size 245, which is longer than the specified 100
    Created a chunk of size 371, which is longer than the specified 100
    Created a chunk of size 484, which is longer than the specified 100
    Created a chunk of size 306, which is longer than the specified 100
    Created a chunk of size 190, which is longer than the specified 100
    Created a chunk of size 499, which is longer than the specified 100
    Created a chunk of size 480, which is longer than the specified 100
    Created a chunk of size 634, which is longer than the specified 100
    Created a chunk of size 611, which is longer than the specified 100
    Created a chunk of size 356, which is longer than the specified 100
    Created a chunk of size 478, which is longer than the specified 100
    Created a chunk of size 369, which is longer than the specified 100
    Created a chunk of size 526, which is longer than the specified 100
    Created a chunk of size 311, which is longer than the specified 100
    Created a chunk of size 181, which is longer than the specified 100
    Created a chunk of size 637, which is longer than the specified 100
    Created a chunk of size 219, which is longer than the specified 100
    Created a chunk of size 305, which is longer than the specified 100
    Created a chunk of size 409, which is longer than the specified 100
    Created a chunk of size 235, which is longer than the specified 100
    Created a chunk of size 302, which is longer than the specified 100
    Created a chunk of size 236, which is longer than the specified 100
    Created a chunk of size 209, which is longer than the specified 100
    Created a chunk of size 366, which is longer than the specified 100
    Created a chunk of size 277, which is longer than the specified 100
    Created a chunk of size 591, which is longer than the specified 100
    Created a chunk of size 232, which is longer than the specified 100
    Created a chunk of size 543, which is longer than the specified 100
    Created a chunk of size 199, which is longer than the specified 100
    Created a chunk of size 214, which is longer than the specified 100
    Created a chunk of size 263, which is longer than the specified 100
    Created a chunk of size 375, which is longer than the specified 100
    Created a chunk of size 221, which is longer than the specified 100
    Created a chunk of size 261, which is longer than the specified 100
    Created a chunk of size 203, which is longer than the specified 100
    Created a chunk of size 758, which is longer than the specified 100
    Created a chunk of size 271, which is longer than the specified 100
    Created a chunk of size 323, which is longer than the specified 100
    Created a chunk of size 275, which is longer than the specified 100
    

    bert_load_from_file: gguf version     = 2
    bert_load_from_file: gguf alignment   = 32
    bert_load_from_file: gguf data offset = 695552
    bert_load_from_file: model name           = BERT
    bert_load_from_file: model architecture   = bert
    bert_load_from_file: model file type      = 1
    bert_load_from_file: bert tokenizer vocab = 30522
    


```python
for d in docs:
    d.metadata = {"some": "metadata"}
settings = ManticoreSearchSettings(table="manticoresearch_vector_search_example")
docsearch = ManticoreSearch.from_documents(docs, embeddings, config=settings)

query = "Robert Morris is"
docs = docsearch.similarity_search(query)
print(docs)
```

    [Document(page_content='Computer Science is an uneasy alliance between two halves, theory and systems. The theory people prove things, and the systems people build things. I wanted to build things. I had plenty of respect for theory — indeed, a sneaking suspicion that it was the more admirable of the two halves — but building things seemed so much more exciting.', metadata={'some': 'metadata'}), Document(page_content="I applied to 3 grad schools: MIT and Yale, which were renowned for AI at the time, and Harvard, which I'd visited because Rich Draves went there, and was also home to Bill Woods, who'd invented the type of parser I used in my SHRDLU clone. Only Harvard accepted me, so that was where I went.", metadata={'some': 'metadata'}), Document(page_content='For my undergraduate thesis, I reverse-engineered SHRDLU. My God did I love working on that program. It was a pleasing bit of code, but what made it even more exciting was my belief — hard to imagine now, but not unique in 1985 — that it was already climbing the lower slopes of intelligence.', metadata={'some': 'metadata'}), Document(page_content="The problem with systems work, though, was that it didn't last. Any program you wrote today, no matter how good, would be obsolete in a couple decades at best. People might mention your software in footnotes, but no one would actually use it. And indeed, it would seem very feeble work. Only people with a sense of the history of the field would even realize that, in its time, it had been good.", metadata={'some': 'metadata'})]
    
