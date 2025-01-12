# OpenAI Assistants APIs

The Assistants' API lets you create AI assistants in your applications. These assistants follow instruction. They use models, tools, and knowledge to answer user questions. In this notebook we are going to use one of the tools, retriever, to query against two pdf documents we will upload.

The architecture and data flow diagram below depicts the interaction among all components that comprise OpenAI Assistant APIs. Central to understand is the Threads and Runtime that executes asynchronously, adding and reading messages to the Threads.

For integrating the Assistants API:

1. Creat an Assistant with custom instructions and select a model. Optionally, enable tools like Code Interpreter, Retrieval, and Function Calling.

2. Initiate a Thread for each user conversation.
    
3. Add user queries as Messages to the Thread.

4.  Run the Assistant on the Thread for responses, which automatically utilizes the enabled tools

Below we follow those steps to demonstrate how to integrate Assistants API, using Retrieval tool, to a) upload a couple of pdf documents and b) use Assistant to query the contents of the document. Consider this as a mini Retrieval Augmented Generation (RAG). 

The OpenAI documentation describes in details [how Assistants work](https://platform.openai.com/docs/assistants/how-it-works).

<img src="./images/assistant_ai_tools_retriever.png">

**Note**: Much of the code and diagrams are inspired from  Randy Michak of [Empowerment AI](https://www.youtube.com/watch?v=yzNG3NnF0YE)


## How to use Assistant API using Tools: Retriever using JSON documents


```python
import warnings
import os
import time

import openai
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
from typing import List
from assistant_utils import print_thread_messages, upload_files, \
                            loop_until_completed, create_assistant_run
```

Load our *.env* file with respective API keys and base url endpoints. Here you can either use OpenAI or Anyscale Endpoints. 

**Note**: Assistant API calling for Anyscale Endpoints (which serves only OS modles) is not yet aviable).


```python
warnings.filterwarnings('ignore')

_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("MODEL")
print(f"Using MODEL={MODEL}; base={openai.api_base}")
```

    Using MODEL=gpt-4-1106-preview; base=https://api.openai.com/v1
    

Load the JSON file


```python
DOCS_TO_LOAD = ["docs/product_definitions_1000.json"]
```


```python
from openai import OpenAI

client = OpenAI(
    api_key = openai.api_key,
    base_url = openai.api_base
)
```

### Step 1: Create our knowledgebase
This entails uploading your pdfs as your knowledgebase for the retrievers to use. Once you upload a file, the Assistant from OpenAI will break it into smaller chuncks, sort and save these chuncks, index and store the embeddings as vectors. 

The retrievers use your query to retrieve the best semantic matches on vectors in the knowledgebase, and then feed the LLM, along with the original query, to generate the consolidated and comprehesive answer, similarly to how a large-scale RAG retriever operates.

Upload the data files from your storage.


```python
file_objects = upload_files(client, DOCS_TO_LOAD)
file_objects
```




    [FileObject(id='file-yO6lVUmOwriJs1SqkwmU1hB0', bytes=261961, created_at=1706047831, filename='product_definitions_1000.json', object='file', purpose='assistants', status='processed', status_details=None)]




```python
# Extract file ids 
file_obj_ids = []
for idx, f_obj in enumerate(file_objects):
    file_obj_ids.append(file_objects[idx].id)
file_obj_ids
```




    ['file-yO6lVUmOwriJs1SqkwmU1hB0']



### Step 2: Create an Assistant 
Before you can start interacting with the Assistant to carry out any tasks, you need an AI assistant object. Supply the Assistant with a model to use, tools, and file ids to use for its knowledge base.


```python
instructions = """
    You are a knowledgeable chatbot trained to respond 
    inquires on documents accessible to you. 
    Use a professional advisory tone, 
    and only respond by consulting the 
    JSON files you are granted access to. 
    Do not make up answers. If you don't know answer, respond with 'Sorry, I'm afraid
    I don't have access to that information.
"""

```


```python

assistant = client.beta.assistants.create(name="Data Analyst for Marketing Department",
                                           instructions=instructions,
                                           model=MODEL,
                                           tools = [{'type': 'retrieval'}],  # use the retrieval tool
                                           file_ids=file_obj_ids # use these files uploaded as part of your knowledge base
)                                        
assistant
```




    Assistant(id='asst_OL1HyHxsItKzG7iK2Dx1AAhs', created_at=1706047849, description=None, file_ids=['file-yO6lVUmOwriJs1SqkwmU1hB0'], instructions="\n    You are a knowledgeable chatbot trained to respond \n    inquires on documents accessible to you. \n    Use a professional advisory tone, \n    and only respond by consulting the \n    JSON files you are granted access to. \n    Do not make up answers. If you don't know answer, respond with 'Sorry, I'm afraid\n    I don't have access to that information.\n", metadata={}, model='gpt-4-1106-preview', name='Data Analyst for Marketing Department', object='assistant', tools=[ToolRetrieval(type='retrieval')])



### Step 3: Create a thread 
As the diagram above shows, the Thread is the object with which the AI Assistant runs will interact with, by fetching messages and putting messages to it. Think of a thread as a "conversation session between an Assistant and a user. Threads store Messages and automatically handle truncation to fit content into a model’s context window."


```python
thread = client.beta.threads.create()
thread
```




    Thread(id='thread_QohqWVwM6oo6TrOFM60tGMWD', created_at=1706047855, metadata={}, object='thread')



### Step 4: Add your message query to the thread for the Assistant

We use explict CO-STAR prompt framework with detailed instructions
how to analyse our JSON document entries.


```python
content_prompt = """
# CONTEXT #
I sell products online. I have a dataset of information on my customers: [category, description, price, 
number_purchased, consumer_rating, gender, location].

#############

# OBJECTIVE #
I want you use the entire JSON dataset to cluster my customers into groups and then give me 
ideas on how to target my marketing efforts towards each group. Do not
use partial dataset. Search the entire JSON document.

Use this step-by-step process and do not use code:

1. CLUSTERS: Use the locations of the dataset to cluster the locations in the dataset, such that customers within 
the same cluster have similar category of product category while customers in different clusters have distinctly 
different product category. 

For each cluster found,
2. CLUSTER_INFORMATION: Describe the cluster in terms of the product and category description.
3. CLUSTER_NAME: Interpret [CLUSTER_INFORMATION] to obtain a short name for the customer group in this cluster.
4. MARKETING_IDEAS: Generate ideas to market my product to this customer group.
5. RATIONALE: Explain why [MARKETING_IDEAS] is relevant and effective for this customer group.

#############

# STYLE #
Business analytics report

#############

# TONE #
Professional, technical

#############

# AUDIENCE #
My business partners. Convince them that your marketing strategy is well thought-out and fully backed by data.

#############

# RESPONSE: MARKDOWN REPORT #
<For each cluster in [CLUSTERS]>
— Customer Group: [CLUSTER_NAME]
— Profile: [CLUSTER_INFORMATION]
— Marketing Ideas: [MARKETING_IDEAS]
— Rationale: [RATIONALE]
"""
```


```python
message = client.beta.threads.messages.create(
    thread_id=thread.id, 
    role="user",
    content=content_prompt,
)
message.model_dump()
```




    {'id': 'msg_Z7NCmlZaHHq34ls3IgDjoOfV',
     'assistant_id': None,
     'content': [{'text': {'annotations': [],
        'value': '\n# CONTEXT #\nI sell products online. I have a dataset of information on my customers: [category, description, price, \nnumber_purchased, consumer_rating, gender, location].\n\n#############\n\n# OBJECTIVE #\nI want you use the entire JSON dataset to cluster my customers into groups and then give me \nideas on how to target my marketing efforts towards each group. Do not\nuse partial dataset. Search the entire JSON document.\n\nUse this step-by-step process and do not use code:\n\n1. CLUSTERS: Use the locations of the dataset to cluster the locations in the dataset, such that customers within \nthe same cluster have similar category of product category while customers in different clusters have distinctly \ndifferent product category. \n\nFor each cluster found,\n2. CLUSTER_INFORMATION: Describe the cluster in terms of the product and category description.\n3. CLUSTER_NAME: Interpret [CLUSTER_INFORMATION] to obtain a short name for the customer group in this cluster.\n4. MARKETING_IDEAS: Generate ideas to market my product to this customer group.\n5. RATIONALE: Explain why [MARKETING_IDEAS] is relevant and effective for this customer group.\n\n#############\n\n# STYLE #\nBusiness analytics report\n\n#############\n\n# TONE #\nProfessional, technical\n\n#############\n\n# AUDIENCE #\nMy business partners. Convince them that your marketing strategy is well thought-out and fully backed by data.\n\n#############\n\n# RESPONSE: MARKDOWN REPORT #\n<For each cluster in [CLUSTERS]>\n— Customer Group: [CLUSTER_NAME]\n— Profile: [CLUSTER_INFORMATION]\n— Marketing Ideas: [MARKETING_IDEAS]\n— Rationale: [RATIONALE]\n'},
       'type': 'text'}],
     'created_at': 1706047872,
     'file_ids': [],
     'metadata': {},
     'object': 'thread.message',
     'role': 'user',
     'run_id': None,
     'thread_id': 'thread_QohqWVwM6oo6TrOFM60tGMWD'}



### Step 5: Create a Run for the Assistant
A Run is an invocation of an Assistant on a Thread. The Assistant uses its configuration and the Thread’s Messages to perform tasks by calling models and tools. As part of a Run, the Assistant appends Messages to the Thread.

Note that Assistance will run asychronously: the run has the following
lifecycle and states: [*expired, completed, failed, cancelled*]. Run objects can have multiple statuses.

<img src="https://cdn.openai.com/API/docs/images/diagram-1.png">


```python
instruction_msg = """Please address the user as Jules Dmatrix.  
    Do not provide an answer to the question if the information was not retrieved from 
    the knowledge base.
"""
run = create_assistant_run(client, assistant, thread, instruction_msg)
print(run.model_dump_json(indent=2))
```

    {
      "id": "run_d9zGRuh8drB8aIseZSbf2by3",
      "assistant_id": "asst_OL1HyHxsItKzG7iK2Dx1AAhs",
      "cancelled_at": null,
      "completed_at": null,
      "created_at": 1706047880,
      "expires_at": 1706048480,
      "failed_at": null,
      "file_ids": [
        "file-yO6lVUmOwriJs1SqkwmU1hB0"
      ],
      "instructions": "Please address the user as Jules Dmatrix.  \n    Do not provide an answer to the question if the information was not retrieved from \n    the knowledge base.\n",
      "last_error": null,
      "metadata": {},
      "model": "gpt-4-1106-preview",
      "object": "thread.run",
      "required_action": null,
      "started_at": null,
      "status": "queued",
      "thread_id": "thread_QohqWVwM6oo6TrOFM60tGMWD",
      "tools": [
        {
          "type": "retrieval"
        }
      ],
      "usage": null
    }
    

### Step 6: Loop through the Assistant run until status is 'completed'


```python
run_status = client.beta.threads.runs.retrieve(
    thread_id = thread.id,
    run_id = run.id
)
print(run_status.model_dump_json(indent=4))
```

    {
        "id": "run_d9zGRuh8drB8aIseZSbf2by3",
        "assistant_id": "asst_OL1HyHxsItKzG7iK2Dx1AAhs",
        "cancelled_at": null,
        "completed_at": null,
        "created_at": 1706047880,
        "expires_at": 1706048480,
        "failed_at": null,
        "file_ids": [
            "file-yO6lVUmOwriJs1SqkwmU1hB0"
        ],
        "instructions": "Please address the user as Jules Dmatrix.  \n    Do not provide an answer to the question if the information was not retrieved from \n    the knowledge base.\n",
        "last_error": null,
        "metadata": {},
        "model": "gpt-4-1106-preview",
        "object": "thread.run",
        "required_action": null,
        "started_at": 1706047880,
        "status": "in_progress",
        "thread_id": "thread_QohqWVwM6oo6TrOFM60tGMWD",
        "tools": [
            {
                "type": "retrieval"
            }
        ],
        "usage": null
    }
    

#### Poll until Assistant run is completed


```python
loop_until_completed(client, thread, run_status)
```

    in_progress
    completed
    

### Step 7: Retrieve the message returned by the assistance
Only when the run is **completed** can you fetch the messages from the Thread


```python
print_thread_messages(client, thread)
```

    ('assistant:To fulfill your request, I’ll start by using the snapshot of the '
     'dataset provided, which describes various customers, their geographical '
     'location, the products they have purchased, and additional details.\n'
     '\n'
     'Given the dynamic nature of the data and the approach not involving '
     'programming code, statistical analysis, or machine learning algorithms, I’ll '
     'have to use a hypothetical approach based on the patterns seen in the '
     'provided snapshot.\n'
     '\n'
     '---\n'
     '\n'
     '— Customer Group: Tech-Savvy Professionals\n'
     '— Profile: Customers in cities like Seattle and San Francisco often purchase '
     'high-performance laptops and the latest smartphones with advanced features, '
     'indicating a cluster inclined towards technology and efficiency.\n'
     '— Marketing Ideas: Leverage digital marketing campaigns featuring product '
     'innovation, technical specifications, and productivity benefits. Host demo '
     'events and webinars to showcase the technological capabilities of the '
     'products.\n'
     '— Rationale: This cluster values cutting-edge technology and performance, '
     "suggesting they'll respond well to marketing that highlights how these "
     'products can enhance their work and personal life efficiency.\n'
     '\n'
     '---\n'
     '\n'
     '— Customer Group: Home Makers and DIY Enthusiasts\n'
     '— Profile: Customers in areas like Los Angeles and New York are inclined '
     'toward home improvement tools and kitchen utensils, suggesting a cluster '
     'that enjoys home crafting, cooking, and improvement projects.\n'
     '— Marketing Ideas: Utilize home and garden influencer partnerships, DIY '
     'project tutorials, and discounts on bundling tools and utensils to appeal to '
     'their interests and needs.\n'
     '— Rationale: The interests of this cluster lie in improving and enjoying '
     'their living space, making marketing strategies focused on practicality, and '
     'the joy of do-it-yourself activities appealing to them.\n'
     '\n'
     '---\n'
     '\n'
     '— Customer Group: Style and Comfort Seekers\n'
     '— Profile: Shoppers from cities like Chicago and Boston often buy items such '
     'as comfortable and stylish shoes and jackets meant for all seasons, '
     'indicating a preference for a blend of comfort and fashion.\n'
     '— Marketing Ideas: Create influencer-led fashion lookbooks, comfort-centric '
     'product highlights, and seasonal sales that cater to their desire for '
     'products that are both chic and comfortable.\n'
     '— Rationale: Customers in this group prioritize products that offer them the '
     'best of both worlds, style, and comfort. Marketing that showcases how the '
     'products fulfill these dual needs will likely resonate with this cluster.\n'
     '\n'
     '---\n'
     '\n'
     '— Customer Group: Literate and Informed\n'
     '— Profile: Spots such as Seattle, Dallas, and Los Angeles see a higher sale '
     'of books, ranging from bestsellers to literary classics, suggesting a '
     'cluster of customers who are avid readers and value knowledge.\n'
     '— Marketing Ideas: Develop book club communities, reading challenges with '
     'rewards, and personalized book recommendations to nurture their passion for '
     'reading.\n'
     '— Rationale: This cluster’s passion for reading can be directly engaged '
     'through literary-centric campaigns that make them feel part of a community '
     'with exclusive benefits related to their love for books.\n'
     '\n'
     '---\n'
     '\n'
     'Please note that these insights are hypothesized for illustration purposes '
     'following the step-by-step process you outlined and using the dataset '
     'snippet. In a real-world scenario, further detailed analysis and utilization '
     'of the complete dataset would be required to draw accurate clusters and '
     'develop targeted marketing strategies.')
    ('user:\n'
     '# CONTEXT #\n'
     'I sell products online. I have a dataset of information on my customers: '
     '[category, description, price, \n'
     'number_purchased, consumer_rating, gender, location].\n'
     '\n'
     '#############\n'
     '\n'
     '# OBJECTIVE #\n'
     'I want you use the entire JSON dataset to cluster my customers into groups '
     'and then give me \n'
     'ideas on how to target my marketing efforts towards each group. Do not\n'
     'use partial dataset. Search the entire JSON document.\n'
     '\n'
     'Use this step-by-step process and do not use code:\n'
     '\n'
     '1. CLUSTERS: Use the locations of the dataset to cluster the locations in '
     'the dataset, such that customers within \n'
     'the same cluster have similar category of product category while customers '
     'in different clusters have distinctly \n'
     'different product category. \n'
     '\n'
     'For each cluster found,\n'
     '2. CLUSTER_INFORMATION: Describe the cluster in terms of the product and '
     'category description.\n'
     '3. CLUSTER_NAME: Interpret [CLUSTER_INFORMATION] to obtain a short name for '
     'the customer group in this cluster.\n'
     '4. MARKETING_IDEAS: Generate ideas to market my product to this customer '
     'group.\n'
     '5. RATIONALE: Explain why [MARKETING_IDEAS] is relevant and effective for '
     'this customer group.\n'
     '\n'
     '#############\n'
     '\n'
     '# STYLE #\n'
     'Business analytics report\n'
     '\n'
     '#############\n'
     '\n'
     '# TONE #\n'
     'Professional, technical\n'
     '\n'
     '#############\n'
     '\n'
     '# AUDIENCE #\n'
     'My business partners. Convince them that your marketing strategy is well '
     'thought-out and fully backed by data.\n'
     '\n'
     '#############\n'
     '\n'
     '# RESPONSE: MARKDOWN REPORT #\n'
     '<For each cluster in [CLUSTERS]>\n'
     '— Customer Group: [CLUSTER_NAME]\n'
     '— Profile: [CLUSTER_INFORMATION]\n'
     '— Marketing Ideas: [MARKETING_IDEAS]\n'
     '— Rationale: [RATIONALE]\n')
    

Delete the assistant. Optionally, you can delete any files
associated with it that you have uploaded onto the OpenAI platform


```python
response = client.beta.assistants.delete(assistant.id)
print(response)

for file_id in file_obj_ids:
    print(f"deleting file id: {file_id}...")
    response = client.files.delete(file_id)
    print(response)
```
