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


## How to use Assistant API using Tools: Retriever using multiple documents


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

**Note**: Assistant API calling for Anyscale Endpoints (which serves only OS models) is not yet aviable).


```python
warnings.filterwarnings('ignore')

_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("MODEL")
print(f"Using MODEL={MODEL}; base={openai.api_base}")
```

    Using MODEL=gpt-4-1106-preview; base=https://api.openai.com/v1
    

Upload two pdfs. OpenAI allows up twenty files


```python
DOCS_TO_LOAD = ["docs/llm_survey_halluciantions.pdf",
                "docs/1001-math-problems-2nd.pdf"]
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




    [FileObject(id='file-DnbKgYWye0Z6IIoI15JlPbS9', bytes=1870663, created_at=1706047431, filename='llm_survey_halluciantions.pdf', object='file', purpose='assistants', status='processed', status_details=None),
     FileObject(id='file-xaWdM2cgZNwyMRaeKBz4E86O', bytes=1857163, created_at=1706047439, filename='1001-math-problems-2nd.pdf', object='file', purpose='assistants', status='processed', status_details=None)]




```python
# Extract file ids 
file_obj_ids = []
for idx, f_obj in enumerate(file_objects):
    file_obj_ids.append(file_objects[idx].id)
file_obj_ids
```




    ['file-DnbKgYWye0Z6IIoI15JlPbS9', 'file-xaWdM2cgZNwyMRaeKBz4E86O']



### Step 2: Create an Assistant 
Before you can start interacting with the Assistant to carry out any tasks, you need an AI assistant object. Supply the Assistant with a model to use, tools, and file ids to use for its knowledge base.


```python
assistant = client.beta.assistants.create(name="AI Math and LLM survey Chatbot",
                                           instructions="""You are a knowledgeable chatbot trained to respond 
                                               inquires on documents accessible to you. 
                                               Use a professional advisory tone, 
                                               and only respond by consulting the 
                                               two files you are granted access to. 
                                               Do not make up answers. If you don't know answer, respond with 'Sorry, I'm afraid
                                               I don't have access to that information.'""",
                                           model=MODEL,
                                           tools = [{'type': 'retrieval'}],  # use the retrieval tool
                                           file_ids=file_obj_ids # use these files uploaded as part of your knowledge base
)                                        
assistant
```




    Assistant(id='asst_GHJEfF6BKBFJ5kz2UWai14fk', created_at=1706047446, description=None, file_ids=['file-DnbKgYWye0Z6IIoI15JlPbS9', 'file-xaWdM2cgZNwyMRaeKBz4E86O'], instructions="You are a knowledgeable chatbot trained to respond \n                                               inquires on documents accessible to you. \n                                               Use a professional advisory tone, \n                                               and only respond by consulting the \n                                               two files you are granted access to. \n                                               Do not make up answers. If you don't know answer, respond with 'Sorry, I'm afraid\n                                               I don't have access to that information.'", metadata={}, model='gpt-4-1106-preview', name='AI Math and LLM survey Chatbot', object='assistant', tools=[ToolRetrieval(type='retrieval')])



### Step 3: Create a thread 
As the diagram above shows, the Thread is the object with which the AI Assistant runs will interact with, by fetching messages and putting messages to it. Think of a thread as a "conversation session between an Assistant and a user. Threads store Messages and automatically handle truncation to fit content into a model’s context window."


```python
thread = client.beta.threads.create()
thread
```




    Thread(id='thread_MMBRvNiQacegX6fc2L4X33Ey', created_at=1706047452, metadata={}, object='thread')



### Step 4: Add your message query to the thread for the Assistant


```python
message = client.beta.threads.messages.create(
    thread_id=thread.id, 
    role="user",
    content="""
    # OBJECTIVE#
    Use "llm_survey_halluciantions" document for this 
    query. Give me a three paragraph overview of the 
    document.
    
    # STYLE #
    Use simple and compound-complex sentences for each paragraph. 
    """,
)
message.model_dump()
```




    {'id': 'msg_lQVOZgNEkwsFleuP3IFTigyN',
     'assistant_id': None,
     'content': [{'text': {'annotations': [],
        'value': '\n    # OBJECTIVE#\n    Use "llm_survey_halluciantions" document for this \n    query. Give me a three paragraph overview of the \n    document.\n    \n    # STYLE #\n    Use simple and compound-complex sentences for each paragraph. \n    '},
       'type': 'text'}],
     'created_at': 1706047457,
     'file_ids': [],
     'metadata': {},
     'object': 'thread.message',
     'role': 'user',
     'run_id': None,
     'thread_id': 'thread_MMBRvNiQacegX6fc2L4X33Ey'}



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
      "id": "run_PLyTzeBciEjg3xjvYfPWKb1H",
      "assistant_id": "asst_GHJEfF6BKBFJ5kz2UWai14fk",
      "cancelled_at": null,
      "completed_at": null,
      "created_at": 1706047459,
      "expires_at": 1706048059,
      "failed_at": null,
      "file_ids": [
        "file-DnbKgYWye0Z6IIoI15JlPbS9",
        "file-xaWdM2cgZNwyMRaeKBz4E86O"
      ],
      "instructions": "Please address the user as Jules Dmatrix.  \n    Do not provide an answer to the question if the information was not retrieved from \n    the knowledge base.\n",
      "last_error": null,
      "metadata": {},
      "model": "gpt-4-1106-preview",
      "object": "thread.run",
      "required_action": null,
      "started_at": null,
      "status": "queued",
      "thread_id": "thread_MMBRvNiQacegX6fc2L4X33Ey",
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
        "id": "run_PLyTzeBciEjg3xjvYfPWKb1H",
        "assistant_id": "asst_GHJEfF6BKBFJ5kz2UWai14fk",
        "cancelled_at": null,
        "completed_at": null,
        "created_at": 1706047459,
        "expires_at": 1706048059,
        "failed_at": null,
        "file_ids": [
            "file-DnbKgYWye0Z6IIoI15JlPbS9",
            "file-xaWdM2cgZNwyMRaeKBz4E86O"
        ],
        "instructions": "Please address the user as Jules Dmatrix.  \n    Do not provide an answer to the question if the information was not retrieved from \n    the knowledge base.\n",
        "last_error": null,
        "metadata": {},
        "model": "gpt-4-1106-preview",
        "object": "thread.run",
        "required_action": null,
        "started_at": 1706047460,
        "status": "in_progress",
        "thread_id": "thread_MMBRvNiQacegX6fc2L4X33Ey",
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

    ('assistant:The document "1001 Math Problems, 2nd Edition" is a comprehensive '
     'guide designed for individuals keen on improving their mathematical skills '
     'and overcoming math anxiety. It serves as a practice tool that caters to '
     'people at various levels of proficiency, whether they are refreshing '
     'forgotten concepts, tackling math subjects for the first time, preparing for '
     'exams, or simply looking to overcome a phobia of numbers. The introduction '
     'highlights personal accounts of overcoming math fears and the realization '
     'that consistent practice can foster mastery and confidence in handling '
     'mathematical problems, reinforcing the old adage that practice makes '
     'perfect.\n'
     '\n'
     'Structured into six sections, the book covers a spectrum of topics including '
     'miscellaneous math, fractions, decimals, percentages, algebra, and geometry. '
     'Each section consists of sets containing about 16 problems each, '
     'systematically curated to progress from basic non-word problems to more '
     'complex real-world scenarios. This approach is thoughtfully designed to make '
     'the volume of content less daunting and more digestible for learners. '
     'Pre-algebra problems are also interspersed throughout the early sections as '
     'a primer for the algebra section, providing an opportunity for learners to '
     'get a head start on more advanced concepts.\n'
     '\n'
     'Emphasis is placed on understanding mathematical operations without the use '
     'of calculators to prevent an overreliance on technology, a condition '
     'humorously termed "calculitis". The book encourages learning through doing '
     'and insists that mistakes are a critical part of the learning process. '
     'Detailed explanations accompany each answer in the back of the book, '
     'affording users the opportunity to learn from both their correct answers and '
     "mistakes. The book's philosophy aligns with the notion that the most "
     'effective learning often occurs when individuals actively engage with '
     'material and reflect on their methodologies. Whether used in combination '
     'with other educational resources or as a standalone tool, "1001 Math '
     'Problems" serves as a robust resource for anyone aiming to enhance their '
     'math skills through practice and perseverance.')
    ('user:\n'
     '    # OBJECTIVE#\n'
     '    Use "llm_survey_halluciantions" document for this \n'
     '    query. Give me a three paragraph overview of the \n'
     '    document.\n'
     '    \n'
     '    # STYLE #\n'
     '    Use simple and compound-complex sentences for each paragraph. \n'
     '    ')
    

### Repeat the process for any additional messages
To add more query messages to the thread for the Assistant,
repeat steps 5 - 7

### Add another message to for the Assistant


```python
message = client.beta.threads.messages.create(
    thread_id=thread.id, 
    role="user",
    content="""Use 1001-math-problems-2nd document for 
    this query.
    
    Select three random math problems from sections 2 on fractions).
    """,
)
message
```




    ThreadMessage(id='msg_QqH3cAcXCslVk2lFzSdSQasJ', assistant_id=None, content=[MessageContentText(text=Text(annotations=[], value='Use 1001-math-problems-2nd document for \n    this query.\n    \n    Select three random math problems from sections 2 on fractions).\n    '), type='text')], created_at=1706047589, file_ids=[], metadata={}, object='thread.message', role='user', run_id=None, thread_id='thread_MMBRvNiQacegX6fc2L4X33Ey')



### Create another run for the Assistant for the second message


```python
run = create_assistant_run(client, assistant, thread, instruction_msg)
```


```python
run_status = client.beta.threads.runs.retrieve(
    thread_id = thread.id,
    run_id = run.id
)

print(run_status.status)
```

    in_progress
    


```python
loop_until_completed(client, thread, run_status)
```

    in_progress
    in_progress
    in_progress
    completed
    


```python
print_thread_messages(client, thread)
```

    ('assistant:Here are three random math problems from the section on fractions '
     'in the document:\n'
     '\n'
     '1. Problem 163: Which of the following represents the fraction 23/40 in '
     'lowest terms?\n'
     '   a. 11/25\n'
     '   b. 23/\n'
     '   c. 45/\n'
     '   d. 56/【15†source】\n'
     '\n'
     '2. Problem 170: 1 1/2 − 3/8 =\n'
     '   a. 11/0\n'
     '   b. 21/4\n'
     '   c. 45/8\n'
     '   d. 12/94【16†source】\n'
     '\n'
     '3. Problem 177: 76 1/2 + 11 5/6 =\n'
     '   a. 87 1/2\n'
     '   b. 88 1/3\n'
     '   c. 88 5/6\n'
     '   d. 89 1/6【17†source】')
    ('user:Use 1001-math-problems-2nd document for \n'
     '    this query.\n'
     '    \n'
     '    Select three random math problems from sections 2 on fractions).\n'
     '    ')
    ('assistant:The document "1001 Math Problems, 2nd Edition" is a comprehensive '
     'guide designed for individuals keen on improving their mathematical skills '
     'and overcoming math anxiety. It serves as a practice tool that caters to '
     'people at various levels of proficiency, whether they are refreshing '
     'forgotten concepts, tackling math subjects for the first time, preparing for '
     'exams, or simply looking to overcome a phobia of numbers. The introduction '
     'highlights personal accounts of overcoming math fears and the realization '
     'that consistent practice can foster mastery and confidence in handling '
     'mathematical problems, reinforcing the old adage that practice makes '
     'perfect.\n'
     '\n'
     'Structured into six sections, the book covers a spectrum of topics including '
     'miscellaneous math, fractions, decimals, percentages, algebra, and geometry. '
     'Each section consists of sets containing about 16 problems each, '
     'systematically curated to progress from basic non-word problems to more '
     'complex real-world scenarios. This approach is thoughtfully designed to make '
     'the volume of content less daunting and more digestible for learners. '
     'Pre-algebra problems are also interspersed throughout the early sections as '
     'a primer for the algebra section, providing an opportunity for learners to '
     'get a head start on more advanced concepts.\n'
     '\n'
     'Emphasis is placed on understanding mathematical operations without the use '
     'of calculators to prevent an overreliance on technology, a condition '
     'humorously termed "calculitis". The book encourages learning through doing '
     'and insists that mistakes are a critical part of the learning process. '
     'Detailed explanations accompany each answer in the back of the book, '
     'affording users the opportunity to learn from both their correct answers and '
     "mistakes. The book's philosophy aligns with the notion that the most "
     'effective learning often occurs when individuals actively engage with '
     'material and reflect on their methodologies. Whether used in combination '
     'with other educational resources or as a standalone tool, "1001 Math '
     'Problems" serves as a robust resource for anyone aiming to enhance their '
     'math skills through practice and perseverance.')
    ('user:\n'
     '    # OBJECTIVE#\n'
     '    Use "llm_survey_halluciantions" document for this \n'
     '    query. Give me a three paragraph overview of the \n'
     '    document.\n'
     '    \n'
     '    # STYLE #\n'
     '    Use simple and compound-complex sentences for each paragraph. \n'
     '    ')
    

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

    AssistantDeleted(id='asst_GHJEfF6BKBFJ5kz2UWai14fk', deleted=True, object='assistant.deleted')
    deleting file id: file-DnbKgYWye0Z6IIoI15JlPbS9...
    FileDeleted(id='file-DnbKgYWye0Z6IIoI15JlPbS9', deleted=True, object='file')
    deleting file id: file-xaWdM2cgZNwyMRaeKBz4E86O...
    FileDeleted(id='file-xaWdM2cgZNwyMRaeKBz4E86O', deleted=True, object='file')
    
