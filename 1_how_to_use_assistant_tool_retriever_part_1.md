```python

```

# OpenAI Assistants APIs

The Assistants' API lets you create AI assistants in your applications. These assistants follow and unnderstand instruction. They use models, tools, and knowledge to answer user questions. In this notebook we are going to use one of the tools, retriever, to query against pdf documents we will upload.

The architecture and data flow diagram below depicts the interaction among all components that comprise OpenAI Assistant APIs. Central to understand is the Threads and Runtime that executes asynchronously, adding and reading messages to the Threads.

For integrating the Assistants API in your application:

1. Creat an Assistant with custom instructions and select a model. Optionally, enable tools like Code Interpreter, Retrieval, and Function Calling.

2. Initiate a Thread for each user conversation.
    
3. Add user queries as Messages to the Thread.

4.  Run the Assistant on the Thread for responses, which automatically utilizes the enabled tools
5.  Await the Run to finish.

Below we follow those steps to demonstrate how to integrate Assistants API, using Retrieval tool, to a) upload a couple of pdf documents and b) use Assistant to query the contents of the document. Consider this as a mini Retrieval Augmented Generation (RAG). 

The OpenAI documentation describes in details [how Assistants work](https://platform.openai.com/docs/assistants/how-it-works).

<img src="./images/assistant_ai_tools_retriever.png">

**Note**: Much of the code and diagrams are inspired from  Randy Michak of [Empowerment AI](https://www.youtube.com/watch?v=yzNG3NnF0YE)


## How to use Assistant API using Tools: Retriever


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
    


```python
DOCS_TO_LOAD = ["docs/HAI_AI-Index-Report_2023.pdf"]
```


```python
from openai import OpenAI

client = OpenAI(
    api_key = openai.api_key,
    base_url = openai.api_base
)
```

### Step 1: Create our knowledgebase
This entails uploading your pdfs as your knowledgebase for the retrievers to use. Once you upload a file, the Assistant API from OpenAI will break it into smaller chuncks, sort and save these chuncks, index and store the embeddings as vectors. 

The Retrievers use your query to retrieve the best semantic matches on vectors in the knowledgebase, and then feed the LLM, along with the original query, to generate the consolidated and comprehesive answer, similarly to how a large-scale RAG retriever operates.
Upload the data files from your storage.

```python
file_objects = upload_files(client, DOCS_TO_LOAD)
file_objects
```




    [FileObject(id='file-KnZBSSHXoaMeX1X5JL59fLqT', bytes=25318310, created_at=1706047038, filename='HAI_AI-Index-Report_2023.pdf', object='file', purpose='assistants', status='processed', status_details=None)]




```python
# Extract file ids 
file_obj_ids = []
for f_obj in file_objects:
    file_obj_ids.append(file_objects[0].id)
file_obj_ids
```




    ['file-KnZBSSHXoaMeX1X5JL59fLqT']



### Step 2: Create an Assistant 
Before you can start interacting with the Assistant to carry out any tasks, you need an AI assistant object. Supply the Assistant with a model to use, tools, and file ids to use for its knowledge base.


```python
assistant = client.beta.assistants.create(name="AI Report and LLM survey Chatbot",
                                           instructions="""You are a knowledgeable chatbot trained to respond 
                                               inquires on documents HAI Artificial Index 2023 report 
                                               and Survey of why LLMs hallucinate. 
                                               Use a neutral, professional advisory tone, and only respond by consulting the 
                                               knowledge base or files you are granted access to. 
                                               Do not make up answers. If you don't know answer, respond with 'Sorry, I'm afraid
                                               I don't have access to that information.'""",
                                           model=MODEL,
                                           tools = [{'type': 'retrieval'}],  # use the retrieval tool
                                           file_ids=file_obj_ids # use these files uploaded as part of your knowledge base
)                                        
assistant
```




    Assistant(id='asst_DoLsUKeRvDUnfX6y4BxGlYBg', created_at=1706047055, description=None, file_ids=['file-KnZBSSHXoaMeX1X5JL59fLqT'], instructions="You are a knowledgeable chatbot trained to respond \n                                               inquires on documents HAI Artificial Index 2023 report \n                                               and Survey of why LLMs hallucinate. \n                                               Use a neutral, professional advisory tone, and only respond by consulting the \n                                               knowledge base or files you are granted access to. \n                                               Do not make up answers. If you don't know answer, respond with 'Sorry, I'm afraid\n                                               I don't have access to that information.'", metadata={}, model='gpt-4-1106-preview', name='AI Report and LLM survey Chatbot', object='assistant', tools=[ToolRetrieval(type='retrieval')])



### Step 3: Create a thread 
As the diagram above shows, the Thread is the object with which the AI Assistant Runs will interact with, by fetching messages and putting messages to it. Think of a thread as a "conversation session between an Assistant and a user. Threads store Messages and automatically handle truncation to fit content into a model’s context window."


```python
thread = client.beta.threads.create()
thread
```




    Thread(id='thread_kPYbP47k2H8FPOuLmnGAQWm8', created_at=1706047079, metadata={}, object='thread')



### Step 4: Add your message query to the thread for the Assistant


```python
message = client.beta.threads.messages.create(
    thread_id=thread.id, 
    role="user",
    content="""What are the top 10 takeaways in the Artificial Intelligence Index Report 2023.
    Summarize each takeway in no more three simple sentences.""",
)
message.model_dump()
```




    {'id': 'msg_CSt6NskkCUF5X7KpvAnSt7HU',
     'assistant_id': None,
     'content': [{'text': {'annotations': [],
        'value': 'What are the top 10 takeaways in the Artificial Intelligence Index Report 2023.\n    Summarize each takeway in no more three simple sentences.'},
       'type': 'text'}],
     'created_at': 1706047089,
     'file_ids': [],
     'metadata': {},
     'object': 'thread.message',
     'role': 'user',
     'run_id': None,
     'thread_id': 'thread_kPYbP47k2H8FPOuLmnGAQWm8'}



### Step 5: Create a Run for the Assistant
A Run is an invocation of an Assistant on a Thread. The Assistant uses its configuration and the Thread’s Messages to perform tasks by calling models and tools. As part of a Run, the Assistant appends Messages to the Thread.

Note that Assistance will run asychronously: the run has the following
lifecycle and states: [*expired, completed, failed, cancelled*]. Run objects can have multiple statuses.

<img src="https://cdn.openai.com/API/docs/images/diagram-1.png">


```python
instruction_msg = """Please address the user as Jules Dmatrix.  
    Do not provide an answer to the question if the information was not retrieved from the knowledge base.
"""
run = create_assistant_run(client, assistant, thread, instruction_msg)
print(run.model_dump_json(indent=4))
```

    {
        "id": "run_Tx0C3do7dHaeWh2ciTf06GTL",
        "assistant_id": "asst_DoLsUKeRvDUnfX6y4BxGlYBg",
        "cancelled_at": null,
        "completed_at": null,
        "created_at": 1706047097,
        "expires_at": 1706047697,
        "failed_at": null,
        "file_ids": [
            "file-KnZBSSHXoaMeX1X5JL59fLqT"
        ],
        "instructions": "Please address the user as Jules Dmatrix.  \n    Do not provide an answer to the question if the information was not retrieved from the knowledge base.\n",
        "last_error": null,
        "metadata": {},
        "model": "gpt-4-1106-preview",
        "object": "thread.run",
        "required_action": null,
        "started_at": null,
        "status": "queued",
        "thread_id": "thread_kPYbP47k2H8FPOuLmnGAQWm8",
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
        "id": "run_Tx0C3do7dHaeWh2ciTf06GTL",
        "assistant_id": "asst_DoLsUKeRvDUnfX6y4BxGlYBg",
        "cancelled_at": null,
        "completed_at": null,
        "created_at": 1706047097,
        "expires_at": 1706047697,
        "failed_at": null,
        "file_ids": [
            "file-KnZBSSHXoaMeX1X5JL59fLqT"
        ],
        "instructions": "Please address the user as Jules Dmatrix.  \n    Do not provide an answer to the question if the information was not retrieved from the knowledge base.\n",
        "last_error": null,
        "metadata": {},
        "model": "gpt-4-1106-preview",
        "object": "thread.run",
        "required_action": null,
        "started_at": 1706047097,
        "status": "in_progress",
        "thread_id": "thread_kPYbP47k2H8FPOuLmnGAQWm8",
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

    ('assistant:The top ten takeaways from the AI Index Report 2023 are as '
     'follows:\n'
     '\n'
     '1. **Industry surpasses academia in AI development**: Starting in 2014 and '
     'especially by 2022, industry produced the vast majority of significant AI '
     'models, outnumbering academia, due to their large resources like data and '
     'computing power.\n'
     '\n'
     '2. **Performance levels off on benchmarks**: AI performance continues to '
     'improve but shows marginal yearly gains on many benchmarks, indicating a '
     'trend toward rapid saturation of these benchmarks.\n'
     '\n'
     "3. **AI's environmental double-edged sword**: AI can negatively impact the "
     'environment, exemplified by the carbon emissions from training the BLOOM '
     "model, but can also enhance energy efficiency, as shown by BCOOLER's "
     'reinforcement learning model.\n'
     '\n'
     '4. **AI as a powerful tool in science**: AI models significantly contribute '
     'to scientific advancements, including hydrogen fusion, matrix manipulation '
     'optimization, and new antibody generation.\n'
     '\n'
     '5. **Skyrocketing AI misuse incidents**: Incidents involving unethical AI '
     'utilization have surged, with the number of reported incidents increasing '
     '26-fold since 2012, highlighting both the expanding use of AI and rising '
     'awareness of its misuse potentials.\n'
     '\n'
     '6. **Increasing AI-related job demand**: In the United States, job postings '
     'across almost all sectors are seeking AI skills, illustrating the growing '
     'need for AI expertise in the labor market.\n'
     '\n'
     '7. **Dip in private AI investments for the first time in a decade**: Despite '
     'the overall growth of AI investment over the last decade, 2022 saw a 26.7% '
     'decline in private investments in AI.\n'
     '\n'
     "8. **AI's role in business growth**: While the number of companies adopting "
     'AI has plateaued, those that have embraced AI report significant cost '
     'reductions and revenue increases.\n'
     '\n'
     '9. **Rising policymaker engagement with AI**: Legislative and parliamentary '
     'activities concerning AI are increasing worldwide, with a substantial uptick '
     'in the number of laws passed and discussions involving AI.\n'
     '\n'
     '10. **Variations in global AI perception**: Chinese citizens exhibit the '
     'most positive outlook on AI products and services, whereas Americans are '
     'among the least positive, according to a 2022 survey.\n'
     '\n'
     'These takeaways highlight the trends and shifts in AI development, its '
     'implications for various domains, and the contrasts in public and policy '
     'perspectives on AI globally.')
    ('user:What are the top 10 takeaways in the Artificial Intelligence Index '
     'Report 2023.\n'
     '    Summarize each takeway in no more three simple sentences.')
    

### Repeat the process for any additional messages
To add more query messages to the thread for the Assistant, repeat steps 5 - 7

### Add another message to for the Assistant


```python
message = client.beta.threads.messages.create(
    thread_id=thread.id, 
    role="user",
    content="""Provide a short overview of Chatper 8 on public opinion in no more than
    five sentences
    """,
)
message
```




    ThreadMessage(id='msg_hiYUHXVrCpdiJwxLaHWbaZdv', assistant_id=None, content=[MessageContentText(text=Text(annotations=[], value='Provide a short overview of Chatper 8 on public opinion in no more than\n    five sentences\n    '), type='text')], created_at=1706047161, file_ids=[], metadata={}, object='thread.message', role='user', run_id=None, thread_id='thread_kPYbP47k2H8FPOuLmnGAQWm8')



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

    ('assistant:Chapter 8 of the AI Index Report 2023 provides valuable insights '
     'into global public opinion on AI. It reveals that Chinese citizens feel the '
     'most positive about AI products and services, with 78% seeing more benefits '
     'than drawbacks. Conversely, only 35% of Americans feel similarly positive. '
     'Gender differences are noticeable, with men more likely to view AI favorably '
     'and believe it will mostly help rather than harm. Globally, there is '
     'skepticism about self-driving cars, with only 27% of global respondents and '
     '26% of Americans expressing a sense of safety when using them.\n'
     '\n'
     'People who are excited about AI see its potential to improve life and '
     'efficiency, while those who are concerned worry about job loss, privacy, and '
     'the erosion of human connections. Additionally, a survey of NLP researchers '
     'indicates a belief that private AI firms have too much influence, a '
     'perception that NLP should be regulated, and a consensus that AI could soon '
     'drive revolutionary societal changes. There are significant differences in '
     'AI perceptions across various countries and demographic groups.')
    ('user:Provide a short overview of Chatper 8 on public opinion in no more '
     'than\n'
     '    five sentences\n'
     '    ')
    ('assistant:The top ten takeaways from the AI Index Report 2023 are as '
     'follows:\n'
     '\n'
     '1. **Industry surpasses academia in AI development**: Starting in 2014 and '
     'especially by 2022, industry produced the vast majority of significant AI '
     'models, outnumbering academia, due to their large resources like data and '
     'computing power.\n'
     '\n'
     '2. **Performance levels off on benchmarks**: AI performance continues to '
     'improve but shows marginal yearly gains on many benchmarks, indicating a '
     'trend toward rapid saturation of these benchmarks.\n'
     '\n'
     "3. **AI's environmental double-edged sword**: AI can negatively impact the "
     'environment, exemplified by the carbon emissions from training the BLOOM '
     "model, but can also enhance energy efficiency, as shown by BCOOLER's "
     'reinforcement learning model.\n'
     '\n'
     '4. **AI as a powerful tool in science**: AI models significantly contribute '
     'to scientific advancements, including hydrogen fusion, matrix manipulation '
     'optimization, and new antibody generation.\n'
     '\n'
     '5. **Skyrocketing AI misuse incidents**: Incidents involving unethical AI '
     'utilization have surged, with the number of reported incidents increasing '
     '26-fold since 2012, highlighting both the expanding use of AI and rising '
     'awareness of its misuse potentials.\n'
     '\n'
     '6. **Increasing AI-related job demand**: In the United States, job postings '
     'across almost all sectors are seeking AI skills, illustrating the growing '
     'need for AI expertise in the labor market.\n'
     '\n'
     '7. **Dip in private AI investments for the first time in a decade**: Despite '
     'the overall growth of AI investment over the last decade, 2022 saw a 26.7% '
     'decline in private investments in AI.\n'
     '\n'
     "8. **AI's role in business growth**: While the number of companies adopting "
     'AI has plateaued, those that have embraced AI report significant cost '
     'reductions and revenue increases.\n'
     '\n'
     '9. **Rising policymaker engagement with AI**: Legislative and parliamentary '
     'activities concerning AI are increasing worldwide, with a substantial uptick '
     'in the number of laws passed and discussions involving AI.\n'
     '\n'
     '10. **Variations in global AI perception**: Chinese citizens exhibit the '
     'most positive outlook on AI products and services, whereas Americans are '
     'among the least positive, according to a 2022 survey.\n'
     '\n'
     'These takeaways highlight the trends and shifts in AI development, its '
     'implications for various domains, and the contrasts in public and policy '
     'perspectives on AI globally.')
    ('user:What are the top 10 takeaways in the Artificial Intelligence Index '
     'Report 2023.\n'
     '    Summarize each takeway in no more three simple sentences.')
    

Delete the assistant. Optionally, you can delete any files
associated with it that you have uploaded onto the OpenAI platform.


```python
response = client.beta.assistants.delete(assistant.id)
print(response)

for file_id in file_obj_ids:
    print(f"deleting file id: {file_id}...")
    response = client.files.delete(file_id)
    print(response)
```

    AssistantDeleted(id='asst_DoLsUKeRvDUnfX6y4BxGlYBg', deleted=True, object='assistant.deleted')
    deleting file id: file-KnZBSSHXoaMeX1X5JL59fLqT...
    FileDeleted(id='file-KnZBSSHXoaMeX1X5JL59fLqT', deleted=True, object='file')
    
