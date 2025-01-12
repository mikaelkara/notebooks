<img src="./images/llm_prompt_req_resp.png" height="35%" width="%65">

## Zero-shot prompting

Zero-prompt learning is a challenging yet fascinating area where models are trained to perform tasks without explicit learning examples in the input prompt. Here are some notable examples:

GPT-3 and Llama Language Model:

GPT-3, Llama 2, and Claude are powerful language models. The have demonstrated zero-shot learning. That is, without specific learning prompts or examples, it can generate coherent and contextually relevant responses, showcasing its ability to understand and respond to diverse queries.

### Named Entity Recognition (NER):

Models trained with zero-prompt learning for NER can identify and categorize named entities in text without being explicitly provided with examples for each specific entity.

### Dialogue Generation:

Zero-shot dialogue generation models can engage in conversations and respond appropriately to user input without being given explicit dialogues as training examples.

In our prompt engineering notebooks, we saw examples of zero-shot prompting: Text generation, summarization, translation, etc. None of the prompts were given any language examples to learn from; they model has prior learned knowledge of the language. 

Let's demonstrate how you can do NER and Dialogue generation with zero-shot learning.

**Note**: 
To run any of these relevant notebooks you will need an account on Anyscale Endpoints, Anthropic, or OpenAI, depending on what model you elect, along with the respective environment file. Use the template environment files to create respective `.env` file for either Anyscale Endpoints, Anthropic, or OpenAI.


```python
import warnings
import os

import openai
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv, find_dotenv
from llm_clnt_factory_api import ClientFactory, get_commpletion
```

#### Based on .env file use the appropriate LLM service provider APIs


```python
_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings('ignore')
llm_client_service_provider = os.getenv("LLM_SERVICE_PROVIDER", None)
if llm_client_service_provider not in ['anyscale', 'openai', 'anthropic']:
    raise ValueError(f"Client {'LLM_SERVICE_PROVIDER'} missing in the .env file.")
elif llm_client_service_provider == "openai" or llm_client_service_provider == "anyscale":                                   
    openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
    openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
    MODEL = os.getenv("MODEL")
    print(f"Using MODEL={MODEL}; base={openai.api_base}")
else: 
    api_key = os.getenv("ANTHROPIC_API_KEY", None)
    MODEL = os.getenv("MODEL")
    print(f"Using MODEL={MODEL}; base={llm_client_service_provider}")
```

    Using MODEL=claude-3-opus-20240229; base=anthropic
    

#### Creat the respective client using our factory class


```python
client_type = llm_client_service_provider
client_kwargs = {}
client = None
client_factory = ClientFactory()
if client_type == "openai" or client_type == "anyscale":
    client_factory.register_client(client_type, OpenAI)
    client_kwargs = {"api_key": openai.api_key,
                     "base_url": openai.api_base}
else:
    client_factory.register_client(client_type, Anthropic)
    client_kwargs = {"api_key": api_key}

# create respective client
client = client_factory.create_client(client_type, **client_kwargs)
print(client)
```

    <anthropic.Anthropic object at 0x109560610>
    


```python
system_content = """You are master of all knowledge, and a helpful sage.
                    You must complete any incomplete sentence by drawing from your vast
                    knowledge about history, literature, science, social science, philosophy, religion, economics, sports, etc.
                    You can also identify and categorize named entities.
                    You are also an helpful assitant to converse in a dialogue.
                  """
```

## Named Entity Recognition (NER):


```python
user_text = """Tesla, headquartered in Palo Alto, was founded by Elon Musk. 
The company recently announced a collaboration with NASA to explore sustainable technologies for space travel."""

zero_learning_prompt = f"""Analyze the text provided in three ticks and identify the named entities present. 
Categorize them into types such as persons, organizations, and locations. 
'''{user_text}'''.
"""
```


```python
response = get_commpletion(client, MODEL, system_content, zero_learning_prompt)
print(f"{response}\n")
```

    Here is the analysis of the text with named entities identified and categorized:
    
    Persons:
    1. Elon Musk
    
    Organizations:
    1. Tesla
    2. NASA 
    
    Locations:
    1. Palo Alto
    
    The text mentions that Tesla, a company founded by Elon Musk and headquartered in Palo Alto, California, has announced a partnership with NASA to develop sustainable technologies for space exploration.
    
    


```python
user_text = """ In the year 1969, Neil Armstrong became the first person to walk on the moon during the Apollo 11 mission. 
NASA, headquartered in Washington D.C., spearheaded this historic achievement. 
Armstrong's fellow astronaut, Buzz Aldrin, joined him in this extraordinary venture. 
The event took place on July 20, 1969, forever marking a significant milestone in human history."
"""
zero_learning_prompt = f"""Analyze the text provided in three ticks and identify the named entities present. 
Categorize them into types such as persons, organizations, and locations. 
'''{user_text}'''.
"""
```


```python
response = get_commpletion(client, MODEL, system_content, zero_learning_prompt)
print(f"{response}\n")
```

    Here is the analysis of the named entities in the provided text, categorized into types:
    
    Persons:
    1. Neil Armstrong
    2. Buzz Aldrin
    
    Organizations:
    1. NASA (National Aeronautics and Space Administration)
    
    Locations:
    1. Washington D.C.
    2. Moon
    
    Events:
    1. Apollo 11 mission
    
    Dates:
    1. 1969
    2. July 20, 1969
    
    

## Dialogue Generation


```python
user_text = """Hello, I've been experiencing issues with the software. It keeps crashing whenever I try to open a specific file. 
Can you help?
"""
dialogue_zero_learning_promt = f"""Generate a conversation between a customer and a support agent discussing a technical issue related to a software product
provided in the {user_text}. 
Note that the model has not been provided with specific examples of this dialogue during training
"""
```


```python
response = get_commpletion(client, MODEL, system_content, dialogue_zero_learning_promt)
print(f"{response}\n")
```

    Here is a sample conversation between a customer and support agent about a software issue:
    
    Customer: Hello, I've been experiencing issues with the software. It keeps crashing whenever I try to open a specific file. Can you help?
    
    Support Agent: Hello, I'm sorry to hear you're having trouble with the software crashing. I'd be happy to assist you with this issue. Can you please provide some more details about the problem? What type of file are you trying to open when the crash occurs?
    
    Customer: It's a large Excel spreadsheet, about 15MB in size. The software starts to open the file but then unexpectedly quits after about 10 seconds. I don't have any issues with smaller Excel files, only this one large spreadsheet.
    
    Support Agent: Thank you for those details. It sounds like the software may be having trouble handling such a large file size. As a first troubleshooting step, could you please try opening the file on a different computer to see if the crash still occurs? That will help determine if it's an issue with that specific Excel file or with the software installation on your machine.
    
    Customer: I just tried on my colleague's computer and had the same issue there - the software crashed when trying to open this large Excel file. So it doesn't seem to be specific to my computer. 
    
    Support Agent: Okay, thanks for testing that. Since the issue is happening on multiple machines, the problem likely lies with either the Excel file itself being corrupted in some way, or a bug with how our software handles larger files. Let's try a couple things:
    
    1. If possible, could you try saving that Excel spreadsheet as a new file and see if the newly saved version can be opened in our software without crashing? Sometimes re-saving can resolve corruption.
    
    2. If a re-saved file doesn't work, I'd like to get a copy of that spreadsheet so our development team can use it to diagnose the issue and develop a fix. Could you please upload the Excel file to this secure link? I'll open a ticket with development to investigate further.
    
    Customer: I just tried re-saving the spreadsheet as a new Excel file, but the software is still crashing when I try to open it. I have now uploaded the file to the link you provided. Please let me know once the developers are able to find a solution. This spreadsheet contains important data for my weekly reports, so I need to be able to open it as soon as possible.
    
    Support Agent: I completely understand the urgency and we'll make this a priority to resolve for you. I've downloaded the file you provided and opened a ticket with our development team, marked as high priority. They'll work to identify the root cause and develop a patch to fix the crashing issue. I've also noted the business impact this is having on your weekly reporting.
    
    Our standard SLA for high priority bugs is 2 business days for a patch to be released. I will monitor the ticket closely and reach out to development for any updates. Once a fix is available, I will contact you immediately with instructions to update your software. In the meantime, please let me know if you have any other questions or if there is anything else I can assist with.
    
    Customer: Thank you for escalating this to your development team so quickly. I appreciate you understanding the urgency and business impact. I look forward to hearing from you once a fix is available. I don't have any other questions at this time.
    
    Support Agent: You're welcome! I'm happy I could help get this resolved for you. I'll be in touch soon once I have an update from development. In the meantime, please don't hesitate to contact me if you need anything else. Thank you for your patience and understanding. Have a great rest of your day!
    
    

## All this is amazing! 😜 Feel the wizardy prompt power 🧙‍♀️
