# **Getting to know Llama 3: Everything you need to start building**
Our goal in this session is to provide a guided tour of Llama 3, including understanding different Llama 3 models, how and where to access them, Generative AI and Chatbot architectures, prompt engineering, RAG (Retrieval Augmented Generation), Fine-tuning and more. All this is implemented with a starter code for you to take it and use it in your Llama 3 projects.

### **Install dependencies**


```python
# Install dependencies and initialize
%pip install \
    langchain==0.1.19 \
    matplotlib \
    octoai-sdk==0.10.1 \
    openai \
    sentence_transformers \
    pdf2image \
    pdfminer \
    pdfminer.six \
    unstructured \
    faiss-cpu \
    pillow-heif \
    opencv-python \
    unstructured-inference \
    pikepdf
```

##**0 - Prerequisites**
* Basic understanding of Large Language Models

* Basic understanding of Python


```python
# presentation layer code

import base64
from IPython.display import Image, display
import matplotlib.pyplot as plt

def mm(graph):
  graphbytes = graph.encode("ascii")
  base64_bytes = base64.b64encode(graphbytes)
  base64_string = base64_bytes.decode("ascii")
  display(Image(url="https://mermaid.ink/img/" + base64_string))

def genai_app_arch():
  mm("""
  flowchart TD
    A[Users] --> B(Applications e.g. mobile, web)
    B --> |Hosted API|C(Platforms e.g. Custom, OctoAI, HuggingFace, Replicate)
    B -- optional --> E(Frameworks e.g. LangChain)
    C-->|User Input|D[Llama 3]
    D-->|Model Output|C
    E --> C
    classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def rag_arch():
  mm("""
  flowchart TD
    A[User Prompts] --> B(Frameworks e.g. LangChain)
    B <--> |Database, Docs, XLS|C[fa:fa-database External Data]
    B -->|API|D[Llama 3]
    classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def llama3_family():
  mm("""
  graph LR;
      llama-3 --> llama-3-8b-instruct
      llama-3 --> llama-3-70b-instruct
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def apps_and_llms():
  mm("""
  graph LR;
    users --> apps
    apps --> frameworks
    frameworks --> platforms
    platforms --> Llama 3
    classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

import ipywidgets as widgets
from IPython.display import display, Markdown

# Create a text widget
API_KEY = widgets.Password(
    value='',
    placeholder='',
    description='API_KEY:',
    disabled=False
)

def md(t):
  display(Markdown(t))

def bot_arch():
  mm("""
  graph LR;
  user --> prompt
  prompt --> i_safety
  i_safety --> context
  context --> Llama_3
  Llama_3 --> output
  output --> o_safety
  i_safety --> memory
  o_safety --> memory
  memory --> context
  o_safety --> user
  classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def fine_tuned_arch():
  mm("""
  graph LR;
      Custom_Dataset --> Pre-trained_Llama
      Pre-trained_Llama --> Fine-tuned_Llama
      Fine-tuned_Llama --> RLHF
      RLHF --> |Loss:Cross-Entropy|Fine-tuned_Llama
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def load_data_faiss_arch():
  mm("""
  graph LR;
      documents --> textsplitter
      textsplitter --> embeddings
      embeddings --> vectorstore
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def mem_context():
  mm("""
      graph LR
      context(text)
      user_prompt --> context
      instruction --> context
      examples --> context
      memory --> context
      context --> tokenizer
      tokenizer --> embeddings
      embeddings --> LLM
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

```

##**1 - Understanding Llama 3**

### **1.1 - What is Llama 3?**

* State of the art (SOTA), Open Source LLM
* Llama 3 8B, 70B
* Pretrained + Chat
* Choosing model: Size, Quality, Cost, Speed
* [Llama 3 blog](https://ai.meta.com/blog/meta-llama-3/)
* [Responsible use guide](https://ai.meta.com/llama/responsible-use-guide/)


```python
llama3_family()
```

###**1.2 - Accessing Llama 3**
* Download + Self Host (on-premise)
* Hosted API Platform (e.g. [OctoAI](https://octoai.cloud/), [Replicate](https://replicate.com/meta))
* Hosted Container Platform (e.g. [Azure](https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/introducing-llama-2-on-azure/ba-p/3881233), [AWS](https://aws.amazon.com/blogs/machine-learning/llama-2-foundation-models-from-meta-are-now-available-in-amazon-sagemaker-jumpstart/), [GCP](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/139))

### **1.3 - Use Cases of Llama 3**
* Content Generation
* Chatbots
* Summarization
* Programming (e.g. Code Llama)

* and many more...

##**2 - Using Llama 3**

In this notebook, we are going to access [Llama 3 8b instruct model](https://octoai.cloud/text/chat?model=meta-llama-3-8b-instruct&mode=api) using hosted API from OctoAI.


```python
# model on OctoAI platform that we will use for inferencing
# We will use llama 3 8b instruct model hosted on OctoAI server

llama3_8b = "meta-llama-3-8b-instruct"
```


```python
# We will use OctoAI hosted cloud environment
# Obtain OctoAI API key → https://octo.ai/docs/getting-started/how-to-create-an-octoai-access-token

# enter your replicate api token
from getpass import getpass
import os

OCTOAI_API_TOKEN = getpass()
os.environ["OCTOAI_API_TOKEN"] = OCTOAI_API_TOKEN

# alternatively, you can also store the tokens in environment variables and load it here
```


```python
# We will use OpenAI's APIs to talk to OctoAI's hosted model endpoint
from openai import OpenAI

client = OpenAI(
   base_url = "https://text.octoai.run/v1",
   api_key = os.environ["OCTOAI_API_TOKEN"]
)

# text completion with input prompt
def Completion(prompt):
    output = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=llama3_8b,
        max_tokens=1000
    )
    return output.choices[0].message.content

# chat completion with input prompt and system prompt
def ChatCompletion(prompt, system_prompt=None):
    output = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        model=llama3_8b,
        max_tokens=1000
    )
    return output.choices[0].message.content
```

# **2.1 - Basic completion**


```python
output = Completion(prompt="The typical color of a llama is: ")
md(output)
```

## **2.2 - System prompts**



```python
output = ChatCompletion(
    prompt="The typical color of a llama is: ",
    system_prompt="respond with only one word"
  )
md(output)
```

### **2.3 - Response formats**
* Can support different formatted outputs e.g. text, JSON, etc.


```python
output = ChatCompletion(
    prompt="The typical color of a llama is: ",
    system_prompt="response in json format"
  )
md(output)
```

## **3 - Gen AI Application Architecture**

Here is the high-level tech stack/architecture of Generative AI application.


```python
genai_app_arch()
```

##4 - **Chatbot Architecture**

Here are the key components and the information flow in a chatbot.

* User Prompts
* Input Safety
* Llama 3
* Output Safety

* Memory & Context


```python
bot_arch()
```

### **4.1 - Chat conversation**
* LLMs are stateless
* Single Turn

* Multi Turn (Memory)




```python
# example of single turn chat
prompt_chat = "What is the average lifespan of a Llama?"
output = ChatCompletion(prompt=prompt_chat, system_prompt="answer the last question in few words")
md(output)
```


```python
# example without previous context. LLM's are stateless and cannot understand "they" without previous context
prompt_chat = "What animal family are they?"
output = ChatCompletion(prompt=prompt_chat, system_prompt="answer the last question in few words")
md(output)
```

Chat app requires us to send in previous context to LLM to get in valid responses. Below is an example of Multi-turn chat.


```python
# example of multi-turn chat, with storing previous context
prompt_chat = """
User: What is the average lifespan of a Llama?
Assistant: Sure! The average lifespan of a llama is around 20-30 years.
User: What animal family are they?
"""
output = ChatCompletion(prompt=prompt_chat, system_prompt="answer the last question")
md(output)
```

### **4.2 - Prompt Engineering**
* Prompt engineering refers to the science of designing effective prompts to get desired responses

* Helps reduce hallucination


#### **4.2.1 - In-Context Learning (e.g. Zero-shot, Few-shot)**
 * In-context learning - specific method of prompt engineering where demonstration of task are provided as part of prompt.
  1. Zero-shot learning - model is performing tasks without any
input examples.
  2. Few or “N-Shot” Learning - model is performing and behaving based on input examples in user's prompt.


```python
# Zero-shot example. To get positive/negative/neutral sentiment, we need to give examples in the prompt
prompt = '''
Classify: I saw a Gecko.
Sentiment: ?
'''
output = ChatCompletion(prompt, system_prompt="one word response")
md(output)
```


```python
# By giving examples to Llama, it understands the expected output format.

prompt = '''
Classify: I love Llamas!
Sentiment: Positive
Classify: I dont like Snakes.
Sentiment: Negative
Classify: I saw a Gecko.
Sentiment:'''

output = ChatCompletion(prompt, system_prompt="One word response")
md(output)
```


```python
# another zero-shot learning
prompt = '''
QUESTION: Vicuna?
ANSWER:'''

output = ChatCompletion(prompt, system_prompt="one word response")
md(output)
```


```python
# Another few-shot learning example with formatted prompt.

prompt = '''
QUESTION: Llama?
ANSWER: Yes
QUESTION: Alpaca?
ANSWER: Yes
QUESTION: Rabbit?
ANSWER: No
QUESTION: Vicuna?
ANSWER:'''

output = ChatCompletion(prompt, system_prompt="one word response")
md(output)
```

#### **4.2.2 - Chain of Thought**
"Chain of thought" enables complex reasoning through logical step by step thinking and generates meaningful and contextually relevant responses.


```python
# Standard prompting
prompt = '''
Llama started with 5 tennis balls. It buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does Llama have now?
'''

output = ChatCompletion(prompt, system_prompt="provide short answer")
md(output)
```


```python
# Chain-Of-Thought prompting
prompt = '''
Llama started with 5 tennis balls. It buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does Llama have now?
Let's think step by step.
'''

output = ChatCompletion(prompt, system_prompt="provide short answer")
md(output)
```

### **4.3 - Retrieval Augmented Generation (RAG)**
* Prompt Eng Limitations - Knowledge cutoff & lack of specialized data

* Retrieval Augmented Generation(RAG) allows us to retrieve snippets of information from external data sources and augment it to the user's prompt to get tailored responses from Llama 3.

For our demo, we are going to download an external PDF file from a URL and query against the content in the pdf file to get contextually relevant information back with the help of Llama!


```python
rag_arch()
```

#### **4.3.1 - LangChain**
LangChain is a framework that helps make it easier to implement RAG.


```python
# langchain setup
from langchain.llms.octoai_endpoint import OctoAIEndpoint

# Use the Llama 3 model hosted on OctoAI
# max_tokens: Maximum number of tokens to generate. A word is generally 2-3 tokens
# temperature: Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value
# top_p: When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens
llama_model = OctoAIEndpoint(
    model=llama3_8b,
    max_tokens=1000,
    temperature=0.75,
    top_p=1
)
```


```python
# Step 1: load the external data source. In our case, we will load Meta’s “Responsible Use Guide” pdf document.
from langchain.document_loaders import OnlinePDFLoader
loader = OnlinePDFLoader("https://ai.meta.com/static-resource/responsible-use-guide/")
documents = loader.load()

# Step 2: Get text splits from document
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

# Step 3: Use the embedding model
from langchain.vectorstores import FAISS
from langchain.embeddings import OctoAIEmbeddings
embeddings = OctoAIEmbeddings(endpoint_url="https://text.octoai.run/v1/embeddings")

# Step 4: Use vector store to store embeddings
vectorstore = FAISS.from_documents(all_splits, embeddings)
```

#### **4.3.2 - LangChain Q&A Retriever**
* ConversationalRetrievalChain

* Query the Source documents



```python
# Query against your own data
from langchain.chains import ConversationalRetrievalChain
chain = ConversationalRetrievalChain.from_llm(llama_model, vectorstore.as_retriever(), return_source_documents=True)

chat_history = []
query = "How is Meta approaching open science in two short sentences?"
result = chain.invoke({"question": query, "chat_history": chat_history})
md(result['answer'])
```


```python
# This time your previous question and answer will be included as a chat history which will enable the ability
# to ask follow up questions.
chat_history = [(query, result["answer"])]
query = "How is it benefiting the world?"
result = chain({"question": query, "chat_history": chat_history})
md(result['answer'])
```

## **5 - Fine-Tuning Models**

* Limitatons of Prompt Eng and RAG
* Fine-Tuning Arch
* Types (PEFT, LoRA, QLoRA)
* Using PyTorch for Pre-Training & Fine-Tuning

* Evals + Quality



```python
fine_tuned_arch()
```

## **6 - Responsible AI**

* Power + Responsibility
* Hallucinations
* Input & Output Safety
* Red-teaming (simulating real-world cyber attackers)
* [Responsible Use Guide](https://ai.meta.com/llama/responsible-use-guide/)



##**7 - Conclusion**
* Active research on LLMs and Llama
* Leverage the power of Llama and its open community
* Safety and responsible use is paramount!

* Call-To-Action
  * [Replicate Free Credits](https://replicate.fyi/connect2023) for Connect attendees!
  * This notebook is available through Llama Github recipes
  * Use Llama in your projects and give us feedback


#### **Resources**
- [GitHub - Llama](https://github.com/facebookresearch/llama)
- [Github - LLama Recipes](https://github.com/facebookresearch/llama-recipes)
- [Llama](https://ai.meta.com/llama/)
- [Research Paper on Llama 2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
- [Llama 3 Page](https://ai.meta.com/blog/meta-llama-3/)
- [Model Card](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)
- [Responsible Use Guide](https://ai.meta.com/llama/responsible-use-guide/)
- [Acceptable Use Policy](https://ai.meta.com/llama/use-policy/)
- [OctoAI](https://octoai.cloud/)
- [LangChain](https://www.langchain.com/)

#### **Authors & Contact**
  * asangani@meta.com, [Amit Sangani | LinkedIn](https://www.linkedin.com/in/amitsangani/)
  * mohsena@meta.com, [Mohsen Agsen | LinkedIn](https://www.linkedin.com/in/dr-thierry-moreau/)

Adapted to run on OctoAI and use Llama 3 by tmoreau@octo.ai [Thierry Moreay | LinkedIn]()
