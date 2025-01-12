```python
import os
import openai
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType

# Replace these with your own values, either in environment variables or directly here
AZURE_STORAGE_ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT") or "mystorageaccount"
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER") or "content"
AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE") or "gptkb"
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX") or "gptkbindex"
AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE") or "myopenai"
AZURE_OPENAI_GPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT_DEPLOYMENT") or "davinci"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT") or "chat"

KB_FIELDS_CONTENT = os.environ.get("KB_FIELDS_CONTENT") or "content"
KB_FIELDS_CATEGORY = os.environ.get("KB_FIELDS_CATEGORY") or "category"
KB_FIELDS_SOURCEPAGE = os.environ.get("KB_FIELDS_SOURCEPAGE") or "sourcepage"

# Use the current user identity to authenticate with Azure OpenAI, Cognitive Search and Blob Storage (no secrets needed, 
# just use 'az login' locally, and managed identity when deployed on Azure). If you need to use keys, use separate AzureKeyCredential instances with the 
# keys for each service
azure_credential = DefaultAzureCredential()

# Used by the OpenAI SDK
openai.api_type = "azure"
openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
openai.api_version = "2022-12-01"

# Comment these two lines out if using keys, set your API key in the OPENAI_API_KEY environment variable instead
openai.api_type = "azure_ad"
openai.api_key = azure_credential.get_token("https://cognitiveservices.azure.com/.default").token

# Set up clients for Cognitive Search and Storage
search_client = SearchClient(
    endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
    index_name=AZURE_SEARCH_INDEX,
    credential=azure_credential)
```


```python
# ChatGPT uses a particular set of tokens to indicate turns in conversations
prompt_prefix = """<|im_start|>system
Assistant helps the company employees with their healthcare plan questions and employee handbook questions. 
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question. 
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].

Sources:
{sources}

<|im_end|>"""

turn_prefix = """
<|im_start|>user
"""

turn_suffix = """
<|im_end|>
<|im_start|>assistant
"""

prompt_history = turn_prefix

history = []

summary_prompt_template = """Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base. Generate a search query based on the conversation and the new question. Source names are not good search terms to include in the search query.

Summary:
{summary}

Question:
{question}

Search query:
"""
```


```python
# Execute this cell multiple times updating user_input to accumulate chat history
user_input = "Does my plan cover annual eye exams?"

# Exclude category, to simulate scenarios where there's a set of docs you can't see
exclude_category = None

if len(history) > 0:
    completion = openai.Completion.create(
        engine=AZURE_OPENAI_GPT_DEPLOYMENT,
        prompt=summary_prompt_template.format(summary="\n".join(history), question=user_input),
        temperature=0.7,
        max_tokens=32,
        stop=["\n"])
    search = completion.choices[0].text
else:
    search = user_input

# Alternatively simply use search_client.search(q, top=3) if not using semantic search
print("Searching:", search)
print("-------------------")
filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None
r = search_client.search(search, 
                         filter=filter,
                         query_type=QueryType.SEMANTIC, 
                         query_language="en-us", 
                         query_speller="lexicon", 
                         semantic_configuration_name="default", 
                         top=3)
results = [doc[KB_FIELDS_SOURCEPAGE] + ": " + doc[KB_FIELDS_CONTENT].replace("\n", "").replace("\r", "") for doc in r]
content = "\n".join(results)

prompt = prompt_prefix.format(sources=content) + prompt_history + user_input + turn_suffix

completion = openai.Completion.create(
    engine=AZURE_OPENAI_CHATGPT_DEPLOYMENT, 
    prompt=prompt, 
    temperature=0.7, 
    max_tokens=1024,
    stop=["<|im_end|>", "<|im_start|>"])

prompt_history += user_input + turn_suffix + completion.choices[0].text + "\n<|im_end|>" + turn_prefix
history.append("user: " + user_input)
history.append("assistant: " + completion.choices[0].text)

print("\n-------------------\n".join(history))
print("\n-------------------\nPrompt:\n" + prompt)
```
