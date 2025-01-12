# Databricks Notebook with MLFlow AutoLogging for LiteLLM Proxy calls



```python
%pip install -U -qqqq databricks-agents mlflow langchain==0.3.1 langchain-core==0.3.6 
```


```python
%pip install "langchain-openai<=0.3.1"
```


```python
# Before logging this chain using the driver notebook, you must comment out this line.
dbutils.library.restartPython() 
```


```python
import mlflow
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_databricks import ChatDatabricks
from langchain_openai import ChatOpenAI
```


```python
import mlflow
mlflow.langchain.autolog()
```


```python
# These helper functions parse the `messages` array.

# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]
```


```python
model = ChatOpenAI(
    openai_api_base="LITELLM_PROXY_BASE_URL", # e.g.: http://0.0.0.0:4000
    model = "gpt-3.5-turbo", # LITELLM 'model_name'
    temperature=0.1, 
    api_key="LITELLM_PROXY_API_KEY" # e.g.: "sk-1234"
)
```


```python
############
# Prompt Template for generation
############
prompt = PromptTemplate(
    template="You are a hello world bot.  Respond with a reply to the user's question that is fun and interesting to the user.  User's question: {question}",
    input_variables=["question"],
)

############
# FM for generation
# ChatDatabricks accepts any /llm/v1/chat model serving endpoint
############
model = ChatDatabricks(
    endpoint="databricks-dbrx-instruct",
    extra_params={"temperature": 0.01, "max_tokens": 500},
)


############
# Simple chain
############
# The framework requires the chain to return a string value.
chain = (
    {
        "question": itemgetter("messages")
        | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
    }
    | prompt
    | model
    | StrOutputParser()
)
```


```python
# This is the same input your chain's REST API will accept.
question = {
    "messages": [
               {
            "role": "user",
            "content": "what is rag?",
        },
    ]
}

chain.invoke(question)
```




    'Hello there! I\'m here to help with your questions. Regarding your query about "rag," it\'s not something typically associated with a "hello world" bot, but I\'m happy to explain!\n\nRAG, or Remote Angular GUI, is a tool that allows you to create and manage Angular applications remotely. It\'s a way to develop and test Angular components and applications without needing to set up a local development environment. This can be particularly useful for teams working on distributed systems or for developers who prefer to work in a cloud-based environment.\n\nI hope this explanation of RAG has been helpful and interesting! If you have any other questions or need further clarification, feel free to ask.'




    Trace(request_id=tr-ea2226413395413ba2cf52cffc523502)



```python
mlflow.models.set_model(model=model)
```
