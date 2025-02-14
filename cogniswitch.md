# Cogniswitch Toolkit

CogniSwitch is used to build production ready applications that can consume, organize and retrieve knowledge flawlessly. Using the framework of your choice, in this case Langchain, CogniSwitch helps alleviate the stress of decision making when it comes to, choosing the right storage and retrieval formats. It also eradicates reliability issues and hallucinations when it comes to responses that are generated. 

## Setup

Visit [this page](https://www.cogniswitch.ai/developer?utm_source=langchain&utm_medium=langchainbuild&utm_id=dev) to register a Cogniswitch account.

- Signup with your email and verify your registration 

- You will get a mail with a platform token and oauth token for using the services.



```python
%pip install -qU langchain-community
```

## Import necessary libraries


```python
import warnings

warnings.filterwarnings("ignore")

import os

from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_community.agent_toolkits import CogniswitchToolkit
from langchain_openai import ChatOpenAI
```

## Cogniswitch platform token, OAuth token and OpenAI API key


```python
cs_token = "Your CogniSwitch token"
OAI_token = "Your OpenAI API token"
oauth_token = "Your CogniSwitch authentication token"

os.environ["OPENAI_API_KEY"] = OAI_token
```

## Instantiate the cogniswitch toolkit with the credentials


```python
cogniswitch_toolkit = CogniswitchToolkit(
    cs_token=cs_token, OAI_token=OAI_token, apiKey=oauth_token
)
```

### Get the list of cogniswitch tools


```python
tool_lst = cogniswitch_toolkit.get_tools()
```

## Instantiate the LLM


```python
llm = ChatOpenAI(
    temperature=0,
    openai_api_key=OAI_token,
    max_tokens=1500,
    model_name="gpt-3.5-turbo-0613",
)
```

## Use the LLM with the Toolkit

### Create an agent with the LLM and Toolkit


```python
agent_executor = create_conversational_retrieval_agent(llm, tool_lst, verbose=False)
```

### Invoke the agent to upload a URL


```python
response = agent_executor.invoke("upload this url https://cogniswitch.ai/developer")

print(response["output"])
```

    The URL https://cogniswitch.ai/developer has been uploaded successfully. The status of the document is currently being processed. You will receive an email notification once the processing is complete.
    

### Invoke the agent to upload a File


```python
response = agent_executor.invoke("upload this file example_file.txt")

print(response["output"])
```

    The file example_file.txt has been uploaded successfully. The status of the document is currently being processed. You will receive an email notification once the processing is complete.
    

### Invoke the agent to get the status of a document


```python
response = agent_executor.invoke("Tell me the status of this document example_file.txt")

print(response["output"])
```

    The status of the document example_file.txt is as follows:
    
    - Created On: 2024-01-22T19:07:42.000+00:00
    - Modified On: 2024-01-22T19:07:42.000+00:00
    - Document Entry ID: 153
    - Status: 0 (Processing)
    - Original File Name: example_file.txt
    - Saved File Name: 1705950460069example_file29393011.txt
    
    The document is currently being processed.
    

### Invoke the agent with query and get the answer


```python
response = agent_executor.invoke("How can cogniswitch help develop GenAI applications?")

print(response["output"])
```

    CogniSwitch can help develop GenAI applications in several ways:
    
    1. Knowledge Extraction: CogniSwitch can extract knowledge from various sources such as documents, websites, and databases. It can analyze and store data from these sources, making it easier to access and utilize the information for GenAI applications.
    
    2. Natural Language Processing: CogniSwitch has advanced natural language processing capabilities. It can understand and interpret human language, allowing GenAI applications to interact with users in a more conversational and intuitive manner.
    
    3. Sentiment Analysis: CogniSwitch can analyze the sentiment of text data, such as customer reviews or social media posts. This can be useful in developing GenAI applications that can understand and respond to the emotions and opinions of users.
    
    4. Knowledge Base Integration: CogniSwitch can integrate with existing knowledge bases or create new ones. This allows GenAI applications to access a vast amount of information and provide accurate and relevant responses to user queries.
    
    5. Document Analysis: CogniSwitch can analyze documents and extract key information such as entities, relationships, and concepts. This can be valuable in developing GenAI applications that can understand and process large amounts of textual data.
    
    Overall, CogniSwitch provides a range of AI-powered capabilities that can enhance the development of GenAI applications by enabling knowledge extraction, natural language processing, sentiment analysis, knowledge base integration, and document analysis.
    
