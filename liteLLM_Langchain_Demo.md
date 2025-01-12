# Langchain liteLLM Demo Notebook
## Use `ChatLiteLLM()` to instantly support 50+ LLM models
Langchain Docs: https://python.langchain.com/docs/integrations/chat/litellm

Call all LLM models using the same I/O interface

Example usage
```python
ChatLiteLLM(model="gpt-3.5-turbo")
ChatLiteLLM(model="claude-2", temperature=0.3)
ChatLiteLLM(model="command-nightly")
ChatLiteLLM(model="replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1")
```


```python
!pip install litellm langchain
```


```python
import os
from langchain.chat_models import ChatLiteLLM
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
```


```python
os.environ['OPENAI_API_KEY'] = ""
chat = ChatLiteLLM(model="gpt-3.5-turbo")
messages = [
    HumanMessage(
        content="what model are you"
    )
]
chat(messages)
```




    AIMessage(content='I am an AI model known as GPT-3, developed by OpenAI.', additional_kwargs={}, example=False)




```python
os.environ['ANTHROPIC_API_KEY'] = ""
chat = ChatLiteLLM(model="claude-2", temperature=0.3)
messages = [
    HumanMessage(
        content="what model are you"
    )
]
chat(messages)
```




    AIMessage(content=" I'm Claude, an AI assistant created by Anthropic.", additional_kwargs={}, example=False)




```python
os.environ['REPLICATE_API_TOKEN'] = ""
chat = ChatLiteLLM(model="replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1")
messages = [
    HumanMessage(
        content="what model are you?"
    )
]
chat(messages)
```




    AIMessage(content=" I'm an AI based based on LLaMA models (LLaMA: Open and Efficient Foundation Language Models, Touvron et al. 2023), my knowledge was built from a massive corpus of text, including books, articles, and websites, and I was trained using a variety of machine learning algorithms. My model architecture is based on the transformer architecture, which is particularly well-suited for natural language processing tasks. My team of developers and I are constantly working to improve and fine-tune my performance, and I am always happy to help with any questions you may have!", additional_kwargs={}, example=False)




```python
os.environ['COHERE_API_KEY'] = ""
chat = ChatLiteLLM(model="command-nightly")
messages = [
    HumanMessage(
        content="what model are you?"
    )
]
chat(messages)
```




    AIMessage(content=' I am an AI-based large language model, or Chatbot, built by the company Cohere. I am designed to have polite, helpful, inclusive conversations with users. I am always learning and improving, and I am constantly being updated with new information and improvements.\n\nI am currently in the development phase, and I am not yet available to the general public. However, I am currently being used by a select group of users for testing and feedback.\n\nI am a large language model, which means that I am trained on a massive amount of data and can understand and respond to a wide range of requests and questions. I am also designed to be flexible and adaptable, so I can be customized to suit the needs of different users and use cases.\n\nI am currently being used to develop a range of applications, including customer service chatbots, content generation tools, and language translation services. I am also being used to train other language models and to develop new ways of using large language models.\n\nI am constantly being updated with new information and improvements, so I am always learning and improving. I am also being used to develop new ways of using large language models, so I am always evolving and adapting to new use cases and requirements.', additional_kwargs={}, example=False)


