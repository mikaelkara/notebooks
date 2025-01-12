# Building a Memory-Enhanced Conversational Agent

## Overview
This tutorial outlines the process of creating a conversational AI agent with enhanced memory capabilities. The agent incorporates both short-term and long-term memory to maintain context and improve the quality of interactions over time.

## Motivation
Traditional chatbots often struggle with maintaining context beyond immediate interactions. This limitation can lead to disjointed conversations and a lack of personalization. By implementing both short-term and long-term memory, we aim to create an agent that can:
- Maintain context within a single conversation
- Remember important information across multiple sessions
- Provide more coherent and personalized responses

## Key Components
1. **Language Model**: The core AI component for understanding and generating responses.
2. **Short-term Memory**: Stores the immediate conversation history.
3. **Long-term Memory**: Retains important information across multiple conversations.
4. **Prompt Template**: Structures the input for the language model, incorporating both types of memory.
5. **Memory Manager**: Handles the storage and retrieval of information in both memory types.

## Method Details

### Setting Up the Environment
1. Import necessary libraries for the language model, memory management, and prompt handling.
2. Initialize the language model with desired parameters (e.g., model type, token limit).

### Implementing Memory Systems
1. Create a store for short-term memory (conversation history):
   - Use a dictionary to manage multiple conversation sessions.
   - Implement a function to retrieve or create new conversation histories.

2. Develop a long-term memory system:
   - Create a separate store for persistent information.
   - Implement functions to update and retrieve long-term memories.
   - Define simple criteria for what information to store long-term (e.g., longer user inputs).

### Designing the Conversation Structure
1. Create a prompt template that includes:
   - A system message defining the AI's role.
   - A section for long-term memory context.
   - A placeholder for the conversation history (short-term memory).
   - The current user input.

### Building the Conversational Chain
1. Combine the prompt template with the language model.
2. Wrap this combination with a component that manages message history.
3. Ensure the chain can access and update both short-term and long-term memory.

### Creating the Interaction Loop
1. Develop a main chat function that:
   - Retrieves relevant long-term memories.
   - Invokes the conversational chain with the current input and memories.
   - Updates the long-term memory based on the interaction.
   - Returns the AI's response.

### Testing and Refinement
1. Run example conversations to test the agent's ability to maintain context.
2. Review both short-term and long-term memories after interactions.
3. Adjust memory management criteria and prompt structure as needed.

## Conclusion
The Memory-Enhanced Conversational Agent offers several advantages over traditional chatbots:

- **Improved Context Awareness**: By utilizing both short-term and long-term memory, the agent can maintain context within and across conversations.
- **Personalization**: Long-term memory allows the agent to remember user preferences and past interactions, enabling more personalized responses.
- **Flexible Memory Management**: The implementation allows for easy adjustment of what information is stored long-term and how it's used in conversations.
- **Scalability**: The session-based approach allows for managing multiple independent conversations.

This implementation provides a foundation for creating more sophisticated AI agents. Future enhancements could include:
- More advanced criteria for long-term memory storage
- Implementation of memory consolidation or summarization techniques
- Integration with external knowledge bases
- Emotional or sentiment tracking across interactions

By focusing on memory enhancement, this conversational agent design significantly improves upon basic chatbot functionality, paving the way for more engaging, context-aware, and intelligent AI assistants.

## Setup and Imports

First, we'll import the necessary modules and set up our environment.


```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)
```

## Memory Stores

We'll create stores for both short-term (chat history) and long-term memory.


```python
chat_store = {}
long_term_memory = {}

def get_chat_history(session_id: str):
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]

def update_long_term_memory(session_id: str, input: str, output: str):
    if session_id not in long_term_memory:
        long_term_memory[session_id] = []
    if len(input) > 20:  # Simple logic: store inputs longer than 20 characters
        long_term_memory[session_id].append(f"User said: {input}")
    if len(long_term_memory[session_id]) > 5:  # Keep only last 5 memories
        long_term_memory[session_id] = long_term_memory[session_id][-5:]

def get_long_term_memory(session_id: str):
    return ". ".join(long_term_memory.get(session_id, []))
```

## Prompt Template

We'll create a prompt template that includes both short-term and long-term memory.


```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the information from long-term memory if relevant."),
    ("system", "Long-term memory: {long_term_memory}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
```

## Conversational Chain

Now, we'll set up the conversational chain with message history.


```python
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

## Chat Function

We'll create a function to handle chat interactions, including updating long-term memory.


```python
def chat(input_text: str, session_id: str):
    long_term_mem = get_long_term_memory(session_id)
    response = chain_with_history.invoke(
        {"input": input_text, "long_term_memory": long_term_mem},
        config={"configurable": {"session_id": session_id}}
    )
    update_long_term_memory(session_id, input_text, response.content)
    return response.content
```

## Example Usage

Let's test our memory-enhanced conversational agent with a series of interactions.


```python
session_id = "user_123"

print("AI:", chat("Hello! My name is Alice.", session_id))
print("AI:", chat("What's the weather like today?", session_id))
print("AI:", chat("I love sunny days.", session_id))
print("AI:", chat("Do you remember my name?", session_id))
```

    AI: Hello, Alice! How can I assist you today?
    AI: I don't have real-time weather data, but you can check a weather website or app for the most accurate and up-to-date information. If you tell me your location, I can suggest what to look for!
    AI: Sunny days are wonderful! They can really lift your mood and are perfect for outdoor activities. Do you have any favorite things you like to do on sunny days?
    AI: Yes, your name is Alice! How can I assist you further today?
    

## Review Memory

Let's review the conversation history and long-term memory.


```python
print("Conversation History:")
for message in chat_store[session_id].messages:
    print(f"{message.type}: {message.content}")

print("\nLong-term Memory:")
print(get_long_term_memory(session_id))
```

    Conversation History:
    human: Hello! My name is Alice.
    ai: Hello, Alice! How can I assist you today?
    human: What's the weather like today?
    ai: I don't have real-time weather data, but you can check a weather website or app for the most accurate and up-to-date information. If you tell me your location, I can suggest what to look for!
    human: I love sunny days.
    ai: Sunny days are wonderful! They can really lift your mood and are perfect for outdoor activities. Do you have any favorite things you like to do on sunny days?
    human: Do you remember my name?
    ai: Yes, your name is Alice! How can I assist you further today?
    
    Long-term Memory:
    User said: Hello! My name is Alice.. User said: What's the weather like today?. User said: Do you remember my name?
    
