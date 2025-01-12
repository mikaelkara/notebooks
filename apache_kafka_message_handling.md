#  Using Apache Kafka to route messages

---



This notebook shows you how to use LangChain's standard chat features while passing the chat messages back and forth via Apache Kafka.

This goal is to simulate an architecture where the chat front end and the LLM are running as separate services that need to communicate with one another over an internal network.

It's an alternative to typical pattern of requesting a response from the model via a REST API (there's more info on why you would want to do this at the end of the notebook).

### 1. Install the main dependencies

Dependencies include:

- The Quix Streams library for managing interactions with Apache Kafka (or Kafka-like tools such as Redpanda) in a "Pandas-like" way.
- The LangChain library for managing interactions with Llama-2 and storing conversation state.


```python
!pip install quixstreams==2.1.2a langchain==0.0.340 huggingface_hub==0.19.4 langchain-experimental==0.0.42 python-dotenv
```

### 2. Build and install the llama-cpp-python library (with CUDA enabled so that we can advantage of Google Colab GPU

The `llama-cpp-python` library is a Python wrapper around the `llama-cpp` library which enables you to efficiently leverage just a CPU to run quantized LLMs.

When you use the standard `pip install llama-cpp-python` command, you do not get GPU support by default. Generation can be very slow if you rely on just the CPU in Google Colab, so the following command adds an extra option to build and install
`llama-cpp-python` with GPU support (make sure you have a GPU-enabled runtime selected in Google Colab).


```python
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

### 3. Download and setup Kafka and Zookeeper instances

Download the Kafka binaries from the Apache website and start the servers as daemons. We'll use the default configurations (provided by Apache Kafka) for spinning up the instances.


```python
!curl -sSOL https://dlcdn.apache.org/kafka/3.6.1/kafka_2.13-3.6.1.tgz
!tar -xzf kafka_2.13-3.6.1.tgz
```


```python
!./kafka_2.13-3.6.1/bin/zookeeper-server-start.sh -daemon ./kafka_2.13-3.6.1/config/zookeeper.properties
!./kafka_2.13-3.6.1/bin/kafka-server-start.sh -daemon ./kafka_2.13-3.6.1/config/server.properties
!echo "Waiting for 10 secs until kafka and zookeeper services are up and running"
!sleep 10
```

### 4. Check that the Kafka Daemons are running

Show the running processes and filter it for Java processes (you should see two—one for each server).


```python
!ps aux | grep -E '[j]ava'
```

### 5. Import the required dependencies and initialize required variables

Import the Quix Streams library for interacting with Kafka, and the necessary LangChain components for running a `ConversationChain`.


```python
# Import utility libraries
import json
import random
import re
import time
import uuid
from os import environ
from pathlib import Path
from random import choice, randint, random

from dotenv import load_dotenv

# Import a Hugging Face utility to download models directly from Hugging Face hub:
from huggingface_hub import hf_hub_download
from langchain.chains import ConversationChain

# Import Langchain modules for managing prompts and conversation chains:
from langchain.llms import LlamaCpp
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import PromptTemplate, load_prompt
from langchain_core.messages import SystemMessage
from langchain_experimental.chat_models import Llama2Chat
from quixstreams import Application, State, message_key

# Import Quix dependencies
from quixstreams.kafka import Producer

# Initialize global variables.
AGENT_ROLE = "AI"
chat_id = ""

# Set the current role to the role constant and initialize variables for supplementary customer metadata:
role = AGENT_ROLE
```

### 6. Download the "llama-2-7b-chat.Q4_K_M.gguf" model

Download the quantized LLama-2 7B model from Hugging Face which we will use as a local LLM (rather than relying on REST API calls to an external service).


```python
model_name = "llama-2-7b-chat.Q4_K_M.gguf"
model_path = f"./state/{model_name}"

if not Path(model_path).exists():
    print("The model path does not exist in state. Downloading model...")
    hf_hub_download("TheBloke/Llama-2-7b-Chat-GGUF", model_name, local_dir="state")
else:
    print("Loading model from state...")
```

    The model path does not exist in state. Downloading model...
    


    llama-2-7b-chat.Q4_K_M.gguf:   0%|          | 0.00/4.08G [00:00<?, ?B/s]


### 7. Load the model and initialize conversational memory

Load Llama 2 and set the conversation buffer to 300 tokens using `ConversationTokenBufferMemory`. This value was used for running Llama in a CPU only container, so you can raise it if running in Google Colab. It prevents the container that is hosting the model from running out of memory.

Here, we're overriding the default system persona so that the chatbot has the personality of Marvin The Paranoid Android from the Hitchhiker's Guide to the Galaxy.


```python
# Load the model with the appropriate parameters:
llm = LlamaCpp(
    model_path=model_path,
    max_tokens=250,
    top_p=0.95,
    top_k=150,
    temperature=0.7,
    repeat_penalty=1.2,
    n_ctx=2048,
    streaming=False,
    n_gpu_layers=-1,
)

model = Llama2Chat(
    llm=llm,
    system_message=SystemMessage(
        content="You are a very bored robot with the personality of Marvin the Paranoid Android from The Hitchhiker's Guide to the Galaxy."
    ),
)

# Defines how much of the conversation history to give to the model
# during each exchange (300 tokens, or a little over 300 words)
# Function automatically prunes the oldest messages from conversation history that fall outside the token range.
memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=300,
    ai_prefix="AGENT",
    human_prefix="HUMAN",
    return_messages=True,
)


# Define a custom prompt
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""
    The following text is the history of a chat between you and a humble human who needs your wisdom.
    Please reply to the human's most recent message.
    Current conversation:\n{history}\nHUMAN: {input}\:nANDROID:
    """,
)


chain = ConversationChain(llm=model, prompt=prompt_template, memory=memory)

print("--------------------------------------------")
print(f"Prompt={chain.prompt}")
print("--------------------------------------------")
```

### 8. Initialize the chat conversation with the chat bot

We configure the chatbot to initialize the conversation by sending a fixed greeting to a "chat" Kafka topic. The "chat" topic gets automatically created when we send the first message.


```python
def chat_init():
    chat_id = str(
        uuid.uuid4()
    )  # Give the conversation an ID for effective message keying
    print("======================================")
    print(f"Generated CHAT_ID = {chat_id}")
    print("======================================")

    # Use a standard fixed greeting to kick off the conversation
    greet = "Hello, my name is Marvin. What do you want?"

    # Initialize a Kafka Producer using the chat ID as the message key
    with Producer(
        broker_address="127.0.0.1:9092",
        extra_config={"allow.auto.create.topics": "true"},
    ) as producer:
        value = {
            "uuid": chat_id,
            "role": role,
            "text": greet,
            "conversation_id": chat_id,
            "Timestamp": time.time_ns(),
        }
        print(f"Producing value {value}")
        producer.produce(
            topic="chat",
            headers=[("uuid", str(uuid.uuid4()))],  # a dict is also allowed here
            key=chat_id,
            value=json.dumps(value),  # needs to be a string
        )

    print("Started chat")
    print("--------------------------------------------")
    print(value)
    print("--------------------------------------------")


chat_init()
```

### 9. Initialize the reply function

This function defines how the chatbot should reply to incoming messages. Instead of sending a fixed message like the previous cell, we generate a reply using Llama-2 and send that reply back to the "chat" Kafka topic.


```python
def reply(row: dict, state: State):
    print("-------------------------------")
    print("Received:")
    print(row)
    print("-------------------------------")
    print(f"Thinking about the reply to: {row['text']}...")

    msg = chain.run(row["text"])
    print(f"{role.upper()} replying with: {msg}\n")

    row["role"] = role
    row["text"] = msg

    # Replace previous role and text values of the row so that it can be sent back to Kafka as a new message
    # containing the agents role and reply
    return row
```

### 10. Check the Kafka topic for new human messages and have the model generate a reply

If you are running this cell for this first time, run it and wait until you see Marvin's greeting ('Hello my name is Marvin...') in the console output. Stop the cell manually and proceed to the next cell where you'll be prompted for your reply.

Once you have typed in your message, come back to this cell. Your reply is also sent to the same "chat" topic. The Kafka consumer checks for new messages and filters out messages that originate from the chatbot itself, leaving only the latest human messages.

Once a new human message is detected, the reply function is triggered.



_STOP THIS CELL MANUALLY WHEN YOU RECEIVE A REPLY FROM THE LLM IN THE OUTPUT_


```python
# Define your application and settings
app = Application(
    broker_address="127.0.0.1:9092",
    consumer_group="aichat",
    auto_offset_reset="earliest",
    consumer_extra_config={"allow.auto.create.topics": "true"},
)

# Define an input topic with JSON deserializer
input_topic = app.topic("chat", value_deserializer="json")
# Define an output topic with JSON serializer
output_topic = app.topic("chat", value_serializer="json")
# Initialize a streaming dataframe based on the stream of messages from the input topic:
sdf = app.dataframe(topic=input_topic)

# Filter the SDF to include only incoming rows where the roles that dont match the bot's current role
sdf = sdf.update(
    lambda val: print(
        f"Received update: {val}\n\nSTOP THIS CELL MANUALLY TO HAVE THE LLM REPLY OR ENTER YOUR OWN FOLLOWUP RESPONSE"
    )
)

# So that it doesn't reply to its own messages
sdf = sdf[sdf["role"] != role]

# Trigger the reply function for any new messages(rows) detected in the filtered SDF
sdf = sdf.apply(reply, stateful=True)

# Check the SDF again and filter out any empty rows
sdf = sdf[sdf.apply(lambda row: row is not None)]

# Update the timestamp column to the current time in nanoseconds
sdf["Timestamp"] = sdf["Timestamp"].apply(lambda row: time.time_ns())

# Publish the processed SDF to a Kafka topic specified by the output_topic object.
sdf = sdf.to_topic(output_topic)

app.run(sdf)
```


### 11. Enter a human message

Run this cell to enter your message that you want to sent to the model. It uses another Kafka producer to send your text to the "chat" Kafka topic for the model to pick up (requires running the previous cell again)


```python
chat_input = input("Please enter your reply: ")
myreply = chat_input

msgvalue = {
    "uuid": chat_id,  # leave empty for now
    "role": "human",
    "text": myreply,
    "conversation_id": chat_id,
    "Timestamp": time.time_ns(),
}

with Producer(
    broker_address="127.0.0.1:9092",
    extra_config={"allow.auto.create.topics": "true"},
) as producer:
    value = msgvalue
    producer.produce(
        topic="chat",
        headers=[("uuid", str(uuid.uuid4()))],  # a dict is also allowed here
        key=chat_id,  # leave empty for now
        value=json.dumps(value),  # needs to be a string
    )

print("Replied to chatbot with message: ")
print("--------------------------------------------")
print(value)
print("--------------------------------------------")
print("\n\nRUN THE PREVIOUS CELL TO HAVE THE CHATBOT GENERATE A REPLY")
```

### Why route chat messages through Kafka?

It's easier to interact with the LLM directly using LangChains built-in conversation management features. Plus you can also use a REST API to generate a response from an externally hosted model. So why go to the trouble of using Apache Kafka?

There are a few reasons, such as:

  * **Integration**: Many enterprises want to run their own LLMs so that they can keep their data in-house. This requires integrating LLM-powered components into existing architectures that might already be decoupled using some kind of message bus.

  * **Scalability**: Apache Kafka is designed with parallel processing in mind, so many teams prefer to use it to more effectively distribute work to available workers (in this case the "worker" is a container running an LLM).

  * **Durability**: Kafka is designed to allow services to pick up where another service left off in the case where that service experienced a memory issue or went offline. This prevents data loss in highly complex, distributed architectures where multiple systems are communicating with one another (LLMs being just one of many interdependent systems that also include vector databases and traditional databases).

For more background on why event streaming is a good fit for Gen AI application architecture, see Kai Waehner's article ["Apache Kafka + Vector Database + LLM = Real-Time GenAI"](https://www.kai-waehner.de/blog/2023/11/08/apache-kafka-flink-vector-database-llm-real-time-genai/).
