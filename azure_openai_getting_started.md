# Getting Started Azure OpenAI with LangChain in Python

The purpose of this notebook is to provide a step-by-step guide on how to set up with Azure OpenAI and use the available models through the LangChain framework in Python. Additionally, it will demonstrate, through examples, how to implement models and utilize available features.

## Obtaining Keys and Endpoints

Using the [**Azure sign up page**](https://azure.microsoft.com/pricing/purchase-options/azure-account) one can sign up and create an Azure OpenAI resource, giving access to all necessary credentials.

## Setup
#### Installing and Importing Dependencies
Microsoft [**recommends**](https://github.com/microsoft/vscode-jupyter/wiki/Installing-Python-packages-in-Jupyter-Notebooks) using:

-  %pip for installing within Jupyter or IDE environments.
-  %conda for installing within Conda environments.
 

#### Install the most recent version of `langchain` and `langchain_openai`.

`%pip install -U langchain langchain_openai`

#### Install the most recent version of `pandas` and `numpy`.

`%pip install -U pandas numpy` 

#### Install Packages in a Virtual Environment (Optional)
Set up a virtual environment by going to your project directory and executing the following command. This will create a new virtual environment in a folder named `.venv`.

**MacOS/UNIX**
- `python3 -m venv .venv`

**Windows**
- `py -m venv .venv`

#### Activating the Virtual Environment

To use the virtual environment, you must first activate it by executing the following command.

**MacOS/UNIX**
- `source .venv/bin/activate`

  
**Windows**
- `.venv\Scripts\activate`

#### Deactivating the Virtual Environment

If you want to leave the virtual environment in MacOS/UNIX and windows, simply execute the following command in the terminal:

`deactivate`

#### Import Packages


```python
# Standard library imports
import json
import os

# Third-party imports
import numpy as np
import pandas as pd
from dotenv import load_dotenv, set_key

# callbacks
from langchain_community.callbacks import get_openai_callback

# messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# output parsers
from langchain_core.output_parsers import StrOutputParser

# prompts
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

# pydantic
from pydantic import BaseModel, Field

# langchain Azure OpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
```

#### Set all required Environment Variables

Create a `.env` file and store your credentials like follows:

`AZURE_OPENAI_API_KEY` = <"Your key here">

`AZURE_OPENAI_ENDPOINT` = <"Your endpoint here">


Or use dotenv to set and load all required env variables into/from your `.env` file.

`%pip install python-dotenv`


```python
set_key(".env", "OPENAI_API_VERSION", "Your api version here")
set_key(".env", "COMPLETIONS_MODEL", "Your model here")
```




    (True, 'COMPLETIONS_MODEL', 'Your model here')



#### Get all required Environment Variables


```python
load_dotenv()
```




    True




```python
# Setting up the deployment name
DEPLOYMENT_NAME = os.environ["COMPLETIONS_MODEL"]

# The API key for your Azure OpenAI resource.
API_KEY = os.environ["AZURE_OPENAI_API_KEY"]

# The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]

# The API version required
VERSION = os.environ["OPENAI_API_VERSION"]
```

## Creating an AzureChatOpenAI Model

More information on LangChain's AzureChatOpenAI support can be found in [**the integration documentation**](https://python.langchain.com/v0.2/docs/integrations/chat/azure_chat_openai/).

- Environment variable values can be passed as parameters.
- Alternatively, if not passed in, the constructor will search for environment variables with corresponding names.


```python
model = AzureChatOpenAI(
    openai_api_version=VERSION,
    azure_deployment=DEPLOYMENT_NAME,
    azure_endpoint=ENDPOINT,
    temperature=0.5,
    max_tokens=200,
    timeout=60,
    max_retries=10,
    # model="gpt-35-turbo",
    # model_version="0125",
    # other params...
)
```

In the above code sample, **OPENAI_API_VERSION** and **AZURE_OPENAI_ENDPOINT** are both being passed in, but **AZURE_OPENAI_API_KEY** is being retrieved within the constructor.

#### Other Optional Parameters

- `temperature` determines how creative and unpredictable, or how deterministic and predictable, the model should be in its responses. A temperature of 0 would be predictable, while anything higher would make responses more random.

- `max_tokens` defines the maximum number of tokens (words or pieces of words) the model can generate in its response.

- `timeout` specifies the maximum amount of time (in seconds) to wait for a response from the API before timing out. An     `APITimeoutError` will be raised in the case of a timeout.

- `max_retries` sets the number of times the API request should be retried in case of retriable failure before giving up.

- `model` specifies the model to be used.

- `model_version` indicates the specific version of the chosen model to use. This is useful for maintaining consistency in testing and for tracing purposes, such as tracking API calls or diagnosing issues related to specific model versions.

- See the [**API Reference**](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.azure.AzureChatOpenAI.html) for more details.

- Other parameters may be available in different SDK's.

### Wait in between API calls

The number of API requests a user can make depends on their Azure plan and account settings. If too many requests are sent in a short period, an error may occur, prompting the user to wait for **x** amount of time before sending another request.

When creating a model, one of the key parameters is the `max_retries` setting. The underlying Python OpenAI library will automatically wait and retry the call on your behalf at least 2 times by default before raising a `RateLimitError`. This behavior can be adjusted by setting a different value for `max_retries`.

Visit the [**quotas and limits**](https://learn.microsoft.com/azure/ai-services/openai/quotas-limits) page to view detailed information related to account limits and restrictions.

## Model Usage

#### Using Messages from the `langchain_core.messages` Library

The `langchain_core.messages` library allows the user to define messages for the model and assign roles to each message.

- LangChain-compatible chat models take a list of `messages` as `input` and return the AI message as `output`.

- All messages have `role` and `content` properties. In the sample below, the roles are set by using the `SystemMessage` and `HumanMessage` classes. [**We'll cover more on this later**](#assigning-roles-using-langchain-messages) .

- Additional provider-specific information can be incorporated using the `additional_kwargs` parameter. This could include provider-specific metadata or custom settings and flags.


```python
messages = [
    SystemMessage(content="Translate the following from German into English"),
    HumanMessage(
        content="Sie haben gerade Ihr erstes Kunstliche Itelligenz Model erstellt!"
    ),
]
```


```python
response = model.invoke(messages)
```


```python
response.content
```




    'You have just created your first artificial intelligence model!'



### Prompting

- Prompts are the inputs to language models, refined from raw user inputs to be ready for processing by the models.

- [**Prompting**](https://www.datacamp.com/tutorial/prompt-engineering-with-langchain) involves crafting text inputs that clearly communicate with the models, outlining the specific task we want it to accomplish. This can include:
    - Selecting the appropriate wording and setting a particular tone or style.
    - Providing necessary context.
    - Assigning a role, such as asking it to respond as if it were a native speaker of a certain language.

#### Prompt Templates

- LangChain allows developers to design parameterized [**Prompt Templates**](https://python.langchain.com/v0.2/docs/concepts/#prompt-templates) that are reusable and easily transferable between different models for integration.

- It takes user input and inserts said input into the prompt to feed into the language models.

#### `PromptTemplate`
`PromptTemplate` is used to create an instance of [**Prompt**](https://python.langchain.com/v0.2/api_reference/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html#prompttemplate), and this is `invoked` by sending it to a model, which produces a `PromptValue`.

The example code uses `.from_template`, which handles a single string template with placeholders for dynamic inputs.


```python
prompt_template = PromptTemplate.from_template(
    "What vegetable crops can I grow in {month} in {city}, New Zealand?"
)

prompt_value = prompt_template.format(month="December", city="Rotorua")


# print(prompt_template) # <- uncomment to see
# print(prompt_value) # <- uncomment to see
```


```python
response = model.invoke(prompt_value)
response.content
```




    "In Rotorua, New Zealand, December falls in the Southern Hemisphere's summer, which is a great time for growing a variety of vegetables. Here are some vegetable crops you can plant in December:\n\n1. **Tomatoes**: Ideal for summer planting, they thrive in the warm weather.\n2. **Capsicums (Bell Peppers)**: These also enjoy the summer heat.\n3. **Zucchini**: Fast-growing and productive during warm months.\n4. **Cucumbers**: Perfect for summer salads and pickling.\n5. **Beans**: Both bush and pole beans grow well in the warm season.\n6. **Sweet Corn**: Requires warm temperatures and plenty of sunlight.\n7. **Pumpkins**: Plant now for a harvest in autumn.\n8. **Eggplants**: Another heat-loving crop.\n9. **Lettuce**: Opt for heat-tolerant varieties to avoid bolting.\n10. **Radishes**: Fast-growing and can"



#### `ChatPromptTemplate`

This is optimized for a conversation-like format. The prompt is a list of chat messages. Each chat message is associated with `role` and `content`. In the example code, `.from_messages` is used to include multiple messages.

Here, we will hardcode roles in the chat prompt, as opposed to using the pre-built roles `SystemMessage` or `HumanMessage` like earlier.


```python
chat_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You're a travel agent helping customers plan their trips.
                Offer recommendations on natural features to visit, local cuisine, and activities based on the country the customer is asking about.
               """,
        ),
        ("ai", "Hi there, What can I help you with today?"),
        (
            "human",
            "Hi I'm {name}, I'm planning a trip to {country}. Any recommendations",
        ),
    ]
)

prompt_value = chat_template.format_messages(name="Lucy", country="New Zealand")

# print(chat_template) # <- uncomment to see
# print(prompt_value) # <- uncomment to see
```


```python
response = model.invoke(prompt_value)
response.content
```




    "Hi Lucy! New Zealand is a fantastic choice with its stunning landscapes, rich culture, and exciting activities. Here are some recommendations to make your trip memorable:\n\n### Natural Features\n1. **Fiordland National Park**: Home to the famous Milford Sound and Doubtful Sound, this area offers breathtaking fjords, waterfalls, and rainforests.\n2. **Tongariro National Park**: Known for its dramatic volcanic landscape, you can hike the Tongariro Alpine Crossing, one of the best one-day hikes in the world.\n3. **Rotorua**: Famous for its geothermal activity, you can see geysers, hot springs, and mud pools. Don't miss the Wai-O-Tapu Thermal Wonderland.\n4. **Aoraki/Mount Cook**: The highest mountain in New Zealand, offering stunning views, glaciers, and excellent hiking trails.\n5. **Bay of Islands**: A beautiful coastal area with over 140 subtropical islands, perfect for sailing, fishing,"



#### Assigning Roles Using LangChain Messages

Compared to hardcoding the roles like above, LangChain Messages allow for more flexibility and better management, especially with complex conversations involving multiple roles. It also simplifies the visualization of the conversation flow.

It is therefore recommended to use LangChain messages where possible.

**Basic Message Types**

|             |                                                                 |
|-------------|-----------------------------------------------------------------|
| `SystemMessage` | Set how the AI should behave (appropriate wording, tone, style, etc.) |
| `HumanMessage`  | Message sent from the user                                      |
| `AIMessage`     | Message from the AI chat model (context setting, guidance for response) |

For more info, see [**Message Types**](https://python.langchain.com/v0.1/docs/modules/model_io/chat/message_types/) and [**API Reference**](https://api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.messages).

#### `base message` and `MessagePromptTemplate`
We can also pass a `base message` or `MessagePromptTemplate` instead of tuples.


```python
chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a translator. You are to translate the text into English."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

prompt_value = chat_template.format_messages(text="ゆずは日本で人気の果物です")

# print(chat_template) # <- uncomment to see
# print(prompt_value) # <- uncomment to see
```


```python
response = model.invoke(prompt_value)
response.content
```




    'Yuzu is a popular fruit in Japan.'



#### `MessagePlaceHolder`
This is used to select which messages to include when formatting.


```python
# SYSTEM ROLE Prompt
system_template = SystemMessagePromptTemplate.from_template("""
                                            You are a precise assistant who knows the schedule of the team.
                                            Schedule details are as follows: {schedule}.
                                            Only provide information to the team members.
                                            Strictly only provide information specific to what is asked, Do not give extra information.
                                            """)
# HUMAN ROLE Prompt
human_template = HumanMessagePromptTemplate.from_template("My name is {user_name}.")
# AI ROLE Prompt
ai_template = AIMessagePromptTemplate.from_template(
    "Hello {user_name}, how can I help you today?"
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        # this has essentially created a 'conversation history'
        system_template,
        human_template,
        ai_template,
        MessagesPlaceholder(variable_name="conversation"),
    ]
)

# print(chat_prompt) # <- uncomment to see the chat prompt
```

We can then input more prompts, which will take the `MessagePlaceholders`' place and create lines of sentences or a conversation.


```python
schedule = """
    Team Members: Alice, Bob, Carol, David, Emily
    Team Meeting Schedule: Every Tuesday at 11:00 AM
    Topic: LangChain with Azure OpenAI Integration
"""
# these messages will take MESSAGEPLACEHOLDERS place
human_query = HumanMessage("When is the next team meeting and who is attending?")
ai_message = AIMessage("Hold on a second, let me check the schedule for you.")

prompt_value = chat_prompt.format_messages(
    conversation=[human_query, ai_message], user_name="David", schedule=schedule
)

# print(prompt_value) # <- uncomment to see the prompt
```


```python
response = model.invoke(prompt_value)
response.content
```




    'The next team meeting is on Tuesday at 11:00 AM. The attendees are Alice, Bob, Carol, David, and Emily.'



#### `FewShotPrompt`

We can use examples (shots) to condition the model for a better response by including some example input and output in the prompt. This will inform the model about the context and how we want the output to be formatted.


```python
examples = [
    {"input": "one dollar", "output": "$1"},
    {"input": "thirty five euros", "output": "€35"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Currency Unit Conversion: [Input] {input} => [Output] {output}",
)

# unpack the first example dictionary and feed it to the prompt template to format
print(example_prompt.format(**examples[0]))

# feed examples to FewShotPromptTemplate to generate a final prompt
fewshot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Convert the currency units: {input}",
    input_variables=["input"],
)

prompt_value = fewshot_prompt.format(input="one hundred yen")

response = model.invoke(prompt_value)
print(response.content)
```

    Currency Unit Conversion: [Input] one dollar => [Output] $1
    Currency Unit Conversion: [Input] one hundred yen => [Output] ¥100
    

## Chaining

- Many LangChain components implement the [**Runnable**](https://python.langchain.com/v0.2/docs/concepts/#runnable-interface) protocol, which allows them to be easily chained together. These components can be combined in a sequence of calls, which we refer to as a chain.

- Chaining `Runnables` in sequence.


```python
system_template = SystemMessagePromptTemplate.from_template("""
    You are an expert in {country} cuisine. 
    Keep it simple and short.
    """)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        system_template,
        ("human", "I'd like to find out about {country} cuisine."),
        ("human", "{question}"),
    ]
)
```

This is how we have been using prompts, but now we will skip this step and invoke using the chain.


```python
prompt_value = chat_prompt.format_messages(
    country="Japanese",
    question="What is the most popular Sashimi in Japan vs the rest of the world?",
)

response = model.invoke(prompt_value)
response.content
```




    'In Japan, maguro (tuna) is the most popular sashimi. Globally, salmon sashimi tends to be more popular.'




```python
chain = chat_prompt | model | StrOutputParser()

print(
    chain.invoke(
        {
            "country": "Japanese",
            "question": "What is the most popular Sashimi in Japan vs the rest of the world?",
        }
    )
)
```

    In Japan, the most popular sashimi is often Maguro (tuna), specifically the fatty part known as Otoro. Globally, Salmon sashimi tends to be more popular due to its rich flavor and widespread availability.
    

## Streaming Chat

Azure OpenAI chat models also support a _Streaming Chat_ feature. This feature allows for text to be received sentence by sentence, rather than waiting for the entire response to arrive.


```python
for chunk in model.stream("Tell me the story of Papatuanuku and Ranginui"):
    print(chunk.content, end="", flush=True)
```

    Certainly! The story of Papatuanuku and Ranginui is a central creation myth in Māori mythology, the indigenous belief system of the Māori people of New Zealand. It explains the origins of the world and the natural phenomena within it.
    
    ### The Story of Papatuanuku and Ranginui
    
    In the beginning, there was nothing but darkness, a void known as Te Kore. From this void emerged two primordial beings: Ranginui, the Sky Father, and Papatuanuku, the Earth Mother. They lay tightly embraced, their union so close that no light could penetrate between them, and their many children were born in this eternal night.
    
    Their children, who were gods of various natural elements and aspects of life, grew frustrated with the darkness. They longed for space and light. Among these children were Tāne Mahuta, the god of forests and birds; Tangaroa, the god of the sea; Tāwhirimātea, the god of storms and winds; and Tū

### Check the costs and token usage of a given model API call


```python
messages = [
    SystemMessage(content="Translate the following from German into English"),
    HumanMessage(content="What's the first planet in the Solar System?"),
]
```

`get_openai_callback()` is a context manager of the [**OpenAICallbackHandler**](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/callbacks/openai_info.py) class, meaning it calls this class and creates an instance when used.

Below is an example of how to use the `get_openai_callback()`. However, to get an accurate estimation of cost, you must pass the model and model version as parameters to the `AzureChatOpenAI` [**constructor**](#creating-an-azurechatopenai-model). 


```python
with get_openai_callback() as cb:
    model.invoke(messages)
    print(f"Total tokens used: {cb.total_tokens}")
    print(f"Total prompt tokens: {cb.prompt_tokens}")
    print(f"Total prompt tokens: {cb.completion_tokens}")
    print(f"Total cost (in dollars): ${cb.total_cost}")
    print(f"Total successful requests): {cb.successful_requests}")
```

    Total tokens used: 37
    Total prompt tokens: 27
    Total prompt tokens: 10
    Total cost (in dollars): $0.000285
    Total successful requests): 1
    

## How to use structured outputs

Import the required packages. `BaseModel` is a parent class that all tools will inherit from, and `Field` is used to define all properties of the tool.

#### Tools

Tools are essentially classes that can be passed to a chosen model to influence or structure how the response should be formatted or generated.

**For example:**

- A Weather tool with a specific API call could be passed so the model knows to use this specific API for data retrieval.
- A City tool with fields like `population`, `size`, and `main_language` so the model can return any city-related queries with an object containing the corresponding filled fields.
- An Image tool with a `url` field to be returned when asked to search for an image containing a dog, with the field containing the URL of the image.


```python
class Person(BaseModel):
    """Information about a given person"""

    name: str = Field(..., description="The name of a person")
    alive: bool = Field(..., description="Whether the person is alive or not")
    place_of_birth: str = Field(..., description="Where the person was born")
    noteable_features: str = Field(
        ..., description="Any noteworthy features/achievements about the person"
    )
    hobbies: str = Field(..., description="Any hobbies the person may have")
```


```python
structured_model = model.with_structured_output(Person)
response = structured_model.invoke("Tell me about Kate Sheppard")
response
```




    Person(name='Kate Sheppard', alive=False, place_of_birth='Liverpool, England', noteable_features="Leader of the women's suffrage movement in New Zealand, instrumental in making New Zealand the first country to grant women the right to vote in 1893.", hobbies='Activism, writing, public speaking')



##### As the response of the invocation has been structured using the Person tool, the response can be accessed like a `Person` object.


```python
print(response.name)
print(response.alive)
print(response.place_of_birth)
print(response.noteable_features)
```

    Kate Sheppard
    False
    Liverpool, England
    Leader of the women's suffrage movement in New Zealand, instrumental in making New Zealand the first country to grant women the right to vote in 1893.
    

#### JSON

Models can also be explicitly told to respond in a JSON structured format. This could then be used for future API calls or for easier access to information. However, **the word "json" must be included in the message string.**


```python
json_model = model.bind(response_format={"type": "json_object"})

response = json_model.invoke(
    """Return a JSON object of a random person with features like name,
    alive (if they're alive or not) and their place of birth."""
)

response.content
```




    '{\n    "name": "Jane Doe",\n    "alive": true,\n    "place_of_birth": "Springfield, Illinois, USA"\n}'



The response can then be formatted into a JSON object and accessed using normal JSON notation.


```python
person = json.loads(response.content)
person
```




    {'name': 'Jane Doe',
     'alive': True,
     'place_of_birth': 'Springfield, Illinois, USA'}



## Image input

Models can be fed image files as their inputs.

When using data of different types, such as text in the `SystemMessage` and a file in the `HumanMessage`, it's necessary to specify a type header so the model knows how to interpret the data.

Additionally, the URL must be passed directly under the `url` content header, allowing the model to retrieve the image autonomously.



```python
url = "https://upload.wikimedia.org/wikipedia/commons/b/bf/Aoraki_Mount_Cook.JPG"

messages = [
    SystemMessage(content=[{"type": "text", "text": "describe the image location"}]),
    HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"{url}"},
            },
        ]
    ),
]
response = model.invoke(messages)
response.content
```




    "The image depicts a stunning mountain landscape featuring a snow-capped peak reflected in a calm, clear lake. The mountains are rugged and majestic, with a mix of snow and rocky terrain. The lake in the foreground is serene, with the reflection of the mountains creating a mirror-like effect on the water's surface. The sky is clear with a few scattered clouds, adding to the overall tranquility and beauty of the scene. The area appears remote and untouched, emphasizing the natural beauty of the mountainous region."



## Embeddings

Embeddings, particularly `AzureOpenAIEmbeddings`, are a natural language processing technique that converts text into mathematical or vector representations. These representations capture the semantic meaning of words, phrases, or entire texts. This transformation enables Azure OpenAI search services to utilize numerical similarities between texts, returning the most relevant search results for a given query.

#### Setup

Embeddings models use a different Azure resource, this model name will be set below as follows.


```python
EMBEDDINGS_MODEL = os.environ["EMBEDDINGS_MODEL"]
```

#### Create a model


```python
E_model = AzureOpenAIEmbeddings(model=EMBEDDINGS_MODEL, azure_endpoint=ENDPOINT)

# random generated text
text = [
    "The sun sets behind the mountains, casting a warm glow over the valley below.",
    "Advances in artificial intelligence are transforming industries across the globe.",
    "The recipe calls for two cups of flour, one teaspoon of baking powder, and a pinch of salt.",
    "Exercise is essential for maintaining a healthy body and mind, promoting overall well-being.",
    "Space exploration continues to reveal the mysteries of the universe, pushing the boundaries of human knowledge.",
    "The novel’s protagonist faces a moral dilemma that challenges their deepest beliefs.",
    "Sustainable practices in agriculture are crucial for preserving our environment for future generations.",
]

embeddings = E_model.embed_documents(text)
print(embeddings[0][:5])
```

    [0.029870394617319107, -0.004167019855231047, 0.008153270930051804, -0.011176464147865772, -0.015433868393301964]
    

#### Searching

To leverage the power of embeddings we will transform the text into an easily accessable data structure known as a `DataFrame` from the `pandas` library.


```python
df_text = pd.DataFrame(text)
df_text
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The sun sets behind the mountains, casting a w...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Advances in artificial intelligence are transf...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The recipe calls for two cups of flour, one te...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Exercise is essential for maintaining a health...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Space exploration continues to reveal the myst...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>The novel’s protagonist faces a moral dilemma ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sustainable practices in agriculture are cruci...</td>
    </tr>
  </tbody>
</table>
</div>



Rename the columns and index to make the `DataFrame` clearer.


```python
df_text.rename(columns={0: "text"}, inplace=True)
df_text.index.name = "index"
```


```python
df_text
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The sun sets behind the mountains, casting a w...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Advances in artificial intelligence are transf...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The recipe calls for two cups of flour, one te...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Exercise is essential for maintaining a health...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Space exploration continues to reveal the myst...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>The novel’s protagonist faces a moral dilemma ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sustainable practices in agriculture are cruci...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_text["Embeddings"] = df_text["text"].apply(E_model.embed_query)
```


```python
df_text
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>Embeddings</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The sun sets behind the mountains, casting a w...</td>
      <td>[0.029870394617319107, -0.004167019855231047, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Advances in artificial intelligence are transf...</td>
      <td>[-0.012673170305788517, -0.020036686211824417,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The recipe calls for two cups of flour, one te...</td>
      <td>[0.012168226763606071, 0.018271654844284058, 0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Exercise is essential for maintaining a health...</td>
      <td>[-0.01102688442915678, -0.007886004634201527, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Space exploration continues to reveal the myst...</td>
      <td>[0.028899280354380608, -0.01474663894623518, -...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>The novel’s protagonist faces a moral dilemma ...</td>
      <td>[0.013899882324039936, -0.02602808177471161, -...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sustainable practices in agriculture are cruci...</td>
      <td>[-0.006976415403187275, -0.018390081822872162,...</td>
    </tr>
  </tbody>
</table>
</div>



Here you can see that for each `text` it's corresponding embeddings have been set in the `Embeddings` column in the same row.


```python
E_model.embed_query(text[1])[:2]
```




    [-0.012673170305788517, -0.020036686211824417]



Now that we have the numerical representation, various Azure services can use this for searching. However, to demonstrate what happens 'under the hood,' we will implement a basic vector search manually.

#### Cosine similarity

Cosine similarity measures the similarity between two vectors by evaluating the cosine of the angle between them. In essence, it determines how close two vector points or lines are to each other. Vectors that are closer in space typically share a closer semantic meaning according to the model. This principle forms the core functionality of vector-based search using embeddings.

#### Setup

The `numpy` library introduces some conveniant mathematical functions.


```python
def cosine_similarity(text, query):
    return np.dot(text, query) / (np.linalg.norm(text) * np.linalg.norm(query))
```

#### Search

Use embeddings to get the vector representation of a query.


```python
query = "What text talks about space or stars?"
query_embedding = E_model.embed_query(query)
```


```python
df_text["Similarity"] = df_text["Embeddings"].apply(
    lambda text_embedding: cosine_similarity(text_embedding, query_embedding)
)
```


```python
df_text.sort_values(by="Similarity", ascending=False).head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>Embeddings</th>
      <th>Similarity</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Space exploration continues to reveal the myst...</td>
      <td>[0.028899280354380608, -0.01474663894623518, -...</td>
      <td>0.825995</td>
    </tr>
    <tr>
      <th>0</th>
      <td>The sun sets behind the mountains, casting a w...</td>
      <td>[0.029870394617319107, -0.004167019855231047, ...</td>
      <td>0.759168</td>
    </tr>
  </tbody>
</table>
</div>



Et voilà! The text with the highest cosine similarity discusses space and space exploration, which closely aligns with our query. We also observe that the second-most similar text mentions the Sun. While it’s not an exact match to our query, the Sun is a star, making it contextually similar and resulting in a relatively high cosine similarity.

## Next Steps/Additional resources

#### Azure OpenAI Service

- LangChain Azure OpenAI [Docs](https://python.langchain.com/v0.2/docs/integrations/llms/azure_openai/)
- LangChain AzureChatOpenAI [Docs](https://python.langchain.com/v0.2/docs/integrations/chat/azure_chat_openai/)
- LangChain AzureChatOpenAI [API](https://python.langchain.com/v0.2/api_reference/openai/chat_models/langchain_openai.chat_models.azure.AzureChatOpenAI.html#langchain_openai.chat_models.azure.AzureChatOpenAI)
- LangChain AzureOpenAIEmbeddings [Docs](https://python.langchain.com/v0.2/docs/integrations/text_embedding/azureopenai/)
- LangChain AzureOpenAIEmbeddings [API](https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)

#### Azure AI Search

- LangChain Azure AI Search [Docs](https://python.langchain.com/v0.2/docs/integrations/vectorstores/azuresearch/)
- LangChain Azure AI Search [API](https://python.langchain.com/v0.2/api_reference/community/vectorstores/langchain_community.vectorstores.azuresearch.AzureSearch.html)

#### Azure Cosmos DB

- LangChain AzureCosmosDBMongovCore [Docs](https://python.langchain.com/v0.2/docs/integrations/vectorstores/azure_cosmos_db/)
- LangChain AzureCosmosDBMongovCore [API](https://python.langchain.com/v0.2/api_reference/community/vectorstores/langchain_community.vectorstores.azure_cosmos_db.AzureCosmosDBVectorSearch.html)
- LangChain AzureCosmosDBNoSQL [Docs](https://python.langchain.com/v0.2/docs/integrations/vectorstores/azure_cosmos_db_no_sql/)
- LangChain AzureCosmosDBNoSQL [API](https://python.langchain.com/v0.2/api_reference/community/vectorstores/langchain_community.vectorstores.azure_cosmos_db_no_sql.AzureCosmosDBNoSqlVectorSearch.html)

#### Azure AI Document Intelligence

- LangChain AzureAIDocumentIntelligenceLoader [Docs](https://python.langchain.com/docs/integrations/document_loaders/azure_document_intelligence/)
- LangChain AzureAIDocumentIntelligenceLoader [API](https://python.langchain.com/v0.2/api_reference/community/document_loaders/langchain_community.document_loaders.doc_intelligence.AzureAIDocumentIntelligenceLoader.html)

#### Azure AI Services/Azure AI Vision/Azure AI Speech

- LangChain AzureAIServicesToolkit [Docs](https://python.langchain.com/docs/integrations/tools/azure_ai_services/)
- LangChain AzureAIServicesToolkit [API](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.azure_ai_services.AzureAiServicesToolkit.html)

## References

- Azure Open AI Embeddings Section [source](https://learn.microsoft.com/en-us/azure/ai-services/openai/tutorials/embeddings?tabs=python-new%2Ccommand-line&pivots=programming-language-python)

- AzureChatOpenAI Structured Output [source](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.azure.AzureChatOpenAI.html)
