# How to track token usage in ChatModels

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Chat models](/docs/concepts/chat_models)

:::

Tracking token usage to calculate cost is an important part of putting your app in production. This guide goes over how to obtain this information from your LangChain model calls.

This guide requires `langchain-anthropic` and `langchain-openai >= 0.1.9`.


```python
%pip install -qU langchain-anthropic langchain-openai
```

## Using LangSmith

You can use [LangSmith](https://www.langchain.com/langsmith) to help track token usage in your LLM application. See the [LangSmith quick start guide](https://docs.smith.langchain.com/).

## Using AIMessage.usage_metadata

A number of model providers return token usage information as part of the chat generation response. When available, this information will be included on the `AIMessage` objects produced by the corresponding model.

LangChain `AIMessage` objects include a [usage_metadata](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage.usage_metadata) attribute. When populated, this attribute will be a [UsageMetadata](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.UsageMetadata.html) dictionary with standard keys (e.g., `"input_tokens"` and `"output_tokens"`).

Examples:

**OpenAI**:


```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
openai_response = llm.invoke("hello")
openai_response.usage_metadata
```




    {'input_tokens': 8, 'output_tokens': 9, 'total_tokens': 17}



**Anthropic**:


```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-haiku-20240307")
anthropic_response = llm.invoke("hello")
anthropic_response.usage_metadata
```




    {'input_tokens': 8, 'output_tokens': 12, 'total_tokens': 20}



### Using AIMessage.response_metadata

Metadata from the model response is also included in the AIMessage [response_metadata](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage.response_metadata) attribute. These data are typically not standardized. Note that different providers adopt different conventions for representing token counts:


```python
print(f'OpenAI: {openai_response.response_metadata["token_usage"]}\n')
print(f'Anthropic: {anthropic_response.response_metadata["usage"]}')
```

    OpenAI: {'completion_tokens': 9, 'prompt_tokens': 8, 'total_tokens': 17}
    
    Anthropic: {'input_tokens': 8, 'output_tokens': 12}
    

### Streaming

Some providers support token count metadata in a streaming context.

#### OpenAI

For example, OpenAI will return a message [chunk](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessageChunk.html) at the end of a stream with token usage information. This behavior is supported by `langchain-openai >= 0.1.9` and can be enabled by setting `stream_usage=True`. This attribute can also be set when `ChatOpenAI` is instantiated.

:::note
By default, the last message chunk in a stream will include a `"finish_reason"` in the message's `response_metadata` attribute. If we include token usage in streaming mode, an additional chunk containing usage metadata will be added to the end of the stream, such that `"finish_reason"` appears on the second to last message chunk.
:::



```python
llm = ChatOpenAI(model="gpt-4o-mini")

aggregate = None
for chunk in llm.stream("hello", stream_usage=True):
    print(chunk)
    aggregate = chunk if aggregate is None else aggregate + chunk
```

    content='' id='run-adb20c31-60c7-43a2-99b2-d4a53ca5f623'
    content='Hello' id='run-adb20c31-60c7-43a2-99b2-d4a53ca5f623'
    content='!' id='run-adb20c31-60c7-43a2-99b2-d4a53ca5f623'
    content=' How' id='run-adb20c31-60c7-43a2-99b2-d4a53ca5f623'
    content=' can' id='run-adb20c31-60c7-43a2-99b2-d4a53ca5f623'
    content=' I' id='run-adb20c31-60c7-43a2-99b2-d4a53ca5f623'
    content=' assist' id='run-adb20c31-60c7-43a2-99b2-d4a53ca5f623'
    content=' you' id='run-adb20c31-60c7-43a2-99b2-d4a53ca5f623'
    content=' today' id='run-adb20c31-60c7-43a2-99b2-d4a53ca5f623'
    content='?' id='run-adb20c31-60c7-43a2-99b2-d4a53ca5f623'
    content='' response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-mini'} id='run-adb20c31-60c7-43a2-99b2-d4a53ca5f623'
    content='' id='run-adb20c31-60c7-43a2-99b2-d4a53ca5f623' usage_metadata={'input_tokens': 8, 'output_tokens': 9, 'total_tokens': 17}
    

Note that the usage metadata will be included in the sum of the individual message chunks:


```python
print(aggregate.content)
print(aggregate.usage_metadata)
```

    Hello! How can I assist you today?
    {'input_tokens': 8, 'output_tokens': 9, 'total_tokens': 17}
    

To disable streaming token counts for OpenAI, set `stream_usage` to False, or omit it from the parameters:


```python
aggregate = None
for chunk in llm.stream("hello"):
    print(chunk)
```

    content='' id='run-8e758550-94b0-4cca-a298-57482793c25d'
    content='Hello' id='run-8e758550-94b0-4cca-a298-57482793c25d'
    content='!' id='run-8e758550-94b0-4cca-a298-57482793c25d'
    content=' How' id='run-8e758550-94b0-4cca-a298-57482793c25d'
    content=' can' id='run-8e758550-94b0-4cca-a298-57482793c25d'
    content=' I' id='run-8e758550-94b0-4cca-a298-57482793c25d'
    content=' assist' id='run-8e758550-94b0-4cca-a298-57482793c25d'
    content=' you' id='run-8e758550-94b0-4cca-a298-57482793c25d'
    content=' today' id='run-8e758550-94b0-4cca-a298-57482793c25d'
    content='?' id='run-8e758550-94b0-4cca-a298-57482793c25d'
    content='' response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-mini'} id='run-8e758550-94b0-4cca-a298-57482793c25d'
    

You can also enable streaming token usage by setting `stream_usage` when instantiating the chat model. This can be useful when incorporating chat models into LangChain [chains](/docs/concepts/lcel): usage metadata can be monitored when [streaming intermediate steps](/docs/how_to/streaming#using-stream-events) or using tracing software such as [LangSmith](https://docs.smith.langchain.com/).

See the below example, where we return output structured to a desired schema, but can still observe token usage streamed from intermediate steps.


```python
from pydantic import BaseModel, Field


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    stream_usage=True,
)
# Under the hood, .with_structured_output binds tools to the
# chat model and appends a parser.
structured_llm = llm.with_structured_output(Joke)

async for event in structured_llm.astream_events("Tell me a joke", version="v2"):
    if event["event"] == "on_chat_model_end":
        print(f'Token usage: {event["data"]["output"].usage_metadata}\n')
    elif event["event"] == "on_chain_end":
        print(event["data"]["output"])
    else:
        pass
```

    Token usage: {'input_tokens': 79, 'output_tokens': 23, 'total_tokens': 102}
    
    setup='Why was the math book sad?' punchline='Because it had too many problems.'
    

Token usage is also visible in the corresponding [LangSmith trace](https://smith.langchain.com/public/fe6513d5-7212-4045-82e0-fefa28bc7656/r) in the payload from the chat model.

## Using callbacks

There are also some API-specific callback context managers that allow you to track token usage across multiple calls. They are currently only implemented for the OpenAI API and Bedrock Anthropic API, and are available in `langchain-community`:


```python
%pip install -qU langchain-community
```

### OpenAI

Let's first look at an extremely simple example of tracking token usage for a single Chat model call.


```python
from langchain_community.callbacks.manager import get_openai_callback

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    stream_usage=True,
)

with get_openai_callback() as cb:
    result = llm.invoke("Tell me a joke")
    print(cb)
```

    Tokens Used: 27
    	Prompt Tokens: 11
    	Completion Tokens: 16
    Successful Requests: 1
    Total Cost (USD): $2.95e-05
    

Anything inside the context manager will get tracked. Here's an example of using it to track multiple calls in sequence.


```python
with get_openai_callback() as cb:
    result = llm.invoke("Tell me a joke")
    result2 = llm.invoke("Tell me a joke")
    print(cb.total_tokens)
```

    54
    


```python
with get_openai_callback() as cb:
    for chunk in llm.stream("Tell me a joke"):
        pass
    print(cb)
```

    Tokens Used: 27
    	Prompt Tokens: 11
    	Completion Tokens: 16
    Successful Requests: 1
    Total Cost (USD): $2.95e-05
    

If a chain or agent with multiple steps in it is used, it will track all those steps.


```python
%pip install -qU langchain langchain-aws wikipedia
```


```python
from langchain.agents import AgentExecutor, create_tool_calling_agent, load_tools
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a helpful assistant"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
tools = load_tools(["wikipedia"])
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```


```python
with get_openai_callback() as cb:
    response = agent_executor.invoke(
        {
            "input": "What's a hummingbird's scientific name and what's the fastest bird species?"
        }
    )
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `wikipedia` with `{'query': 'hummingbird scientific name'}`
    
    
    [0m[36;1m[1;3mPage: Hummingbird
    Summary: Hummingbirds are birds native to the Americas and comprise the biological family Trochilidae. With approximately 366 species and 113 genera, they occur from Alaska to Tierra del Fuego, but most species are found in Central and South America. As of 2024, 21 hummingbird species are listed as endangered or critically endangered, with numerous species declining in population.
    Hummingbirds have varied specialized characteristics to enable rapid, maneuverable flight: exceptional metabolic capacity, adaptations to high altitude, sensitive visual and communication abilities, and long-distance migration in some species. Among all birds, male hummingbirds have the widest diversity of plumage color, particularly in blues, greens, and purples. Hummingbirds are the smallest mature birds, measuring 7.5–13 cm (3–5 in) in length. The smallest is the 5 cm (2.0 in) bee hummingbird, which weighs less than 2.0 g (0.07 oz), and the largest is the 23 cm (9 in) giant hummingbird, weighing 18–24 grams (0.63–0.85 oz). Noted for long beaks, hummingbirds are specialized for feeding on flower nectar, but all species also consume small insects.
    They are known as hummingbirds because of the humming sound created by their beating wings, which flap at high frequencies audible to other birds and humans. They hover at rapid wing-flapping rates, which vary from around 12 beats per second in the largest species to 80 per second in small hummingbirds.
    Hummingbirds have the highest mass-specific metabolic rate of any homeothermic animal. To conserve energy when food is scarce and at night when not foraging, they can enter torpor, a state similar to hibernation, and slow their metabolic rate to 1⁄15 of its normal rate. While most hummingbirds do not migrate, the rufous hummingbird has one of the longest migrations among birds, traveling twice per year between Alaska and Mexico, a distance of about 3,900 miles (6,300 km).
    Hummingbirds split from their sister group, the swifts and treeswifts, around 42 million years ago. The oldest known fossil hummingbird is Eurotrochilus, from the Rupelian Stage of Early Oligocene Europe.
    
    Page: Rufous hummingbird
    Summary: The rufous hummingbird (Selasphorus rufus) is a small hummingbird, about 8 cm (3.1 in) long with a long, straight and slender bill. These birds are known for their extraordinary flight skills, flying 2,000 mi (3,200 km) during their migratory transits. It is one of nine species in the genus Selasphorus.
    
    
    
    Page: Allen's hummingbird
    Summary: Allen's hummingbird (Selasphorus sasin) is a species of hummingbird that breeds in the western United States. It is one of seven species in the genus Selasphorus.[0m[32;1m[1;3m
    Invoking: `wikipedia` with `{'query': 'fastest bird species'}`
    
    
    [0m[36;1m[1;3mPage: List of birds by flight speed
    Summary: This is a list of the fastest flying birds in the world. A bird's velocity is necessarily variable; a hunting bird will reach much greater speeds while diving to catch prey than when flying horizontally. The bird that can achieve the greatest airspeed is the peregrine falcon (Falco peregrinus), able to exceed 320 km/h (200 mph) in its dives. A close relative of the common swift, the white-throated needletail (Hirundapus caudacutus), is commonly reported as the fastest bird in level flight with a reported top speed of 169 km/h (105 mph). This record remains unconfirmed as the measurement methods have never been published or verified. The record for the fastest confirmed level flight by a bird is 111.5 km/h (69.3 mph) held by the common swift.
    
    Page: Fastest animals
    Summary: This is a list of the fastest animals in the world, by types of animal.
    
    Page: Falcon
    Summary: Falcons () are birds of prey in the genus Falco, which includes about 40 species. Falcons are widely distributed on all continents of the world except Antarctica, though closely related raptors did occur there in the Eocene.
    Adult falcons have thin, tapered wings, which enable them to fly at high speed and change direction rapidly. Fledgling falcons, in their first year of flying, have longer flight feathers, which make their configuration more like that of a general-purpose bird such as a broad wing. This makes flying easier while learning the exceptional skills required to be effective hunters as adults.
    The falcons are the largest genus in the Falconinae subfamily of Falconidae, which itself also includes another subfamily comprising caracaras and a few other species. All these birds kill with their beaks, using a tomial "tooth" on the side of their beaks—unlike the hawks, eagles, and other birds of prey in the Accipitridae, which use their feet.
    The largest falcon is the gyrfalcon at up to 65 cm in length.  The smallest falcon species is the pygmy falcon, which measures just 20 cm.  As with hawks and owls, falcons exhibit sexual dimorphism, with the females typically larger than the males, thus allowing a wider range of prey species.
    Some small falcons with long, narrow wings are called "hobbies" and some which hover while hunting are called "kestrels".
    As is the case with many birds of prey, falcons have exceptional powers of vision; the visual acuity of one species has been measured at 2.6 times that of a normal human. Peregrine falcons have been recorded diving at speeds of 320 km/h (200 mph), making them the fastest-moving creatures on Earth; the fastest recorded dive attained a vertical speed of 390 km/h (240 mph).[0m[32;1m[1;3mThe scientific name for a hummingbird is Trochilidae. The fastest bird species in level flight is the common swift, which holds the record for the fastest confirmed level flight by a bird at 111.5 km/h (69.3 mph). The peregrine falcon is known to exceed speeds of 320 km/h (200 mph) in its dives, making it the fastest bird in terms of diving speed.[0m
    
    [1m> Finished chain.[0m
    Total Tokens: 1675
    Prompt Tokens: 1538
    Completion Tokens: 137
    Total Cost (USD): $0.0009745000000000001
    

### Bedrock Anthropic

The `get_bedrock_anthropic_callback` works very similarly:


```python
from langchain_aws import ChatBedrock
from langchain_community.callbacks.manager import get_bedrock_anthropic_callback

llm = ChatBedrock(model_id="anthropic.claude-v2")

with get_bedrock_anthropic_callback() as cb:
    result = llm.invoke("Tell me a joke")
    result2 = llm.invoke("Tell me a joke")
    print(cb)
```

    Tokens Used: 96
    	Prompt Tokens: 26
    	Completion Tokens: 70
    Successful Requests: 2
    Total Cost (USD): $0.001888
    

## Next steps

You've now seen a few examples of how to track token usage for supported providers.

Next, check out the other how-to guides chat models in this section, like [how to get a model to return structured output](/docs/how_to/structured_output) or [how to add caching to your chat models](/docs/how_to/chat_model_caching).


```python

```
