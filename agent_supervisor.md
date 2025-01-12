# Multi-agent supervisor

The [previous example](../multi-agent-collaboration) routed messages automatically based on the output of the initial researcher agent.

We can also choose to use an [LLM to orchestrate](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#supervisor) the different agents.

Below, we will create an agent group, with an agent supervisor to help delegate tasks.

![diagram](8ee0a8ce-f0a8-4019-b5bf-b20933e40956.png)

To simplify the code in each agent node, we will use LangGraph's prebuilt [create_react_agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent). This and other "advanced agent" notebooks are designed to show how you can implement certain design patterns in LangGraph. If the pattern suits your needs, we recommend combining it with some of the other fundamental patterns described elsewhere in the docs for best performance.

## Setup

First, let's install required packages and set our API keys


```python
%%capture --no-stderr
%pip install -U langgraph langchain langchain_openai langchain_experimental langsmith pandas
```


```python
import getpass
import os


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Create tools

For this example, you will make an agent to do web research with a search engine, and one agent to create plots. Define the tools they'll use below:


```python
from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()
```

## Helper Utilities

Define a helper function that we will use to create the nodes in the graph - it takes care of converting the agent response to a human message. This is important because that is how we will add it the global state of the graph


```python
from langchain_core.messages import HumanMessage


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }
```

### Create Agent Supervisor

It will use function calling to choose the next worker node OR finish processing.


```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal

members = ["Researcher", "Coder"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members


class routeResponse(BaseModel):
    next: Literal[*options]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))


llm = ChatOpenAI(model="gpt-4o")


def supervisor_agent(state):
    supervisor_chain = prompt | llm.with_structured_output(routeResponse)
    return supervisor_chain.invoke(state)
```

## Construct Graph

We're ready to start building the graph. Below, define the state and worker nodes using the function we just defined.


```python
import functools
import operator
from typing import Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


research_agent = create_react_agent(llm, tools=[tavily_tool])
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
code_agent = create_react_agent(llm, tools=[python_repl_tool])
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_agent)
```

Now connect all the edges in the graph.


```python
for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.add_edge(START, "supervisor")

graph = workflow.compile()
```

## Invoke the team

With the graph created, we can now invoke it and see how it performs!


```python
for s in graph.stream(
    {
        "messages": [
            HumanMessage(content="Code hello world and print it to the terminal")
        ]
    }
):
    if "__end__" not in s:
        print(s)
        print("----")
```

    {'supervisor': {'next': 'Coder'}}
    ----
    

    Python REPL can execute arbitrary code. Use with caution.
    

    {'Coder': {'messages': [HumanMessage(content='The code to print "Hello, World!" in the terminal has been executed successfully. Here is the output:\n\n```\nHello, World!\n```', additional_kwargs={}, response_metadata={}, name='Coder')]}}
    ----
    {'supervisor': {'next': 'FINISH'}}
    ----
    


```python
for s in graph.stream(
    {"messages": [HumanMessage(content="Write a brief research report on pikas.")]},
    {"recursion_limit": 100},
):
    if "__end__" not in s:
        print(s)
        print("----")
```

    {'supervisor': {'next': 'Researcher'}}
    ----
    {'Researcher': {'messages': [HumanMessage(content='### Research Report on Pikas\n\n#### Introduction\nPikas are small, herbivorous mammals belonging to the family Ochotonidae, closely related to rabbits and hares. These animals are known for their distinctive high-pitched calls and are often found in cold, mountainous regions across Asia, North America, and parts of Europe.\n\n#### Habitat and Behavior\nPikas primarily inhabit talus slopes and alpine meadows, often at elevations ranging from 2,500 to over 13,000 feet. These environments provide the necessary rock crevices and vegetation required for their survival. Pikas are diurnal and exhibit two main foraging behaviors: direct consumption of plants and the collection of vegetation into "haypiles" for winter storage. Unlike many small mammals, pikas do not hibernate and remain active throughout the winter, relying on these haypiles for sustenance.\n\n#### Diet and Feeding Habits\nPikas are generalist herbivores, feeding on a variety of grasses, forbs, and small shrubs. They have a highly developed behavior known as "haying," where they collect and store plant material during the summer months to ensure a food supply during the harsh winter. This behavior is crucial for their survival, as the stored hay provides the necessary nutrients when fresh vegetation is scarce.\n\n#### Reproduction and Lifecycle\nPikas have a relatively short lifespan, averaging around three years. They typically breed once or twice a year, with a gestation period of roughly 30 days. Females usually give birth to litters of two to six young. The young are weaned and become independent within a month, reaching sexual maturity by the following spring.\n\n#### Conservation Status\nThe conservation status of pikas varies by region and species. The American pika (Ochotona princeps), found in the mountains of western North America, is particularly vulnerable to climate change. Rising temperatures and reduced snowpack threaten their habitat, forcing pikas to move to higher elevations or face local extirpation. Despite these challenges, the American pika is not currently listed under the US Endangered Species Act, although several studies indicate localized population declines.\n\n#### Conclusion\nPikas are fascinating creatures that play a vital role in their alpine ecosystems. Their unique behaviors, such as haying, and their sensitivity to climate change make them important indicators of environmental health. Continued research and conservation efforts are essential to ensure the survival of these small but significant mammals in the face of global climatic shifts.\n\n#### References\n1. Wikipedia - Pika: [Link](https://en.wikipedia.org/wiki/Pika)\n2. Wikipedia - American Pika: [Link](https://en.wikipedia.org/wiki/American_pika)\n3. Animal Spot - American Pika: [Link](https://www.animalspot.net/american-pika.html)\n4. Animalia - American Pika: [Link](https://animalia.bio/index.php/american-pika)\n5. National Park Service - Pikas Resource Brief: [Link](https://www.nps.gov/articles/pikas-brief.htm)\n6. Alaska Department of Fish and Game - Pikas: [Link](https://www.adfg.alaska.gov/static/education/wns/pikas.pdf)\n7. NatureMapping Foundation - American Pika: [Link](http://naturemappingfoundation.org/natmap/facts/american_pika_712.html)\n8. USDA Forest Service - Conservation Status of Pikas: [Link](https://www.fs.usda.gov/psw/publications/millar/psw_2022_millar002.pdf)', additional_kwargs={}, response_metadata={}, name='Researcher')]}}
    ----
    {'supervisor': {'next': 'Coder'}}
    ----
    {'Coder': {'messages': [HumanMessage(content='### Research Report on Pikas\n\n#### Introduction\nPikas are small, herbivorous mammals belonging to the family Ochotonidae, closely related to rabbits and hares. These animals are known for their distinctive high-pitched calls and are often found in cold, mountainous regions across Asia, North America, and parts of Europe.\n\n#### Habitat and Behavior\nPikas primarily inhabit talus slopes and alpine meadows, often at elevations ranging from 2,500 to over 13,000 feet. These environments provide the necessary rock crevices and vegetation required for their survival. Pikas are diurnal and exhibit two main foraging behaviors: direct consumption of plants and the collection of vegetation into "haypiles" for winter storage. Unlike many small mammals, pikas do not hibernate and remain active throughout the winter, relying on these haypiles for sustenance.\n\n#### Diet and Feeding Habits\nPikas are generalist herbivores, feeding on a variety of grasses, forbs, and small shrubs. They have a highly developed behavior known as "haying," where they collect and store plant material during the summer months to ensure a food supply during the harsh winter. This behavior is crucial for their survival, as the stored hay provides the necessary nutrients when fresh vegetation is scarce.\n\n#### Reproduction and Lifecycle\nPikas have a relatively short lifespan, averaging around three years. They typically breed once or twice a year, with a gestation period of roughly 30 days. Females usually give birth to litters of two to six young. The young are weaned and become independent within a month, reaching sexual maturity by the following spring.\n\n#### Conservation Status\nThe conservation status of pikas varies by region and species. The American pika (Ochotona princeps), found in the mountains of western North America, is particularly vulnerable to climate change. Rising temperatures and reduced snowpack threaten their habitat, forcing pikas to move to higher elevations or face local extirpation. Despite these challenges, the American pika is not currently listed under the US Endangered Species Act, although several studies indicate localized population declines.\n\n#### Conclusion\nPikas are fascinating creatures that play a vital role in their alpine ecosystems. Their unique behaviors, such as haying, and their sensitivity to climate change make them important indicators of environmental health. Continued research and conservation efforts are essential to ensure the survival of these small but significant mammals in the face of global climatic shifts.\n\n#### References\n1. Wikipedia - Pika: [Link](https://en.wikipedia.org/wiki/Pika)\n2. Wikipedia - American Pika: [Link](https://en.wikipedia.org/wiki/American_pika)\n3. Animal Spot - American Pika: [Link](https://www.animalspot.net/american-pika.html)\n4. Animalia - American Pika: [Link](https://animalia.bio/index.php/american-pika)\n5. National Park Service - Pikas Resource Brief: [Link](https://www.nps.gov/articles/pikas-brief.htm)\n6. Alaska Department of Fish and Game - Pikas: [Link](https://www.adfg.alaska.gov/static/education/wns/pikas.pdf)\n7. NatureMapping Foundation - American Pika: [Link](http://naturemappingfoundation.org/natmap/facts/american_pika_712.html)\n8. USDA Forest Service - Conservation Status of Pikas: [Link](https://www.fs.usda.gov/psw/publications/millar/psw_2022_millar002.pdf)', additional_kwargs={}, response_metadata={}, name='Coder')]}}
    ----
    {'supervisor': {'next': 'FINISH'}}
    ----
    
