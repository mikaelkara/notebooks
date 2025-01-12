```
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# AI Agents for Engineers (Evolution of AI Agents)

<a target="_blank" href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/workshops/ai-agents/ai_agents_for_engineers.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

| | |
|-|-|
| Author(s) | [Kristopher Overholt](https://github.com/koverholt) [Holt Skinner](https://github.com/holtskinner)|

## Overview

This notebook demonstrates 3 different approaches to generating essays using the [Gemini API in Google AI Studio](https://ai.google.dev/gemini-api/docs). Each method illustrates a distinct paradigm for running AI Agents in differing levels of complexity.

1. Zero-Shot Approach with the Gemini API
2. Step-by-Step Approach With LangChain
3. Iterative, AI-Agent Approach with LangGraph

## Get started

### Install Gemini SDK and other required packages



```
%pip install --upgrade --user --quiet \
    google-generativeai \
    langgraph \
    langchain \
    langchain-google-genai \
    # langchain-google-vertexai \
    langchain-community \
    tavily-python \
    pydantic
```

### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.


```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>


### Configure API keys

Get API keys from [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key) and [Tavily](https://tavily.com/).


```
import os

os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"
os.environ["TAVILY_API_KEY"] = "YOUR_API_KEY_HERE"
```


```
# If your API Keys are in Colab Secrets
from google.colab import userdata

os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = userdata.get("TAVILY_API_KEY")
```

## Generating Essays Using a Zero-Shot Approach with the Gemini API

With just a single call to the `generate_content` method, users can create detailed, structured essays on any topic by leveraging state-of-the-art language models such as Gemini 1.5 Pro or Gemini 1.5 Flash.

<img src="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/workshops/ai-agents/1-prompt-essay.png?raw=1" width="350px">

### Import libraries


```
from IPython.display import Markdown, display
import google.generativeai as genai
```

### Load model


```
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
```


```
model = genai.GenerativeModel(model_name="gemini-1.5-pro-002")
```

### Make an API call to generate the essay


```
prompt = "Write a 3-paragraph essay about the application of heat transfer in modern data centers"
response = model.generate_content([prompt])
display(Markdown(response.text))
```


Modern data centers, the backbone of our digital world, generate tremendous amounts of heat due to the constant operation of servers and other hardware.  Efficient heat transfer is therefore critical not only to maintain optimal operating temperatures for reliable performance but also to minimize energy consumption associated with cooling.  Various heat transfer mechanisms are employed, including conduction, convection, and radiation.  At the component level, heat spreaders and heat sinks utilizing conductive materials like copper and aluminum draw heat away from processors and other heat-generating components.  Forced convection, achieved through fans and sophisticated airflow management systems, then removes this heat from within the server racks.  At the data center level, chilled water cooling systems, often utilizing principles of convection and heat exchangers, dissipate the collected heat to the external environment.

Advanced cooling techniques are increasingly crucial as data center power density continues to rise.  Liquid cooling, employing direct-to-chip or immersion cooling methods, offers significantly higher heat transfer efficiency compared to traditional air cooling.  These techniques involve circulating dielectric fluids, which have high thermal conductivity, directly over the heat-generating components or even submerging the servers entirely.  This allows for more effective heat removal, enabling denser packing of hardware and supporting higher performance processors.  Furthermore, waste heat captured from data centers can be repurposed for heating buildings or other industrial processes, increasing overall energy efficiency and promoting sustainability.

The ongoing development and implementation of innovative heat transfer solutions are essential for the sustainable growth of the digital economy.  As data centers become larger and more power-hungry, the focus on optimizing heat management will only intensify.  Research into new materials, more efficient cooling fluids, and advanced heat recovery systems will pave the way for greener, more powerful data centers.  Ultimately, effective heat transfer is not just about keeping servers cool, it’s about enabling the continued expansion of our digital world while minimizing its environmental impact.



---

However, what if we ask the model to write an essay about an event that happened more recently and the LLM doesn't inherently know about that event?


```
prompt = "Write a 3-paragraph essay about the impacts of Hurricane Helene and Hurricane Milton in 2024."
response = model.generate_content([prompt])
display(Markdown(response.text))
```


It is important to clarify that as of October 26, 2023, no Hurricanes Helene or Milton occurred in 2024.  Hurricane season officially ends on November 30th, and it's highly unusual for significant storms to form after October. Therefore, any discussion of the impacts of these fictitious hurricanes would be purely speculative.  To discuss hurricane impacts accurately, it is essential to refer to confirmed meteorological data and official reports following the actual occurrence of such events.  Using imagined scenarios can spread misinformation and create unnecessary anxiety.

Instead of focusing on non-existent storms, it is more productive to discuss the potential impacts hurricanes *could* have in general. Coastal regions are particularly vulnerable to the devastating effects of high winds, storm surge, and heavy rainfall. These can lead to widespread property damage, flooding, displacement of populations, and loss of life.  The economic impact can be severe, disrupting industries like tourism, fishing, and agriculture.  Moreover, hurricanes can cause long-term environmental damage, impacting ecosystems and coastal infrastructure.  Preparedness and mitigation efforts are crucial to minimizing the impact of future hurricanes.

Looking forward, accurate forecasting and timely warnings are vital for communities to prepare effectively for hurricanes.  Strengthening building codes, investing in resilient infrastructure, and developing comprehensive evacuation plans are crucial steps.  Public awareness campaigns can educate individuals on how to protect themselves and their property.  Furthermore, investing in research and technology to improve hurricane prediction models can enhance our ability to anticipate and respond to these powerful storms, ultimately reducing their impact and saving lives.



In this case, the model had no information about these recent events and was unable to write an effective essay.

## Generating Essays Using a Step-by-Step Approach With LangChain

This step demonstrates how to build an essay-writing pipeline using [LangChain](https://www.langchain.com/), the [Gemini API in Google AI Studio](https://ai.google.dev/gemini-api/docs), and [Tavily](https://tavily.com/) for search.

By combining these tools, we create a seamless workflow that plans an essay outline, performs web searches for relevant information, and generates a complete essay draft based on the collected data.

This solution showcases the power of chaining LLM models and external tools to tackle complex tasks with minimal human intervention, providing a robust approach to automated content generation.

<img src="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/workshops/ai-agents/2-langchain-essay.png?raw=1" width="550px">


### Import libraries


```
from IPython.display import Markdown, display
from langchain import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
```

### Initialize Gemini model & search tool


```
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
tavily_tool = TavilySearchResults(max_results=5)
```

### Define prompt templates and Runnables


```
# Planning: Create an outline for the essay
outline_template = ChatPromptTemplate.from_template(
    "Create a detailed outline for an essay on {topic}"
)


# Research: Web search
def research_fn(topic):
    response = tavily_tool.invoke({"query": topic})
    return "\n".join([f"- {result['content']}" for result in response])


# Writing: Write the essay based on outline and research
writing_template = ChatPromptTemplate.from_template(
    "Based on the following outline and research, write a 3-paragraph essay on '{topic}':\n\nOutline:\n{outline}\n\nResearch:\n{research}\n\nEssay:"
)
```

### Define the Runnable Chain using [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/how_to/#langchain-expression-language-lcel)


```
# Define individual chains
outline_chain = LLMChain(llm=model, prompt=outline_template)
writing_chain = LLMChain(llm=model, prompt=writing_template)

# Use the pipe operator to combine chains
chain = (
    outline_chain
    | (
        lambda result: {
            "topic": result["topic"],
            "outline": result["text"],
            "research": research_fn(result["topic"]),
        }
    )
    | writing_chain
    | (lambda result: result["text"])  # Extract the essay text from the final result
    | StrOutputParser()
)
```

### Generate the essay


```
topic = (
    "Write an essay about the impacts of Hurricane Helene and Hurricane Milton in 2024."
)
essay = chain.invoke({"topic": topic})
display(Markdown(essay))
```


The 2024 Atlantic hurricane season will be remembered for its unprecedented intensity, marked by a series of powerful storms that left a trail of destruction across the Caribbean and the southeastern United States.  Among these, Hurricanes Helene and Milton stand out, not only for their individual ferocity but also for the contrasting ways in which they impacted affected regions. This essay explores the hypothetical impacts of these two hurricanes, examining their economic, social, and environmental consequences to understand the complex challenges posed by such extreme weather events. While distinct in their paths and intensities, Helene and Milton left a lasting mark on the 2024 season, demonstrating the devastating power of nature and the ongoing challenges of coastal vulnerability.

Hurricane Helene, a powerful Category 4 hurricane, formed in the mid-Atlantic and carved a destructive path through the Caribbean before making landfall in North Carolina.  The storm's high winds and torrential rainfall caused widespread damage to infrastructure, leveling homes and businesses, crippling transportation networks, and devastating agricultural lands.  The economic toll was immense, with early estimates exceeding $100 billion.  Beyond the financial costs, Helene's impact rippled through communities, displacing thousands, straining emergency services, and leaving a deep scar on the social fabric of the region.  The storm surge inundated coastal ecosystems, causing significant erosion and damaging vital habitats like coral reefs and mangrove forests.  The influx of debris and runoff further polluted waterways, compounding the environmental damage.  North Carolina, home to roughly 45% of the region's small businesses, saw widespread economic disruption, impacting millions of employees and generating substantial revenue losses.

In contrast, Hurricane Milton, a rapidly intensifying Category 5 storm, emerged in the Gulf of Mexico and slammed into the Florida panhandle less than two weeks after Helene.  While Milton's smaller geographic impact concentrated its economic damage, estimated at $50 billion, the storm's extreme intensity resulted in a higher death toll, tragically reaching at least 250 lives.  Florida's tourism industry suffered a significant blow, and the combined impact of Helene and Milton overwhelmed emergency response systems, delaying relief efforts in both regions.  Milton's storm surge, exacerbated by elevated sea surface temperatures attributed to climate change, caused catastrophic flooding in coastal communities, surpassing the inundation experienced during Helene.  The rapid intensification of Milton, fueled by unusually warm ocean waters, underscored the potential link between climate change and the increasing frequency and intensity of extreme weather events.  The combined impact of these two storms highlighted the vulnerability of coastal regions and the urgent need for improved disaster preparedness and mitigation strategies.



## Generating Essays Using an Iterative, AI-Agent Approach with LangGraph

This section demonstrates how to build a [LangGraph](https://langchain-ai.github.io/langgraph/)-powered AI agent to generate, revise, and critique essays using large language models such as Google's [Gemini API in Google AI Studio](https://ai.google.dev/gemini-api/docs) or the [Gemini API in Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/overview). The LangGraph code was adapted from the awesome DeepLearning.AI course on [AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/).

By defining a structured state flow with nodes such as "Planner," "Research Plan," "Generate," "Reflect," and "Research Critique," the system iteratively creates an essay on a given topic, incorporates feedback, and provides research-backed insights.

<img src="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/workshops/ai-agents/3-langgraph-essay.png?raw=1" width="900px">

The workflow enables automated essay generation with revision controls, making it ideal for structured writing tasks or educational use cases. Additionally, the notebook uses external search tools to gather and integrate real-time information into the essay content.

### Import libraries


```
from typing import TypedDict

# Common libraries
from IPython.display import Image, Markdown, display

# LangChain and LangGraph components
from langchain_core.messages import HumanMessage, SystemMessage

# LangChain integrations for Gemini API in Google AI Studio and Vertex AI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

# Typing utilities for data validation and schema definitions
from pydantic.v1 import BaseModel

# Tavily client for performing web searches
from tavily import TavilyClient
```

### Initialize agent memory, agent state, and schema for search queries


```
# Initialize agent memory
memory = MemorySaver()


# Define the agent's state
class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: list[str]
    revision_number: int
    max_revisions: int


# Define a schema for search queries
class Queries(BaseModel):
    """Variants of query to search for"""

    queries: list[str]
```

### Initialize Gemini model and search tool

Remember to set the environment variables `GOOGLE_API_KEY` and `TAVILY_API_KEY`. And configure credentials for Vertex AI if you switch to it.


```
# Initialize Gemini API in Google AI Studio via LangChain
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

# Initialize Gemini API in Vertex AI via LangChain
# model = ChatVertexAI(model="gemini-1.5-pro-002", temperature=0)

# Initialize Tavily client for performing web searches
tavily = TavilyClient()
```

### Define prompt templates for each stage


```
PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay.
Write such an outline for the user provided topic. Give an outline of the essay along with any
relevant notes or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 3-paragraph essays.
Generate the best essay possible for the user's request and the initial outline.
If the user provides critique, respond with a revised version of your previous attempts.
Use Markdown formatting to specify a title and section headers for each paragraph.
Utilize all of the information below as needed:
---
{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission.
Generate critique and recommendations for the user's submission.
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can
be used when writing the following essay. Generate a list of search queries that will gather
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can
be used when making any requested revisions (as outlined below).
Generate a list of search queries that will gather any relevant information.
Only generate 3 queries max."""
```

### Define node functions for each stage


```
# Generate an outline for the essay


def plan_node(state: AgentState):
    messages = [SystemMessage(content=PLAN_PROMPT), HumanMessage(content=state["task"])]
    response = model.invoke(messages)
    return {"plan": response.content}


# Conducts research based on the generated plan and web search results
def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state["task"]),
        ]
    )
    content = state["content"] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response["results"]:
            content.append(r["content"])
    return {"content": content}


# Generates a draft based on the content and plan
def generation_node(state: AgentState):
    content = "\n\n".join(state["content"] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
    )
    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        user_message,
    ]
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }


# Provides feedback or critique on the draft
def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state["draft"]),
    ]
    response = model.invoke(messages)
    return {"critique": response.content}


# Conducts research based on the critique
def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state["critique"]),
        ]
    )
    content = state["content"] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response["results"]:
            content.append(r["content"])
    return {"content": content}


# Determines whether the critique and research cycle should
# continue based on the number of revisions
def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"
```

### Define and compile the graph


```
# Initialize the state graph
builder = StateGraph(AgentState)

# Add nodes for each step in the workflow
builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)

# Set the entry point of the workflow
builder.set_entry_point("planner")

# Add conditional edges for task continuation or end
builder.add_conditional_edges(
    "generate", should_continue, {END: END, "reflect": "reflect"}
)

# Define task sequence edges
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")

builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

# Compile the graph with memory state management
graph = builder.compile(checkpointer=memory)
```

### Show the compiled graph


```
Image(graph.get_graph().draw_mermaid_png())
```

### Run the agent - write on!


```
# Define the topic of the essay
ESSAY_TOPIC = "What were the impacts of Hurricane Helene and Hurricane Milton in 2024?"

# Define a thread configuration with a unique thread ID
thread = {"configurable": {"thread_id": "1"}}

# Stream through the graph execution with an initial task and state
for s in graph.stream(
    {
        "task": ESSAY_TOPIC,  # Initial task
        "max_revisions": 2,  # Maximum number of revisions allowed
        "revision_number": 1,  # Current revision number
        "content": [],  # Initial empty content list
    },
    thread,
):
    step = next(iter(s))
    display(Markdown(f"# {step}"))
    for key, content in s[step].items():
        if key == "revision_number":
            display(Markdown(f"**Revision Number**: {content}"))
        elif isinstance(content, list):
            for c in content:
                display(Markdown(c))
        else:
            display(Markdown(content))
    print("\n---\n")
```


# planner



Essay Outline: The Impacts of Hurricanes Helene and Milton in 2024

I. Introduction
    A. Briefly introduce the 2024 Atlantic hurricane season.
    B. Specifically mention Hurricanes Helene and Milton, highlighting their classifications (category, wind speeds, etc.).
    C. Thesis statement:  While both Helene and Milton reached significant intensities in the 2024 Atlantic hurricane season, their impacts differed considerably. Helene primarily affected [mention areas affected and type of impact, e.g., the Azores with high waves and heavy rainfall], while Milton posed a greater threat to [mention areas affected and type of impact, e.g., Bermuda with potential for storm surge and damaging winds], ultimately resulting in [overall comparative impact - e.g., more economic damage from Milton but greater disruption to shipping from Helene].

II. Hurricane Helene: A Tempest in the Atlantic
    A. Formation and Tracking: Detail Helene's path and intensification. Include dates and locations.  Use maps if possible.
    B. Impacts on Land: Discuss any landfalls or close approaches. Focus on the Azores.  Were there evacuations? What was the extent of damage (infrastructure, agriculture, etc.)?  Quantify impacts where possible (e.g., rainfall amounts, wind speeds recorded).
    C. Marine Impacts:  Helene was a powerful hurricane at sea. Discuss its impact on shipping routes, wave heights, and any other maritime concerns.
    D. Socioeconomic Consequences:  Analyze the economic and social disruption caused by Helene.  Did tourism suffer? Were there long-term recovery efforts?

III. Hurricane Milton: A Threat to Bermuda
    A. Formation and Tracking: Detail Milton's path and intensification. Include dates and locations. Use maps if possible.  Highlight any periods of rapid intensification.
    B. Impacts on Bermuda:  Focus on the specific impacts on Bermuda.  Did it make landfall? What preparations were made?  Discuss the extent of damage (infrastructure, coastal erosion, etc.). Quantify impacts where possible (e.g., rainfall amounts, wind speeds recorded, storm surge height).
    C. Impacts on Other Areas:  Did Milton affect any other regions besides Bermuda?  If so, briefly describe those impacts.
    D. Socioeconomic Consequences: Analyze the economic and social disruption caused by Milton.  What were the insurance costs?  Were there long-term recovery efforts?

IV. Comparison and Contrast
    A. Directly compare the impacts of Helene and Milton.  Use a table or chart if helpful.  Consider:
        1. Intensity and duration
        2. Geographic area affected
        3. Types of damage (wind, rain, surge)
        4. Economic costs
        5. Loss of life (if any)
    B.  Discuss any lessons learned from these two storms.  Did forecasting models perform well? Were emergency preparedness measures effective?

V. Conclusion
    A. Briefly summarize the overall impacts of both hurricanes.
    B. Reiterate the contrasting nature of their effects.
    C. Offer a concluding thought about the 2024 hurricane season in the context of these two storms.  Perhaps mention the increasing importance of hurricane preparedness in a changing climate (if relevant).


**Notes:**

* **Data Sources:** Rely on reputable sources like the National Hurricane Center (NHC), NOAA, and other meteorological agencies for accurate data on storm tracks, intensities, and impacts.
* **Visual Aids:** Incorporate maps showing the hurricanes' paths and graphs illustrating their intensities over time.  Images of the damage caused would also be beneficial.
* **Human Impact:**  While focusing on the physical impacts, remember to include the human element.  Discuss how people were affected and how communities responded.
* **Accuracy:** Double-check all facts and figures.  Hurricane data can sometimes be revised after the initial reports.  Use the most up-to-date information available.
* **Objectivity:** Maintain an objective tone throughout the essay.  Avoid sensationalizing the events.  Let the facts speak for themselves.



    
    ---
    
    


# research_plan



Near Perry, Florida, power flashes were seen as Hurricane Helene downed power lines on September 26, 2024. SVC/Simon Brewer Juston Drake High winds and storm surge caused bridges to close in the



Helene's toll: At least 100 dead, 2M customers without power amid water rescues in Florida, Georgia, Carolinas, Tennessee and Virginia Hurricane Helene, which left massive destruction along the Florida coast since making landfall Thursday, is now causing historic flooding, wide-ranging power outages, and other damage in an 800-mile northward path that's affecting the Carolinas, Tennessee, Georgia, and other states. Helene makes landfall at 11:10 p.m. near Perry, Florida, as a Category 4 storm with 140 mph-winds and large field of hurricane-force and tropical storm winds. 10Best USAT Wine Club Online Sports Betting Online Casino Reviewed Best-selling Booklist Southern Kitchen Jobs Sports Betting Sports Weekly Studio Gannett Classifieds Homefront Home Internet Blueprint Auto Insurance Pet Insurance Travel Insurance Credit Cards Banking Personal Loans LLC Formation Payroll Software



Maps and charts: Hurricane Milton’s impact across Florida | CNN Follow CNN CNN10 CNN 5 Things About CNN Work for CNN CNN  —  Hurricane Milton slammed into Florida’s Gulf Coast on Wednesday night, making landfall as a powerful Category 3 storm. The intense hurricane spawned tornadoes, dumped rain across much of the state, left millions without power and claimed at least 16 lives, including five people in St. Lucie County. Milton is the third hurricane to hit Florida this year — which has only happened during five other hurricane seasons since 1871. CNN10 CNN 5 Things About CNN Work for CNN Follow CNN



See the damage from Hurricane Milton in maps, photos and videos - The Washington Post The damage caused by Hurricane Milton in maps, photos and videos Hurricane Milton caused widespread damage across Florida with deadly tornadoes and storm surge. Milton’s quadruple-whammy of deadly tornadoes, heavy rain, hurricane-force wind and storm surge left damage from coast to coast, but the Tampa Bay area escaped the monster inundation meteorologists had feared. Hurricane Milton blew the roof off of Tropicana Field, home of the Tampa Bay Rays, in St. Petersburg, Fla. Hurricane Milton Milton, an extremely dangerous Category 3 hurricane, made landfall along Florida’s west coast Wednesday night.



AP News Alerts Keep your pulse on the news with breaking news alerts from The AP.AP Top 25 Poll Alerts Get email alerts for every college football Top 25 Poll release.The Morning Wire Our flagship newsletter breaks down the biggest headlines of the day.Ground Game Exclusive insights and key stories from the world of politics.Beyond the Story Executive Editor Julie Pace brings you behind the scenes of the AP newsroom.The Afternoon Wire Get caught up on what you may have missed throughout the day. AP & Elections AP & Elections



An analysis by the international World Weather Attribution, or WWA, initiative, released October 9, analyzed the role of climate change in contributing to Hurricane Helene’s intensification and its torrential rainfall, including as it moved inland across the Southern Appalachian Mountains. And, in a separate alert released October 7, Climate Central reported that elevated sea surface temperatures in the southwestern Gulf of Mexico were also behind the “explosive” increase in intensity of Hurricane Milton. The analysis found that the sea surface temperatures in the Gulf were made 400 to 800 times more likely over the past two weeks due to human-caused climate change. Analysis: Ocean temperatures warmed by climate change provided fuel for Hurricane Milton's extreme rapid intensification.


    
    ---
    
    


# generate



# The Contrasting Impacts of Hurricanes Helene and Milton in 2024

## A Tale of Two Storms

The 2024 Atlantic hurricane season witnessed the destructive forces of several powerful storms, two of the most notable being Hurricanes Helene and Milton. Helene, a major hurricane reaching Category 4 status with 140 mph winds, made landfall near Perry, Florida, on September 26.  Milton, a Category 3 hurricane, struck Florida's west coast on an unspecified date in 2024. While both storms brought significant destruction, their impacts differed considerably. Helene caused widespread damage across Florida, Georgia, the Carolinas, Tennessee, and Virginia, primarily through strong winds, storm surge, and flooding, leading to extensive power outages and over 100 fatalities. Milton, on the other hand, focused its wrath primarily on Florida, causing damage through tornadoes, heavy rain, and storm surge, with a death toll reaching at least 16.  Ultimately, Helene resulted in a broader swathe of destruction and loss of life across multiple states, while Milton's impact was more concentrated, though still devastating, within Florida.

## Helene's Trail of Destruction

Hurricane Helene's landfall near Perry, Florida, as a Category 4 hurricane marked a significant event in the 2024 season.  The storm's large wind field and powerful storm surge caused widespread damage across the region.  Downed power lines left two million customers without power, and bridges were closed due to high winds and surging waters.  The storm's impact extended far beyond Florida, causing historic flooding and damage in Georgia, the Carolinas, Tennessee, and Virginia.  Over 100 deaths were attributed to Helene, highlighting the storm's devastating power.  While specific details regarding evacuations and the extent of damage to infrastructure and agriculture are not available in the provided source material, the widespread power outages and flooding suggest significant disruption to daily life and economic activity.  The storm's impact on maritime activities is also not detailed in the provided information.

## Milton's Concentrated Fury

Hurricane Milton, while less intense than Helene at landfall, left a significant mark on Florida.  Making landfall as a Category 3 hurricane, Milton spawned tornadoes, dumped heavy rainfall, and generated a destructive storm surge.  The Tampa Bay area narrowly avoided the worst of the predicted storm surge, but other areas experienced significant damage.  The storm blew the roof off Tropicana Field in St. Petersburg and caused at least 16 fatalities, including five in St. Lucie County.  The provided source material does not specify the exact date of Milton's landfall or provide details on preparations made for the storm.  The extent of damage beyond the mentioned incidents is also not detailed, nor is information available on the storm's socioeconomic consequences, including insurance costs and long-term recovery efforts.  However, the available information paints a picture of a powerful storm causing significant damage and disruption within Florida.




**Revision Number**: 2


    
    ---
    
    


# reflect



This essay provides a decent starting point for comparing and contrasting Hurricanes Helene and Milton, but it needs significant expansion and refinement to be truly effective.  Here's a breakdown of the strengths and weaknesses, along with recommendations for improvement:

**Strengths:**

* **Clear Thesis:** The essay establishes a clear thesis about the contrasting impacts of the two hurricanes.
* **Organized Structure:** The use of headings and separate sections for each hurricane helps organize the information.
* **Factual Basis:** The essay presents factual information about the hurricanes' paths and impacts.

**Weaknesses:**

* **Lack of Depth and Detail:** The essay lacks specific details about the storms' impacts.  It mentions flooding, power outages, and fatalities, but doesn't delve into the specifics of the damage, the economic consequences, or the human stories.
* **Missing Information:** Key information is missing, such as the landfall date of Hurricane Milton, specific locations of major damage, evacuation efforts, and long-term recovery efforts.
* **Repetitive Language:**  Phrases like "significant damage" and "devastating power" are used repeatedly, making the writing feel monotonous.
* **Overreliance on Source Material Limitations:** The essay repeatedly mentions the limitations of the source material. While acknowledging limitations is important, constantly highlighting them weakens the essay.  Instead, seek out additional sources to fill in the gaps.
* **Lack of Visual Aids:** Maps, charts, or graphs would greatly enhance the essay by visually representing the hurricanes' paths and impacts.
* **Insufficient Analysis:** The essay describes the impacts but doesn't analyze *why* these differences occurred. Were there differences in preparedness, infrastructure, or the storms' characteristics that contributed to the varying levels of devastation?

**Recommendations for Improvement:**

1. **Expand on the Impacts:** Provide more specific details about the damage caused by each hurricane.  For example, instead of just saying "widespread flooding," describe the areas that were flooded, the depth of the floodwaters, and the specific consequences of the flooding (e.g., displacement of residents, damage to businesses, contamination of water supplies).  Quantify the damage whenever possible (e.g., number of homes destroyed, estimated cost of repairs).

2. **Fill in Missing Information:** Research and include the missing information, such as Milton's landfall date, specific locations of major damage, details about evacuation efforts, and the long-term recovery process for both storms.

3. **Diversify Language:** Use a wider range of vocabulary to avoid repetition and make the writing more engaging.

4. **Focus on Analysis:**  Go beyond simply describing the impacts and analyze the reasons behind the differences. Consider factors like storm intensity, population density in affected areas, preparedness measures, and geographical features.

5. **Incorporate Visual Aids:** Include maps showing the hurricanes' paths and the areas affected.  Charts or graphs could be used to compare the storms' strengths, fatalities, and economic costs.

6. **Consider Human Impact:** Include personal stories or anecdotes from people affected by the hurricanes to add a human dimension to the essay.

7. **Length:** Aim for a more substantial essay.  The current length is too brief to provide a comprehensive comparison.  A good target would be at least double the current length, possibly more depending on the level of detail you include.

8. **Cite Sources:**  Properly cite all sources used in your research.

By addressing these weaknesses and incorporating the recommendations, you can transform this essay into a much stronger and more informative piece of writing.



    
    ---
    
    


# research_critique



Near Perry, Florida, power flashes were seen as Hurricane Helene downed power lines on September 26, 2024. SVC/Simon Brewer Juston Drake High winds and storm surge caused bridges to close in the



Helene's toll: At least 100 dead, 2M customers without power amid water rescues in Florida, Georgia, Carolinas, Tennessee and Virginia Hurricane Helene, which left massive destruction along the Florida coast since making landfall Thursday, is now causing historic flooding, wide-ranging power outages, and other damage in an 800-mile northward path that's affecting the Carolinas, Tennessee, Georgia, and other states. Helene makes landfall at 11:10 p.m. near Perry, Florida, as a Category 4 storm with 140 mph-winds and large field of hurricane-force and tropical storm winds. 10Best USAT Wine Club Online Sports Betting Online Casino Reviewed Best-selling Booklist Southern Kitchen Jobs Sports Betting Sports Weekly Studio Gannett Classifieds Homefront Home Internet Blueprint Auto Insurance Pet Insurance Travel Insurance Credit Cards Banking Personal Loans LLC Formation Payroll Software



Maps and charts: Hurricane Milton’s impact across Florida | CNN Follow CNN CNN10 CNN 5 Things About CNN Work for CNN CNN  —  Hurricane Milton slammed into Florida’s Gulf Coast on Wednesday night, making landfall as a powerful Category 3 storm. The intense hurricane spawned tornadoes, dumped rain across much of the state, left millions without power and claimed at least 16 lives, including five people in St. Lucie County. Milton is the third hurricane to hit Florida this year — which has only happened during five other hurricane seasons since 1871. CNN10 CNN 5 Things About CNN Work for CNN Follow CNN



See the damage from Hurricane Milton in maps, photos and videos - The Washington Post The damage caused by Hurricane Milton in maps, photos and videos Hurricane Milton caused widespread damage across Florida with deadly tornadoes and storm surge. Milton’s quadruple-whammy of deadly tornadoes, heavy rain, hurricane-force wind and storm surge left damage from coast to coast, but the Tampa Bay area escaped the monster inundation meteorologists had feared. Hurricane Milton blew the roof off of Tropicana Field, home of the Tampa Bay Rays, in St. Petersburg, Fla. Hurricane Milton Milton, an extremely dangerous Category 3 hurricane, made landfall along Florida’s west coast Wednesday night.



AP News Alerts Keep your pulse on the news with breaking news alerts from The AP.AP Top 25 Poll Alerts Get email alerts for every college football Top 25 Poll release.The Morning Wire Our flagship newsletter breaks down the biggest headlines of the day.Ground Game Exclusive insights and key stories from the world of politics.Beyond the Story Executive Editor Julie Pace brings you behind the scenes of the AP newsroom.The Afternoon Wire Get caught up on what you may have missed throughout the day. AP & Elections AP & Elections



An analysis by the international World Weather Attribution, or WWA, initiative, released October 9, analyzed the role of climate change in contributing to Hurricane Helene’s intensification and its torrential rainfall, including as it moved inland across the Southern Appalachian Mountains. And, in a separate alert released October 7, Climate Central reported that elevated sea surface temperatures in the southwestern Gulf of Mexico were also behind the “explosive” increase in intensity of Hurricane Milton. The analysis found that the sea surface temperatures in the Gulf were made 400 to 800 times more likely over the past two weeks due to human-caused climate change. Analysis: Ocean temperatures warmed by climate change provided fuel for Hurricane Milton's extreme rapid intensification.



Helene's toll: At least 100 dead, 2M customers without power amid water rescues in Florida, Georgia, Carolinas, Tennessee and Virginia Hurricane Helene, which left massive destruction along the Florida coast since making landfall Thursday, is now causing historic flooding, wide-ranging power outages, and other damage in an 800-mile northward path that's affecting the Carolinas, Tennessee, Georgia, and other states. Helene makes landfall at 11:10 p.m. near Perry, Florida, as a Category 4 storm with 140 mph-winds and large field of hurricane-force and tropical storm winds. 10Best USAT Wine Club Online Sports Betting Online Casino Reviewed Best-selling Booklist Southern Kitchen Jobs Sports Betting Sports Weekly Studio Gannett Classifieds Homefront Home Internet Blueprint Auto Insurance Pet Insurance Travel Insurance Credit Cards Banking Personal Loans LLC Formation Payroll Software



Hurricane Helene has laid waste to the southeastern United States. Its sheer wind force and deadly floods left behind a path of destruction stretching over 500 miles from Florida to the Southern



See the damage from Hurricane Milton in maps, photos and videos - The Washington Post The damage caused by Hurricane Milton in maps, photos and videos Hurricane Milton caused widespread damage across Florida with deadly tornadoes and storm surge. Milton’s quadruple-whammy of deadly tornadoes, heavy rain, hurricane-force wind and storm surge left damage from coast to coast, but the Tampa Bay area escaped the monster inundation meteorologists had feared. Hurricane Milton blew the roof off of Tropicana Field, home of the Tampa Bay Rays, in St. Petersburg, Fla. Hurricane Milton Milton, an extremely dangerous Category 3 hurricane, made landfall along Florida’s west coast Wednesday night.



Milton brings strong winds and storm surge after making landfall near city of Siesta Key on Florida’s western coast. Hurricane Milton has made landfall in the US state of Florida, where it is expected to lash communities with devastating winds and a dangerous storm surge. Milton made landfall as an intense Category 3 storm on Wednesday evening near the city of Siesta Key on Florida’s west coast, the National Hurricane Center (NHC) reported. Fed by warm waters in the Gulf of Mexico, Milton became the third-fastest intensifying storm on record in the Atlantic Ocean, the NHC said on Monday, as it surged from a tropical storm to a Category 5 hurricane in less than 24 hours.



FEMA Continues Recovery Efforts Following Hurricanes Helene and Milton, over $1.2 Billion in Direct Assistance to Survivors | FEMA.gov FEMA personnel remain on the ground in communities across the Southeast conducting damage assessments, coordinating with local officials, and helping individuals apply for disaster assistance programs. More than 1,400 FEMA Disaster Survivor Assistance team members are in affected neighborhoods helping survivors apply for assistance and connecting them with additional state, local, federal and voluntary agency resources. Disaster survivors in certain areas of Georgia, Florida (Helene), Florida (Milton), North Carolina, South Carolina, Tennessee and Virginia can begin their recovery process by applying for federal assistance through FEMA. FEMA now has 75 Disaster Recovery Centers open throughout the hurricane affected communities.



WASHINGTON – As communities across the Southeast recover from the devastation caused by Hurricanes Milton and Helene, FEMA and the federal family continue to support those affected, work side by side with state and local officials to assist survivors, and coordinate recovery operations. FEMA Disaster Survivor Assistance Teams are on the ground in neighborhoods across the affected counties continuing to help survivors apply for FEMA assistance and connect them with additional state, local, federal and voluntary agency resources. FEMA Disaster Survivor Assistance Teams are on the ground in neighborhoods across the affected counties helping survivors apply for FEMA assistance and connecting them with additional state, local, federal and voluntary agency resources.


    
    ---
    
    


# generate



# The Impacts of Hurricanes Helene and Milton in 2024

## A Season of Contrasts

The 2024 Atlantic hurricane season witnessed the destructive power of two major hurricanes, Helene and Milton. Helene, a powerful Category 4 hurricane with 140 mph winds, made landfall near Perry, Florida, on September 26.  Milton, a rapidly intensifying Category 3 hurricane, struck Florida's west coast near Siesta Key on Wednesday night. While both storms brought significant impacts to the southeastern United States, their effects differed considerably. Helene caused widespread destruction across Florida, Georgia, the Carolinas, Tennessee, and Virginia, primarily through flooding, power outages, and high winds. Milton, while also causing damage across Florida, including tornadoes and storm surge, focused its most intense impacts on the western coast. Ultimately, Helene resulted in a higher death toll and more widespread power outages, while Milton caused significant structural damage and economic disruption in a more concentrated area.

## Helene's Trail of Destruction

Hurricane Helene's path began in the Atlantic, rapidly intensifying before making landfall in Florida.  The storm's 140 mph winds and torrential rainfall caused widespread damage, downing power lines, flooding communities, and forcing bridge closures.  Over two million customers lost power across Florida, Georgia, the Carolinas, Tennessee, and Virginia.  Tragically, at least 100 deaths were attributed to Helene, primarily due to flooding and falling debris.  The storm's impact extended far inland, causing historic flooding in the Southern Appalachian Mountains.  While the exact economic cost is still being assessed, the widespread damage to infrastructure, agriculture, and homes suggests a substantial financial burden.  The storm also disrupted daily life, forcing evacuations and school closures across multiple states.

## Milton's Western Blow

Hurricane Milton's rapid intensification in the Gulf of Mexico fueled its destructive potential.  Upon making landfall near Siesta Key, Florida, as a Category 3 hurricane, Milton unleashed strong winds, storm surge, and tornadoes.  The storm caused significant damage along Florida's western coast, including blowing the roof off of Tropicana Field in St. Petersburg. While the Tampa Bay area avoided the worst of the storm surge, coastal communities still experienced flooding and erosion.  Milton's strong winds also caused widespread power outages and downed trees.  The storm's rapid intensification, attributed to unusually warm ocean temperatures exacerbated by climate change, highlighted the increasing threat of powerful hurricanes.  FEMA has been actively involved in recovery efforts for both Helene and Milton, providing over $1.2 billion in direct assistance to survivors.  The long-term economic and social consequences of Milton are still being evaluated, but the storm's impact on infrastructure and businesses will undoubtedly require significant recovery efforts.




**Revision Number**: 3


    
    ---
    
    

### Output the final draft of the essay


```
display(Markdown(s["generate"]["draft"]))
```


# Impacts of Hurricanes Helene and Milton in 2024

## A Season of Destruction

The 2024 Atlantic hurricane season proved to be a destructive one, marked by several powerful storms. Among the most impactful were Hurricanes Helene and Milton, which left a trail of devastation across different regions. Helene, a Category 4 hurricane, made landfall near Perry, Florida, on September 26th with 140 mph winds.  Milton followed less than two weeks later, striking Florida as well. While both storms posed significant threats, their impacts differed considerably due to variations in their landfall locations, intensities at landfall, and the preparedness measures in the affected regions.

## Helene's Havoc

Helene's impact was widespread, affecting Florida, Georgia, the Carolinas, Tennessee, and Virginia.  In Florida, the storm's intense winds and storm surge caused significant damage.  Downed power lines left two million customers without power, and widespread flooding necessitated numerous water rescues.  Coastal areas experienced severe erosion, and infrastructure, including bridges, suffered substantial damage.  The economic toll was significant, with early estimates suggesting billions of dollars in losses.  Beyond Florida, Helene's remnants brought heavy rainfall and flooding further inland, causing disruptions and damage across a wide swath of the southeastern United States.  While early warnings and evacuation orders were issued, the rapid intensification of the storm challenged preparedness efforts, and some areas were caught off guard by the storm's strength.

## Milton's Aftermath

Hurricane Milton, while also impactful, presented a different set of challenges.  Making landfall in Florida shortly after Helene, Milton exacerbated existing damage and further strained resources.  The storm spawned a confirmed EF-1 tornado in Cocoa Beach, damaging buildings and infrastructure.  Milton's flooding compounded the issues created by Helene, inundating areas still recovering from the previous storm.  The close succession of these two hurricanes created unique challenges for emergency response and recovery efforts.  Resources were stretched thin, and communities struggled to cope with the cumulative effects of the two storms.  The economic impact of Milton, estimated at $50 billion, added to the already substantial costs associated with Helene.  The combined impact of these storms highlighted the importance of robust infrastructure and the need for coordinated disaster response strategies.




## Additional Resources

- [Google Cloud Generative AI repository on GitHub](https://github.com/GoogleCloudPlatform/generative-ai/)
- [Gemini API in Google AI Studio](https://ai.google.dev/gemini-api/docs)
- [Gemini API in Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/overview)
- [LangGraph tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [DeepLearning.AI course on AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)
