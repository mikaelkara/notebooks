# Have several agents collaborate in a multi-agent hierarchy 🤖🤝🤖
_Authored by: [Aymeric Roucher](https://huggingface.co/m-ric)_

> This tutorial is advanced. You should have notions from [this other cookbook](agents) first!

In this notebook we will make a **multi-agent web browser: an agentic system with several agents collaborating to solve problems using the web!**

It will be a simple hierarchy, using a `ManagedAgent` object to wrap the managed web search agent:

```
              +----------------+
              | Manager agent  |
              +----------------+
                       |
        _______________|______________
       |                              |
  Code interpreter   +--------------------------------+
       tool          |         Managed agent          |
                     |      +------------------+      |
                     |      | Web Search agent |      |
                     |      +------------------+      |
                     |         |            |         |
                     |  Web Search tool     |         |
                     |             Visit webpage tool |
                     +--------------------------------+
```
Let's set up this system. 

⚡️ Our agent will be powered by [meta-llama/Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct) using `HfApiEngine` class that uses HF's Inference API: the Inference API allows to quickly and easily run any OS model.

Run the line below to install the required dependencies:


```python
!pip install markdownify duckduckgo-search "transformers[agents]" --upgrade -q
```

We will choose to have our model powered by [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) since it's very powerful and available for free in the HF API.


```python
model = "Qwen/Qwen2.5-72B-Instruct"
```

### 🔍 Create a web search tool

For web browsing, we can already use our pre-existing [`DuckDuckGoSearchTool`](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/search.py) tool to provide a Google search equivalent.

But then we will also need to be able to peak into the page found by the `DuckDuckGoSearchTool`.
To do so, we could import the library's built-in `VisitWebpageTool`, but we will build it again to see how it's done.

So let's create our `VisitWebpageTool` tool from scratch using `markdownify`.


```python
import re
import requests
from markdownify import markdownify as md
from requests.exceptions import RequestException
from transformers.agents import tool


@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = md(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
```

Ok, now let's initialize and test our tool!


```python
print(visit_webpage("https://en.wikipedia.org/wiki/Hugging_Face")[:500])
```

    Hugging Face \- Wikipedia
    
    [Jump to content](#bodyContent)
    
    Main menu
    
    Main menu
    move to sidebar
    hide
    
     Navigation
     
    
    * [Main page](/wiki/Main_Page "Visit the main page [z]")
    * [Contents](/wiki/Wikipedia:Contents "Guides to browsing Wikipedia")
    * [Current events](/wiki/Portal:Current_events "Articles related to current events")
    * [Random article](/wiki/Special:Random "Visit a randomly selected article [x]")
    * [About Wikipedia](/wiki/Wikipedia:About "Learn about Wikipedia and how it works")
    * [Co
    

## Build our multi-agent system 🤖🤝🤖

Now that we have all the tools `search` and `visit_webpage`, we can use them to create the web agent.

Which configuration to choose for this agent?
- Web browsing is a single-timeline task that does not require parallel tool calls, so JSON tool calling works well for that. We thus choose a `ReactJsonAgent`.
- Also, since sometimes web search requires exploring many pages before finding the correct answer, we prefer to increase the number of `max_iterations` to 10.


```python
from transformers.agents import (
    ReactCodeAgent,
    ReactJsonAgent,
    HfApiEngine,
    ManagedAgent,
)
from transformers.agents.search import DuckDuckGoSearchTool

llm_engine = HfApiEngine(model)

web_agent = ReactJsonAgent(
    tools=[DuckDuckGoSearchTool(), visit_webpage],
    llm_engine=llm_engine,
    max_iterations=10,
)
```

We then wrap this agent into a `ManagedAgent` that will make it callable by its manager agent.


```python
managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="search",
    description="Runs web searches for you. Give it your query as an argument.",
)
```

Finally we create a manager agent, and upon initialization we pass our managed agent to it in its `managed_agents` argument.

Since this agent is the one tasked with the planning and thinking, advanced reasoning will be beneficial, so a `ReactCodeAgent` will be the best choice.

Also, we want to ask a question that involves the current year: so let us add `additional_authorized_imports=["time", "datetime"]`


```python
manager_agent = ReactCodeAgent(
    tools=[],
    llm_engine=llm_engine,
    managed_agents=[managed_web_agent],
    additional_authorized_imports=["time", "datetime"],
)
```

That's all! Now let's run our system! We select a question that requires some calculation and 


```python
manager_agent.run("How many years ago was Stripe founded?")
```

    [32;20;1m======== New task ========[0m
    [37;1mHow many years ago was Stripe founded?[0m
    [33;1m=== Agent thoughts:[0m
    [0mThought: I need to find out when Stripe was founded and then calculate the number of years since then. I will start by using the `search` tool to find the founding year of Stripe.[0m
    [33;1m>>> Agent is executing the code below:[0m
    [0m[38;5;7mfounding_year[39m[38;5;7m [39m[38;5;109;01m=[39;00m[38;5;7m [39m[38;5;7msearch[39m[38;5;7m([39m[38;5;144m"[39m[38;5;144mWhen was Stripe founded[39m[38;5;144m"[39m[38;5;7m)[39m
    [38;5;109mprint[39m[38;5;7m([39m[38;5;144m"[39m[38;5;144mFounding year:[39m[38;5;144m"[39m[38;5;7m,[39m[38;5;7m [39m[38;5;7mfounding_year[39m[38;5;7m)[39m[0m
    [33;1m====[0m
    [32;20;1m======== New task ========[0m
    [37;1mYou're a helpful agent named 'search'.
    You have been submitted this task by your manager.
    ---
    Task:
    When was Stripe founded
    ---
    You're helping your manager solve a wider task: so make sure to not provide a one-line answer, but give as much information as possible so that they have a clear understanding of the answer.
    
    Your final_answer WILL HAVE to contain these parts:
    ### 1. Task outcome (short version):
    ### 2. Task outcome (extremely detailed version):
    ### 3. Additional context (if relevant):
    
    Put all these in your final_answer tool, everything that you do not pass as an argument to final_answer will be lost.
    And even if your task resolution is not successful, please return as much context as possible, so that your manager can act upon this feedback.[0m
    [33;1m=== Agent thoughts:[0m
    [0mThought: I need to find the founding year of Stripe and related details. The best way to start is by performing a web search.[0m
    [33;1m>>> Calling tool: 'web_search' with arguments: {'query': 'When was Stripe founded'}[0m
    [33;1m=== Agent thoughts:[0m
    [0mThought: The search results provide information on when Stripe was founded and additional details about the company. I will now visit the Stripe Wikipedia page for a more detailed overview.[0m
    [33;1m>>> Calling tool: 'visit_webpage' with arguments: {'url': 'https://en.wikipedia.org/wiki/Stripe,_Inc.'}[0m
    [33;1m=== Agent thoughts:[0m
    [0mThought: I have collected detailed information about Stripe from the Wikipedia page. Now, I will formulate a comprehensive final answer as required for the task.[0m
    [33;1m>>> Calling tool: 'final_answer' with arguments: {'answer': "### 1. Task Outcome (short version):\nStripe, Inc. was founded in 2010 by Irish brothers John Collison and Patrick Collison, who serve as the company's president and CEO, respectively. The company is headquartered in South San Francisco, California, and Dublin, Ireland. Stripe provides payment processing and financial services for businesses, enabling them to accept payments and manage financial transactions online.\n\n### 2. Task Outcome (extremely detailed version):\nStripe, Inc. is an Irish-American multinational financial services and software as a service (SaaS) company co-founded in 2010 by John Collison and Patrick Collison, two Irish brothers. The company is dual-headquartered in South San Francisco, California, and Dublin, Ireland. Stripe offers a wide range of financial services and tools, primarily focused on payment processing and management for businesses. Some key milestones and details include:\n\n- **Founding and Early Years:** The company was founded in 2010 in Palo Alto, California. In 2011, it received a $2 million investment from notable figures such as Elon Musk, Peter Thiel, and venture capital firms like Sequoia Capital. In 2012, Stripe launched its first multiparty payments solution, Stripe Connect.\n- **Growth and Expansion:** Stripe has rapidly expanded its services and reach. In 2013, it made its first acquisition, Kickoff, a chat and task management application. In 2016, Stripe launched Atlas, a platform to help startups register as U.S. corporations. The company has continued to grow, raising significant rounds of funding and expanding its services to new markets, including Europe and Africa.\n- **Product Suite:** Stripe offers a comprehensive suite of financial tools, including payment processing, billing, fraud prevention, point-of-sale solutions, and more. Notable products include Radar (anti-fraud tools), Terminal (point-of-sale hardware), and Stripe Capital (merchant cash advances).\n- **Partnerships and Integrations:** Stripe has formed partnerships with major companies such as Ford, Spotify, and Twitter to handle transactions and payments. It has also launched the Stripe App Marketplace, allowing businesses to integrate third-party apps and services.\n- **Valuation and Funding:** As of the latest data, Stripe has raised over $6.5 billion in funding and is valued at around $70 billion, making it one of the most valuable privately-held startups globally.\n- **Challenges and Layoffs:** In 2022, Stripe announced layoffs, cutting 14% of its workforce to prepare for leaner times. However, the company continues to innovate and expand its offerings.\n\n### 3. Additional Context (if relevant):\n- **Impact on the Founders:** John and Patrick Collison have been influential in shaping the fintech industry. Their vision and leadership have driven Stripe's success and innovation.\n- **Industry Position:** Stripe is a leader in the fintech sector, competing with other payment processors and financial service providers. Its robust product suite and global reach have solidified its position in the market.\n- **Future Outlook:** Stripe continues to invest in new technologies and services, including AI and carbon capture initiatives. The company's focus on innovation and customer needs positions it well for future growth."}[0m
    [33;1mPrint outputs:[0m
    [32;20mFounding year: ### 1. Task Outcome (short version):
    Stripe, Inc. was founded in 2010 by Irish brothers John Collison and Patrick Collison, who serve as the company's president and CEO, respectively. The company is headquartered in South San Francisco, California, and Dublin, Ireland. Stripe provides payment processing and financial services for businesses, enabling them to accept payments and manage financial transactions online.
    
    ### 2. Task Outcome (extremely detailed version):
    Stripe, Inc. is an Irish-American multinational financial services and software as a service (SaaS) company co-founded in 2010 by John Collison and Patrick Collison, two Irish brothers. The company is dual-headquartered in South San Francisco, California, and Dublin, Ireland. Stripe offers a wide range of financial services and tools, primarily focused on payment processing and management for businesses. Some key milestones and details include:
    
    - **Founding and Early Years:** The company was founded in 2010 in Palo Alto, California. In 2011, it received a $2 million investment from notable figures such as Elon Musk, Peter Thiel, and venture capital firms like Sequoia Capital. In 2012, Stripe launched its first multiparty payments solution, Stripe Connect.
    - **Growth and Expansion:** Stripe has rapidly expanded its services and reach. In 2013, it made its first acquisition, Kickoff, a chat and task management application. In 2016, Stripe launched Atlas, a platform to help startups register as U.S. corporations. The company has continued to grow, raising significant rounds of funding and expanding its services to new markets, including Europe and Africa.
    - **Product Suite:** Stripe offers a comprehensive suite of financial tools, including payment processing, billing, fraud prevention, point-of-sale solutions, and more. Notable products include Radar (anti-fraud tools), Terminal (point-of-sale hardware), and Stripe Capital (merchant cash advances).
    - **Partnerships and Integrations:** Stripe has formed partnerships with major companies such as Ford, Spotify, and Twitter to handle transactions and payments. It has also launched the Stripe App Marketplace, allowing businesses to integrate third-party apps and services.
    - **Valuation and Funding:** As of the latest data, Stripe has raised over $6.5 billion in funding and is valued at around $70 billion, making it one of the most valuable privately-held startups globally.
    - **Challenges and Layoffs:** In 2022, Stripe announced layoffs, cutting 14% of its workforce to prepare for leaner times. However, the company continues to innovate and expand its offerings.
    
    ### 3. Additional Context (if relevant):
    - **Impact on the Founders:** John and Patrick Collison have been influential in shaping the fintech industry. Their vision and leadership have driven Stripe's success and innovation.
    - **Industry Position:** Stripe is a leader in the fintech sector, competing with other payment processors and financial service providers. Its robust product suite and global reach have solidified its position in the market.
    - **Future Outlook:** Stripe continues to invest in new technologies and services, including AI and carbon capture initiatives. The company's focus on innovation and customer needs positions it well for future growth.
    [0m
    [33;1m=== Agent thoughts:[0m
    [0mThought: The search result shows that Stripe was founded in 2010. Now I need to calculate how many years ago that was. I will use the current year to make this calculation.[0m
    [33;1m>>> Agent is executing the code below:[0m
    [0m[38;5;109;01mimport[39;00m[38;5;7m [39m[38;5;109mdatetime[39m
    
    [38;5;7mcurrent_year[39m[38;5;7m [39m[38;5;109;01m=[39;00m[38;5;7m [39m[38;5;7mdatetime[39m[38;5;109;01m.[39;00m[38;5;7mdatetime[39m[38;5;109;01m.[39;00m[38;5;7mnow[39m[38;5;7m([39m[38;5;7m)[39m[38;5;109;01m.[39;00m[38;5;7myear[39m
    [38;5;7mfounding_year[39m[38;5;7m [39m[38;5;109;01m=[39;00m[38;5;7m [39m[38;5;139m2010[39m
    [38;5;7myears_since_founded[39m[38;5;7m [39m[38;5;109;01m=[39;00m[38;5;7m [39m[38;5;7mcurrent_year[39m[38;5;7m [39m[38;5;109;01m-[39;00m[38;5;7m [39m[38;5;7mfounding_year[39m
    [38;5;7mfinal_answer[39m[38;5;7m([39m[38;5;7myears_since_founded[39m[38;5;7m)[39m[0m
    [33;1m====[0m
    [33;1mPrint outputs:[0m
    [32;20m[0m
    [33;1mLast output from code snippet:[0m
    [32;20m14[0m
    [32;20;1mFinal answer:[0m
    [32;20m14[0m
    




    14



Our agents managed to efficiently collaborate towards solving the task! ✅

💡 You can easily extend this to more agents: one does the code execution, one the web search, one handles file loadings...

🤔💭 One could even think of doing more complex, tree-like hierarchies, with one CEO agent handling multiple middle managers, each with several reports.

We could even add more intermediate layers of management, each with multiple daily meetings, lots of agile stuff with scrum masters, and each new component adds enough friction to ensure the tasks never get done... Ehm wait, no, let's stick with our simple structure.
