# Comprehensive Research Report Agents
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MervinPraison/PraisonAI/blob/main/cookbooks/notebooks/comprehensive_research_report_agents.ipynb)

## Dependencies


```python
# Install dependencies without output
%pip install langchain_community > /dev/null
%pip install praisonai[crewai] > /dev/null
%pip install requests > /dev/null
```

## Tools


```python
# ToDo: Fix Python related issue with List Index out of range
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from praisonai_tools import BaseTool

class InternetSearchTool(BaseTool):
    name: str = "InternetSearchTool"
    description: str = "Search Internet for relevant information based on a query or latest news"

    def _run(self, query: str):
        ddgs = DDGS()
        results = ddgs.text(keywords=query, region='wt-wt', safesearch='moderate', max_results=5)
        return results


class WebContentReaderTool(BaseTool):
    name: str = "WebContentReaderTool"
    description: str = "Fetches and reads the main text content from a specified webpage URL."

    def _run(self, url: str) -> str:
        """Reads the content of a webpage and returns up to 5000 characters of text."""
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract and clean the text content
            text_content = soup.get_text(separator="\n", strip=True)
            return text_content[:5000]  # Limit content to 5000 characters for brevity
        except requests.exceptions.RequestException as e:
            return f"Failed to retrieve content from {url}: {e}"

```

## YAML Prompt


```python
agent_yaml = """
framework: "crewai"
topic: "AI in Healthcare Research and Storytelling"
roles:
  manager:
    role: "Project Manager"
    backstory: |
      With a strategic mindset and a knack for leadership, you excel at guiding teams towards their goals, ensuring projects not only meet but exceed expectations.
    goal: |
      Coordinate the project to ensure a seamless integration of research findings into compelling narratives.
    verbose: true
    allow_delegation: true
    tools: []

  researcher:
    role: "Senior Researcher"
    backstory: |
      Driven by curiosity, you're at the forefront of innovation, eager to explore and share knowledge that could change the world.
    goal: |
      Uncover groundbreaking technologies and historical insights around AI in healthcare.
    verbose: true
    allow_delegation: true
    tools:
      - "InternetSearchTool"
      - "WebContentReaderTool"

  writer:
    role: "Writer"
    backstory: |
      With a flair for simplifying complex topics, you craft engaging narratives that captivate and educate, bringing new discoveries to light in an accessible manner.
    goal: |
      Narrate compelling tech stories around AI in healthcare.
    verbose: true
    allow_delegation: true
    tools:
      - "InternetSearchTool"
      - "WebContentReaderTool"

tasks:
  list_ideas:
    description: |
      List 5 interesting ideas to explore for an article about AI in healthcare.
    expected_output: |
      Bullet point list of 5 ideas for an article.
    agent: researcher
    tools:
      - "InternetSearchTool"
      - "WebContentReaderTool"
    async_execution: true

  list_important_history:
    description: |
      Research the history of AI in healthcare and identify the 5 most important events.
    expected_output: |
      Bullet point list of 5 important events.
    agent: researcher
    tools:
      - "InternetSearchTool"
      - "WebContentReaderTool"
    async_execution: true
  write_article:
    description: |
      Compose an insightful article on AI in healthcare, including its history and the latest interesting ideas.
    expected_output: |
      A 4-paragraph article about AI in healthcare.
    agent: writer
    context:
      - "list_ideas"
      - "list_important_history"
    tools:
      - "InternetSearchTool"
      - "WebContentReaderTool"
    callback: callback_function
  manager_task:
    description: |
      Oversee the integration of research findings and narrative development to produce a final comprehensive report on AI in healthcare. Ensure the research is accurately represented, and the narrative is engaging and informative.
    expected_output: |
      A final comprehensive report that combines the research findings and narrative on AI in healthcare.
    agent: manager
    tools: []
dependencies: []
"""
```

## Main


```python
import os
from praisonai import PraisonAI
from google.colab import userdata

# Create a PraisonAI instance with the agent_yaml content
praisonai = PraisonAI(agent_yaml=agent_yaml, tools=[InternetSearchTool, WebContentReaderTool])

# Add OPENAI_API_KEY Secrets to Google Colab on the Left Hand Side 🔑 or Enter Manually Below
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY') or "ENTER OPENAI_API_KEY HERE"
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

# Run PraisonAI
result = praisonai.run()

# Print the result
print(result) # 1/10

```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-4-438e945ca301> in <cell line: 13>()
         11 
         12 # Run PraisonAI
    ---> 13 result = praisonai.run()
         14 
         15 # Print the result
    

    /usr/local/lib/python3.10/dist-packages/praisonai/cli.py in run(self)
        110         Run the PraisonAI application.
        111         """
    --> 112         self.main()
        113 
        114     def main(self):
    

    /usr/local/lib/python3.10/dist-packages/praisonai/cli.py in main(self)
        256                 tools=self.tools
        257             )
    --> 258             result = agents_generator.generate_crew_and_kickoff()
        259             print(result)
        260             return result
    

    /usr/local/lib/python3.10/dist-packages/praisonai/agents_generator.py in generate_crew_and_kickoff(self)
        274             if AGENTOPS_AVAILABLE:
        275                 agentops.init(os.environ.get("AGENTOPS_API_KEY"), tags=["crewai"])
    --> 276             return self._run_crewai(config, topic, tools_dict)
        277 
        278     def _run_autogen(self, config, topic, tools_dict):
    

    /usr/local/lib/python3.10/dist-packages/praisonai/agents_generator.py in _run_crewai(self, config, topic, tools_dict)
        465         self.logger.debug(f"Tasks: {crew.tasks}")
        466 
    --> 467         response = crew.kickoff()
        468         result = f"### Task Output ###\n{response}"
        469 
    

    /usr/local/lib/python3.10/dist-packages/crewai/crew.py in kickoff(self, inputs)
        312 
        313         if self.process == Process.sequential:
    --> 314             result = self._run_sequential_process()
        315         elif self.process == Process.hierarchical:
        316             # type: ignore # Unpacking a string is disallowed
    

    /usr/local/lib/python3.10/dist-packages/crewai/crew.py in _run_sequential_process(self)
        408                 self._file_handler.log(agent=role, task=task_output, status="completed")
        409 
    --> 410         token_usage_formatted = self.aggregate_token_usage(token_usage)
        411         self._finish_execution(task_output)
        412 
    

    /usr/local/lib/python3.10/dist-packages/crewai/crew.py in aggregate_token_usage(self, token_usage_list)
        545         return {
        546             key: sum([m[key] for m in token_usage_list if m is not None])
    --> 547             for key in token_usage_list[0]
        548         }
    

    IndexError: list index out of range



```python

```
