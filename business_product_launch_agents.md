# Business Product Launch Agents

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MervinPraison/PraisonAI/blob/main/cookbooks/notebooks/business_product_launch_agents.ipynb)

## Dependencies


```python
# Install dependencies without output
%pip install langchain_community > /dev/null
%pip install praisonai[crewai] > /dev/null
%pip install duckduckgo_search > /dev/null
```

## Tools


```python
from praisonai_tools import BaseTool, FileReadTool, SerperDevTool, ScrapeWebsiteTool
from langchain_community.tools.file_management.write import WriteFileTool
from duckduckgo_search import DDGS
from langchain.tools import tool

class InternetSearchTool(BaseTool):
    name: str = "InternetSearchTool"
    description: str = "Search Internet for relevant information based on a query or latest news"

    def _run(self, query: str):
        ddgs = DDGS()
        results = ddgs.text(keywords=query, region='wt-wt', safesearch='moderate', max_results=5)
        return results
```

## YAML Prompt


```python
agent_yaml = """
framework: crewai
topic: Artificial Intelligence
roles:
  movie_concept_creator:
    backstory: 'Creative thinker with a deep understanding of cinematic storytelling,
      capable of using AI-generated storylines to create unique and compelling movie
      ideas.'
    goal: Generate engaging movie concepts using AI storylines
    role: Movie Concept Creator
    tasks:
      movie_concept_development:
        description: 'Develop movie concepts from AI-generated storylines, ensuring
          they are engaging and have strong narrative arcs.'
        expected_output: 'Well-structured movie concept document with character
          bios, settings, and plot outlines.'
  screenwriter:
    backstory: 'Expert in writing engaging dialogue and script structure, able to
      turn movie concepts into production-ready scripts.'
    goal: Write compelling scripts based on movie concepts
    role: Screenwriter
    tasks:
      scriptwriting_task:
        description: 'Turn movie concepts into polished scripts with well-developed
          characters, strong dialogue, and effective scene transitions.'
        expected_output: 'Production-ready script with a beginning, middle, and
          end, along with character development and engaging dialogues.'
  editor:
    backstory: 'Adept at identifying inconsistencies, improving language usage,
      and maintaining the overall flow of the script.'
    goal: Refine the scripts and ensure continuity of the movie storyline
    role: Editor
    tasks:
      editing_task:
        description: 'Review, edit, and refine the scripts to ensure they are cohesive
          and follow a well-structured narrative.'
        expected_output: 'A polished final draft of the script with no inconsistencies,
          strong character development, and effective dialogue.'
dependencies: []
"""
```

## Main


```python
import os
from praisonai import PraisonAI
from google.colab import userdata

# Create a PraisonAI instance with the agent_yaml content
praisonai = PraisonAI(agent_yaml=agent_yaml, tools=[InternetSearchTool, ScrapeWebsiteTool]) # Add InstagramSearchTool once scripted

# Add OPENAI_API_KEY Secrets to Google Colab on the Left Hand Side 🔑 or Enter Manually Below
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY') or "ENTER OPENAI_API_KEY HERE"
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

# Run PraisonAI
result = praisonai.run()

# Print the result
print(result)

```
