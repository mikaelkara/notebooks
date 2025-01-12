# Model Fee Retreival Agents

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MervinPraison/PraisonAI/blob/main/cookbooks/notebooks/model_fee_retreival_agents.ipynb)

## Dependencies


```python
# Install dependencies without output
%pip install langchain_community > /dev/null
%pip install praisonai[crewai] > /dev/null
%pip install crawl4ai > /dev/null
```

## Tools


```python
from crawl4ai import WebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field
from praisonai_tools import BaseTool

class ModelFee(BaseModel):
    llm_model_name: str = Field(..., description="Name of the model.")
    input_fee: str = Field(..., description="Fee for input token for the model.")
    output_fee: str = Field(..., description="Fee for output token for the model.")

class ModelFeeTool(BaseTool):
    name: str = "ModelFeeTool"
    description: str = "Extracts model fees for input and output tokens from the given pricing page."

    def _run(self, url: str):
        crawler = WebCrawler()
        crawler.warmup()

        result = crawler.run(
            url=url,
            word_count_threshold=1,
            extraction_strategy= LLMExtractionStrategy(
                provider="openai/gpt-4o",
                api_token=os.getenv('OPENAI_API_KEY'),
                schema=ModelFee.schema(),
                extraction_type="schema",
                instruction="""From the crawled content, extract all mentioned model names along with their fees for input and output tokens.
                Do not miss any models in the entire content. One extracted model JSON format should look like this:
                {"model_name": "GPT-4", "input_fee": "US$10.00 / 1M tokens", "output_fee": "US$30.00 / 1M tokens"}."""
            ),
            bypass_cache=True,
        )
        return result.extracted_content
```

## YAML Prompt


```python
agent_yaml = """
framework: "crewai"
topic: "extract model pricing from websites"
roles:
  web_scraper:
    role: "Web Scraper"
    backstory: |
      An expert in web scraping with a deep understanding of extracting structured
      data from online sources.

      https://openai.com/api/pricing/
      https://www.anthropic.com/pricing
      https://cohere.com/pricing
    goal: "Gather model pricing data from various websites"
    tasks:
      scrape_model_pricing:
        description: "Scrape model pricing information from the provided list of websites."
        expected_output: "Raw HTML or JSON containing model pricing data."
    tools:
      - "ModelFeeTool"
  data_cleaner:
    role: "Data Cleaner"
    backstory: "Specialist in data cleaning, ensuring that all collected data is accurate and properly formatted."
    goal: "Clean and organize the scraped pricing data"
    tasks:
      clean_pricing_data:
        description: "Process the raw scraped data to remove any duplicates and inconsistencies, and convert it into a structured format."
        expected_output: "Cleaned and organized JSON or CSV file with model pricing data."
    tools: []
  data_analyzer:
    role: "Data Analyzer"
    backstory: "Data analysis expert focused on deriving actionable insights from structured data."
    goal: "Analyze the cleaned pricing data to extract insights"
    tasks:
      analyze_pricing_data:
        description: "Analyze the cleaned data to extract trends, patterns, and insights on model pricing."
        expected_output: "Detailed report summarizing model pricing trends and insights."
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
praisonai = PraisonAI(agent_yaml=agent_yaml, tools=[ModelFeeTool])

# Add OPENAI_API_KEY Secrets to Google Colab on the Left Hand Side 🔑 or Enter Manually Below
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY') or "ENTER OPENAI_API_KEY HERE"
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

# Run PraisonAI
result = praisonai.run()

# Print the result
print(result) # 2/10

```

    [1m[95m [2024-11-02 12:52:01][DEBUG]: == Working Agent: Web Scraper[00m
    [1m[95m [2024-11-02 12:52:01][INFO]: == Starting Task: Scrape model pricing information from the provided list of websites.[00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mThought: I need to extract model pricing data from the provided websites using the ModelFeeTool for each URL. I will start with the first URL.
    
    Action: ModelFeeTool  
    Action Input: {"url": "https://openai.com/api/pricing/"}  
    [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [LOG] 🚀 Initializing LocalSeleniumCrawlerStrategy
    [LOG] 🌤️  Warming up the WebCrawler
    [LOG] 🌞 WebCrawler is ready to crawl
    [LOG] 🚀 Crawling done for https://openai.com/api/pricing/, success: True, time taken: 0.86 seconds
    [LOG] 🚀 Content extracted for https://openai.com/api/pricing/, success: True, time taken: 0.01 seconds
    [LOG] 🔥 Extracting semantic blocks for https://openai.com/api/pricing/, Strategy: LLMExtractionStrategy
    [LOG] Call LLM for https://openai.com/api/pricing/ - block index: 0
    

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:390: UserWarning: Pydantic serializer warnings:
      Expected `PromptTokensDetails` but got `dict` with value `{'audio_tokens': None, 'cached_tokens': 0}` - serialized value may not be as expected
      return self.__pydantic_serializer__.to_python(
    

    [LOG] Extracted 0 blocks from URL: https://openai.com/api/pricing/ block index: 0
    [LOG] 🚀 Extraction done for https://openai.com/api/pricing/, time taken: 8.94 seconds.
    [95m 
    
    []
    [00m
    [32;1m[1;3mThought: It seems that the first URL did not yield any model pricing data. I will proceed to the second URL to see if I can extract relevant information from there.
    
    Action: ModelFeeTool  
    Action Input: {"url": "https://www.anthropic.com/pricing"}  
    [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [LOG] 🚀 Initializing LocalSeleniumCrawlerStrategy
    [LOG] 🌤️  Warming up the WebCrawler
    [LOG] 🌞 WebCrawler is ready to crawl
    [LOG] 🚀 Crawling done for https://www.anthropic.com/pricing, success: True, time taken: 2.93 seconds
    [LOG] 🚀 Content extracted for https://www.anthropic.com/pricing, success: True, time taken: 0.06 seconds
    [LOG] 🔥 Extracting semantic blocks for https://www.anthropic.com/pricing, Strategy: LLMExtractionStrategy
    [LOG] Call LLM for https://www.anthropic.com/pricing - block index: 0
    [LOG] Extracted 5 blocks from URL: https://www.anthropic.com/pricing block index: 0
    [LOG] 🚀 Extraction done for https://www.anthropic.com/pricing, time taken: 4.38 seconds.
    

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:390: UserWarning: Pydantic serializer warnings:
      Expected `PromptTokensDetails` but got `dict` with value `{'audio_tokens': None, 'cached_tokens': 0}` - serialized value may not be as expected
      return self.__pydantic_serializer__.to_python(
    

    [95m 
    
    [
        {
            "llm_model_name": "Claude 3.5 Sonnet",
            "input_fee": "$3 / MTok",
            "output_fee": "$15 / MTok",
            "error": false
        },
        {
            "llm_model_name": "Claude 3.5 Haiku",
            "input_fee": "$0.25 / MTok",
            "output_fee": "$1.25 / MTok",
            "error": false
        },
        {
            "llm_model_name": "Claude 3 Opus",
            "input_fee": "$15 / MTok",
            "output_fee": "$75 / MTok",
            "error": false
        },
        {
            "llm_model_name": "Claude 3 Haiku",
            "input_fee": "$0.25 / MTok",
            "output_fee": "$1.25 / MTok",
            "error": false
        },
        {
            "llm_model_name": "Claude 3 Sonnet",
            "input_fee": "$3 / MTok",
            "output_fee": "$15 / MTok",
            "error": false
        }
    ]
    [00m
    [32;1m[1;3mAction: ModelFeeTool  
    Action Input: {"url": "https://cohere.com/pricing"}  
    [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [LOG] 🚀 Initializing LocalSeleniumCrawlerStrategy
    [LOG] 🌤️  Warming up the WebCrawler
    [LOG] 🌞 WebCrawler is ready to crawl
    [LOG] 🚀 Crawling done for https://cohere.com/pricing, success: True, time taken: 6.34 seconds
    [LOG] 🚀 Content extracted for https://cohere.com/pricing, success: True, time taken: 0.14 seconds
    [LOG] 🔥 Extracting semantic blocks for https://cohere.com/pricing, Strategy: LLMExtractionStrategy
    [LOG] Call LLM for https://cohere.com/pricing - block index: 0
    

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:390: UserWarning: Pydantic serializer warnings:
      Expected `PromptTokensDetails` but got `dict` with value `{'audio_tokens': None, 'cached_tokens': 0}` - serialized value may not be as expected
      return self.__pydantic_serializer__.to_python(
    

    [LOG] Extracted 6 blocks from URL: https://cohere.com/pricing block index: 0
    [LOG] 🚀 Extraction done for https://cohere.com/pricing, time taken: 3.71 seconds.
    [95m 
    
    [
        {
            "llm_model_name": "Command R+",
            "input_fee": "$2.50 / 1M tokens",
            "output_fee": "$10.00 / 1M tokens",
            "error": false
        },
        {
            "llm_model_name": "Command R",
            "input_fee": "$0.15 / 1M tokens",
            "output_fee": "$0.60 / 1M tokens",
            "error": false
        },
        {
            "llm_model_name": "Command R Fine-tuned Model",
            "input_fee": "$0.30 / 1M tokens",
            "output_fee": "$1.20 / 1M tokens",
            "error": false
        },
        {
            "llm_model_name": "Command R 03-2024",
            "input_fee": "$0.50 / 1M tokens",
            "output_fee": "$1.50 / 1M tokens",
            "error": false
        },
        {
            "llm_model_name": "Command R+ 04-2024",
            "input_fee": "$3.00 / 1M tokens",
            "output_fee": "$15.00 / 1M tokens",
            "error": false
        },
        {
            "llm_model_name": "Aya Expanse",
            "input_fee": "$0.50 / 1M tokens",
            "output_fee": "$1.50 / 1M tokens",
            "error": false
        }
    ]
    [00m
    [32;1m[1;3mThought: I have now scraped model pricing data from the second and third URLs. I need to compile all the data I've gathered so far into a structured format.
    
    Final Answer: {
        "OpenAI": [],
        "Anthropic": [
            {
                "llm_model_name": "Claude 3.5 Sonnet",
                "input_fee": "$3 / MTok",
                "output_fee": "$15 / MTok"
            },
            {
                "llm_model_name": "Claude 3.5 Haiku",
                "input_fee": "$0.25 / MTok",
                "output_fee": "$1.25 / MTok"
            },
            {
                "llm_model_name": "Claude 3 Opus",
                "input_fee": "$15 / MTok",
                "output_fee": "$75 / MTok"
            },
            {
                "llm_model_name": "Claude 3 Haiku",
                "input_fee": "$0.25 / MTok",
                "output_fee": "$1.25 / MTok"
            },
            {
                "llm_model_name": "Claude 3 Sonnet",
                "input_fee": "$3 / MTok",
                "output_fee": "$15 / MTok"
            }
        ],
        "Cohere": [
            {
                "llm_model_name": "Command R+",
                "input_fee": "$2.50 / 1M tokens",
                "output_fee": "$10.00 / 1M tokens"
            },
            {
                "llm_model_name": "Command R",
                "input_fee": "$0.15 / 1M tokens",
                "output_fee": "$0.60 / 1M tokens"
            },
            {
                "llm_model_name": "Command R Fine-tuned Model",
                "input_fee": "$0.30 / 1M tokens",
                "output_fee": "$1.20 / 1M tokens"
            },
            {
                "llm_model_name": "Command R 03-2024",
                "input_fee": "$0.50 / 1M tokens",
                "output_fee": "$1.50 / 1M tokens"
            },
            {
                "llm_model_name": "Command R+ 04-2024",
                "input_fee": "$3.00 / 1M tokens",
                "output_fee": "$15.00 / 1M tokens"
            },
            {
                "llm_model_name": "Aya Expanse",
                "input_fee": "$0.50 / 1M tokens",
                "output_fee": "$1.50 / 1M tokens"
            }
        ]
    }[0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-02 12:52:57][DEBUG]: == [Web Scraper] Task output: {
        "OpenAI": [],
        "Anthropic": [
            {
                "llm_model_name": "Claude 3.5 Sonnet",
                "input_fee": "$3 / MTok",
                "output_fee": "$15 / MTok"
            },
            {
                "llm_model_name": "Claude 3.5 Haiku",
                "input_fee": "$0.25 / MTok",
                "output_fee": "$1.25 / MTok"
            },
            {
                "llm_model_name": "Claude 3 Opus",
                "input_fee": "$15 / MTok",
                "output_fee": "$75 / MTok"
            },
            {
                "llm_model_name": "Claude 3 Haiku",
                "input_fee": "$0.25 / MTok",
                "output_fee": "$1.25 / MTok"
            },
            {
                "llm_model_name": "Claude 3 Sonnet",
                "input_fee": "$3 / MTok",
                "output_fee": "$15 / MTok"
            }
        ],
        "Cohere": [
            {
                "llm_model_name": "Command R+",
                "input_fee": "$2.50 / 1M tokens",
                "output_fee": "$10.00 / 1M tokens"
            },
            {
                "llm_model_name": "Command R",
                "input_fee": "$0.15 / 1M tokens",
                "output_fee": "$0.60 / 1M tokens"
            },
            {
                "llm_model_name": "Command R Fine-tuned Model",
                "input_fee": "$0.30 / 1M tokens",
                "output_fee": "$1.20 / 1M tokens"
            },
            {
                "llm_model_name": "Command R 03-2024",
                "input_fee": "$0.50 / 1M tokens",
                "output_fee": "$1.50 / 1M tokens"
            },
            {
                "llm_model_name": "Command R+ 04-2024",
                "input_fee": "$3.00 / 1M tokens",
                "output_fee": "$15.00 / 1M tokens"
            },
            {
                "llm_model_name": "Aya Expanse",
                "input_fee": "$0.50 / 1M tokens",
                "output_fee": "$1.50 / 1M tokens"
            }
        ]
    }
    
    [00m
    [1m[95m [2024-11-02 12:52:57][DEBUG]: == Working Agent: Data Cleaner[00m
    [1m[95m [2024-11-02 12:52:57][INFO]: == Starting Task: Process the raw scraped data to remove any duplicates and inconsistencies, and convert it into a structured format.[00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI now can give a great answer  
    Final Answer: 
    ```json
    {
        "models": [
            {
                "provider": "Anthropic",
                "llm_model_name": "Claude 3.5 Sonnet",
                "input_fee": "$3 / MTok",
                "output_fee": "$15 / MTok"
            },
            {
                "provider": "Anthropic",
                "llm_model_name": "Claude 3.5 Haiku",
                "input_fee": "$0.25 / MTok",
                "output_fee": "$1.25 / MTok"
            },
            {
                "provider": "Anthropic",
                "llm_model_name": "Claude 3 Opus",
                "input_fee": "$15 / MTok",
                "output_fee": "$75 / MTok"
            },
            {
                "provider": "Cohere",
                "llm_model_name": "Command R+",
                "input_fee": "$2.50 / 1M tokens",
                "output_fee": "$10.00 / 1M tokens"
            },
            {
                "provider": "Cohere",
                "llm_model_name": "Command R",
                "input_fee": "$0.15 / 1M tokens",
                "output_fee": "$0.60 / 1M tokens"
            },
            {
                "provider": "Cohere",
                "llm_model_name": "Command R Fine-tuned Model",
                "input_fee": "$0.30 / 1M tokens",
                "output_fee": "$1.20 / 1M tokens"
            },
            {
                "provider": "Cohere",
                "llm_model_name": "Command R 03-2024",
                "input_fee": "$0.50 / 1M tokens",
                "output_fee": "$1.50 / 1M tokens"
            },
            {
                "provider": "Cohere",
                "llm_model_name": "Command R+ 04-2024",
                "input_fee": "$3.00 / 1M tokens",
                "output_fee": "$15.00 / 1M tokens"
            },
            {
                "provider": "Cohere",
                "llm_model_name": "Aya Expanse",
                "input_fee": "$0.50 / 1M tokens",
                "output_fee": "$1.50 / 1M tokens"
            }
        ]
    }
    ```[0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-02 12:53:29][DEBUG]: == [Data Cleaner] Task output: ```json
    {
        "models": [
            {
                "provider": "Anthropic",
                "llm_model_name": "Claude 3.5 Sonnet",
                "input_fee": "$3 / MTok",
                "output_fee": "$15 / MTok"
            },
            {
                "provider": "Anthropic",
                "llm_model_name": "Claude 3.5 Haiku",
                "input_fee": "$0.25 / MTok",
                "output_fee": "$1.25 / MTok"
            },
            {
                "provider": "Anthropic",
                "llm_model_name": "Claude 3 Opus",
                "input_fee": "$15 / MTok",
                "output_fee": "$75 / MTok"
            },
            {
                "provider": "Cohere",
                "llm_model_name": "Command R+",
                "input_fee": "$2.50 / 1M tokens",
                "output_fee": "$10.00 / 1M tokens"
            },
            {
                "provider": "Cohere",
                "llm_model_name": "Command R",
                "input_fee": "$0.15 / 1M tokens",
                "output_fee": "$0.60 / 1M tokens"
            },
            {
                "provider": "Cohere",
                "llm_model_name": "Command R Fine-tuned Model",
                "input_fee": "$0.30 / 1M tokens",
                "output_fee": "$1.20 / 1M tokens"
            },
            {
                "provider": "Cohere",
                "llm_model_name": "Command R 03-2024",
                "input_fee": "$0.50 / 1M tokens",
                "output_fee": "$1.50 / 1M tokens"
            },
            {
                "provider": "Cohere",
                "llm_model_name": "Command R+ 04-2024",
                "input_fee": "$3.00 / 1M tokens",
                "output_fee": "$15.00 / 1M tokens"
            },
            {
                "provider": "Cohere",
                "llm_model_name": "Aya Expanse",
                "input_fee": "$0.50 / 1M tokens",
                "output_fee": "$1.50 / 1M tokens"
            }
        ]
    }
    ```
    
    [00m
    [1m[95m [2024-11-02 12:53:29][DEBUG]: == Working Agent: Data Analyzer[00m
    [1m[95m [2024-11-02 12:53:29][INFO]: == Starting Task: Analyze the cleaned data to extract trends, patterns, and insights on model pricing.[00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    


```python

```
