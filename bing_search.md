# Bing Search

> [Bing Search](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/) is an Azure service and enables safe, ad-free, location-aware search results, surfacing relevant information from billions of web documents. Help your users find what they're looking for from the world-wide-web by harnessing Bing's ability to comb billions of webpages, images, videos, and news with a single API call. 

## Setup
Following the [instruction](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/create-bing-search-service-resource) to create Azure Bing Search v7 service, and get the subscription key

The integration lives in the `langchain-community` package.


```python
%pip install -U langchain-community
```


```python
import getpass
import os

os.environ["BING_SUBSCRIPTION_KEY"] = getpass.getpass()
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"
```


```python
from langchain_community.utilities import BingSearchAPIWrapper
```


```python
search = BingSearchAPIWrapper(k=4)
```


```python
search.run("python")
```




    '<b>Python is a</b> versatile and powerful language that lets you work quickly and integrate systems more effectively. Learn how to get started, download the latest version, access documentation, find jobs, and join the Python community. <b>Python is a</b> popular programming language for various purposes. Find the latest version of Python for different operating systems, download release notes, and learn about the development process. Learn <b>Python,</b> a popular programming language for web applications, with examples, exercises, and references. Get certified by completing the PYTHON <b>course</b> at W3Schools. Learn the basic concepts and features of <b>Python,</b> a powerful and easy to learn programming language. The tutorial covers topics such as data structures, modules, classes, exceptions, input and output, and more. Learn why and how to use <b>Python,</b> a popular and easy-to-learn programming language. Find installation guides, tutorials, documentation, resources and FAQs for beginners and experienced programmers. Learn about <b>Python,</b> a high-level, general-purpose programming language with a focus on code readability and multiple paradigms. Find out its history, design, features, libraries, implementations, popularity, uses, and influences. Real <b>Python</b> offers tutorials, books, courses, and news for <b>Python</b> developers of all skill levels. Whether you want to learn <b>Python</b> basics, web development, data science, or machine learning, you can find useful articles and code examples here. Learn how to install, use, and extend <b>Python</b> 3.12.3, a popular programming language. Find tutorials, library references, API guides, FAQs, and more. <b>Python</b> is a powerful, fast, friendly and open-source language that runs everywhere. Learn how to get started, explore applications, join the community and access the latest news and events. Learn the basics of <b>Python</b> programming language with examples of numbers, text, variables, and operators. This tutorial covers the syntax, types, and features of <b>Python</b> for beginners.'



## Number of results
You can use the `k` parameter to set the number of results


```python
search = BingSearchAPIWrapper(k=1)
```


```python
search.run("python")
```




    '<b>Python</b> is a versatile and powerful language that lets you work quickly and integrate systems more effectively. Learn how to get started, download the latest version, access documentation, find jobs, and join the Python community.'



## Metadata Results

Run query through BingSearch and return snippet, title, and link metadata.

- Snippet: The description of the result.
- Title: The title of the result.
- Link: The link to the result.


```python
search = BingSearchAPIWrapper()
```


```python
search.results("apples", 5)
```




    [{'snippet': 'Learn about the nutrients, antioxidants, and potential health effects of<b> apples.</b> Find out how<b> apples</b> may help with weight loss, diabetes, heart disease, and cancer.',
      'title': 'Apples 101: Nutrition Facts and Health Benefits',
      'link': 'https://www.healthline.com/nutrition/foods/apples'},
     {'snippet': 'Learn how<b> apples</b> can improve your health with their fiber, antioxidants, and phytochemicals. Find out the best types of<b> apples</b> for different purposes, how to buy and store them, and what side effects to watch out for.',
      'title': 'Apples: Nutrition and Health Benefits - WebMD',
      'link': 'https://www.webmd.com/food-recipes/benefits-apples'},
     {'snippet': '<b>Apples</b> are nutritious, filling, and versatile fruits that may lower your risk of various diseases. Learn how<b> apples</b> can support your weight loss, heart health, gut health, and brain health with scientific evidence.',
      'title': '10 Impressive Health Benefits of Apples',
      'link': 'https://www.healthline.com/nutrition/10-health-benefits-of-apples'},
     {'snippet': 'An apple is a round, edible fruit produced by an apple tree (Malus spp., among them the domestic or orchard apple; Malus domestica).Apple trees are cultivated worldwide and are the most widely grown species in the genus Malus.The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found.<b>Apples</b> have been grown for thousands of years in Eurasia and were introduced ...',
      'title': 'Apple - Wikipedia',
      'link': 'https://en.wikipedia.org/wiki/Apple'},
     {'snippet': 'Learn about the most popular and diverse<b> apples</b> in the world, from ambrosia to winesap, with photos and descriptions. Find out their origins, flavors, uses, and nutritional benefits in this comprehensive guide to<b> apples.</b>',
      'title': '29 Types Of Apples From A to Z (With Photos!) - Live Eat Learn',
      'link': 'https://www.liveeatlearn.com/types-of-apples/'}]



## Tool Usage


```python
import os

from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.utilities import BingSearchAPIWrapper

api_wrapper = BingSearchAPIWrapper()
tool = BingSearchResults(api_wrapper=api_wrapper)
tool
```




    BingSearchResults(api_wrapper=BingSearchAPIWrapper(bing_subscription_key='<your subscription key>', bing_search_url='https://api.bing.microsoft.com/v7.0/search', k=10, search_kwargs={}))




```python
import json

# .invoke wraps utility.results
response = tool.invoke("What is the weather in Shanghai?")
response = json.loads(response.replace("'", '"'))
for item in response:
    print(item)
```

    {'snippet': '<b>Shanghai</b>, <b>Shanghai</b>, China <b>Weather</b> Forecast, with current conditions, wind, air quality, and what to expect for the next 3 days.', 'title': 'Shanghai, Shanghai, China Weather Forecast | AccuWeather', 'link': 'https://www.accuweather.com/en/cn/shanghai/106577/weather-forecast/106577'}
    {'snippet': 'Current <b>weather</b> <b>in Shanghai</b> and forecast for today, tomorrow, and next 14 days', 'title': 'Weather for Shanghai, Shanghai Municipality, China - timeanddate.com', 'link': 'https://www.timeanddate.com/weather/china/shanghai'}
    {'snippet': '<b>Shanghai</b> 14 Day Extended Forecast. <b>Weather</b> Today <b>Weather</b> Hourly 14 Day Forecast Yesterday/Past <b>Weather</b> Climate (Averages) Currently: 73 °F. Rain showers. Partly sunny. (<b>Weather</b> station: <b>Shanghai</b> Hongqiao Airport, China). See more current <b>weather</b>.', 'title': 'Shanghai, Shanghai Municipality, China 14 day weather forecast', 'link': 'https://www.timeanddate.com/weather/china/shanghai/ext'}
    {'snippet': '<b>Shanghai</b> - <b>Weather</b> warnings issued 14-day forecast. <b>Weather</b> warnings issued. Forecast - <b>Shanghai</b>. Day by day forecast. Last updated today at 18:00. Tonight, A clear sky and a gentle breeze. Clear Sky.', 'title': 'Shanghai - BBC Weather', 'link': 'https://www.bbc.com/weather/1796236'}
    

## Chaining

We show here how to use it as part of an [agent](/docs/tutorials/agents). We use the OpenAI Functions Agent, so we will need to setup and install the required dependencies for that. We will also use [LangSmith Hub](https://smith.langchain.com/hub) to pull the prompt from, so we will need to install that.


```python
# you need a model to use in the chain
%pip install --upgrade --quiet langchain langchain-openai langchainhub langchain-community
```


```python
import getpass
import os

from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import AzureChatOpenAI

os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass()
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://<your-endpoint>.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-06-01-preview"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "<your-deployment-name>"

instructions = """You are an assistant."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)
llm = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
tool = BingSearchResults(api_wrapper=api_wrapper)
tools = [tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)
agent_executor.invoke({"input": "What happened in the latest burning man floods?"})
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `bing_search_results_json` with `{'query': 'latest burning man floods'}`
    
    
    [0m[36;1m[1;3m[{'snippet': 'Live Updates. Thousands stranded at <b>Burning</b> <b>Man</b> festival after heavy rains. By Maureen Chowdhury, Steve Almasyand Matt Meyer, CNN. Updated 9:00 PM EDT, Sun September 3, 2023. Link Copied!', 'title': 'Thousands stranded at Burning Man festival after heavy rains', 'link': 'https://www.cnn.com/us/live-news/nevada-desert-burning-man-weather-rain-09-03-23/index.html'}, {'snippet': 'Black Rock Forest, where around 70,000 <b>Burning</b> <b>Man</b> attendees are gathered for the festival, is in the northwest. &quot;Flash <b>flooding</b> caused by excessive rainfall&quot; is possible in parts of eastern ...', 'title': 'Burning Man flooding keeps thousands stranded at Nevada site as ...', 'link': 'https://www.nbcnews.com/news/us-news/live-blog/live-updates-burning-man-flooding-keeps-thousands-stranded-nevada-site-rcna103193'}, {'snippet': 'Thousands of <b>Burning</b> <b>Man</b> attendees finally made their mass exodus after intense rain over the weekend flooded camp sites and filled them with thick, ankle-deep mud – stranding more than 70,000 ...', 'title': 'Burning Man attendees make a mass exodus after a dramatic weekend ... - CNN', 'link': 'https://www.cnn.com/2023/09/05/us/burning-man-storms-shelter-exodus-tuesday/index.html'}, {'snippet': 'FILE - In this satellite photo provided by Maxar Technologies, an overview of <b>Burning</b> <b>Man</b> festival in Black Rock, Nev on Monday, Aug. 28, 2023. Authorities in Nevada were investigating a death at the site of the <b>Burning</b> <b>Man</b> festival where thousands of attendees remained stranded as <b>flooding</b> from storms swept through the Nevada desert.', 'title': 'Wait times to exit Burning Man drop after flooding left tens of ...', 'link': 'https://apnews.com/article/burning-man-flooding-nevada-stranded-0726190c9f8378935e2a3cce7f154785'}][0m[32;1m[1;3mIn the latest Burning Man festival, heavy rains caused flooding and resulted in thousands of attendees being stranded. The festival took place in Black Rock Forest, Nevada, and around 70,000 people were gathered for the event. The excessive rainfall led to flash flooding in some parts of the area. As a result, camp sites were filled with ankle-deep mud, making it difficult for people to leave. Authorities were investigating a death at the festival site, which was affected by the flooding. However, in the following days, thousands of Burning Man attendees were able to make a mass exodus after the rain subsided.[0m
    
    [1m> Finished chain.[0m
    




    {'input': 'What happened in the latest burning man floods?',
     'output': 'In the latest Burning Man festival, heavy rains caused flooding and resulted in thousands of attendees being stranded. The festival took place in Black Rock Forest, Nevada, and around 70,000 people were gathered for the event. The excessive rainfall led to flash flooding in some parts of the area. As a result, camp sites were filled with ankle-deep mud, making it difficult for people to leave. Authorities were investigating a death at the festival site, which was affected by the flooding. However, in the following days, thousands of Burning Man attendees were able to make a mass exodus after the rain subsided.'}


