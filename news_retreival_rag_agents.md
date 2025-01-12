# News Retreival RAG Agents
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MervinPraison/PraisonAI/blob/main/cookbooks/notebooks/news_retreival_rag_agents.ipynb)

## Dependencies


```python
# Install dependencies without output
%pip install langchain_community > /dev/null
%pip install praisonai[crewai] > /dev/null
%pip install duckduckgo_search > /dev/null
%pip install requests > /dev/null
```

## Tools


```python
from google.colab import userdata

import requests
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from praisonai_tools import BaseTool
from duckduckgo_search import DDGS


class InternetSearchTool(BaseTool):
    name: str = "InternetSearchTool"
    description: str = "Search Internet for relevant information based on a query or latest news"

    def _run(self, query: str):
        ddgs = DDGS()
        results = ddgs.text(keywords=query, region='wt-wt', safesearch='moderate', max_results=5)
        return results


class NewsDBTool(BaseTool):
    name: str = "NewsDBTool"
    description: str = "Fetches and stores news articles based on a specified query in a local database"

    def _run(self, query: str):
        NEWSAPI_KEY = userdata.get('NEWSAPI_KEY') or "ENTER NEWSAPI_KEY HERE"
        base_url = "https://newsapi.org/v2/everything"

        params = {
            'q': query,
            'sortBy': 'publishedAt',
            'apiKey': NEWSAPI_KEY,
            'language': 'en',
            'pageSize': 5,
        }

        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            return "Failed to retrieve news."

        articles = response.json().get('articles', [])
        if not articles:
            return "No articles found."

        all_splits = []

        for article in articles:
            loader = WebBaseLoader(article['url'])
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)

        if all_splits:
            embedding_function = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(all_splits, embedding=embedding_function, persist_directory="./chroma_db")
            vectorstore.persist()
            return "News articles processed and stored in the database."
        else:
            return "No content available for processing."



class RetrieveNewsTool(BaseTool):
    name: str = "RetrieveNewsTool"
    description: str = "Retrieves news articles from the local database using a search query"

    def _run(self, query: str):
        embedding_function = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

        retriever = vectorstore.similarity_search(query)

        if retriever:
            return [{"content": result.page_content, "metadata": result.metadata} for result in retriever]
        else:
            return "No relevant articles found in the database."

```

## YAML Prompt


```python
agent_yaml = """
framework: "crewai"
topic: "News Analysis and Reporting"
roles:
  news_search_agent:
    role: "News Researcher"
    backstory: |
      You are an experienced News Researcher, skilled in finding and summarizing key points from the latest news articles.
    goal: |
      Conduct in-depth searches on the latest news, summarize key points for each article, and make the information available for further review.
    verbose: true
    allow_delegation: true
    tools:
      - "NewsDBTool"
      - "RetrieveNewsTool"
    tasks:
      search_and_summarize_news_task:
        description: |
          Search for the latest news articles based on a specified topic, process them, and generate key points for each article.
        expected_output: |
          A list of key points from recent articles related to the specified topic, ready for further summarization.
  writer_agent:
    role: "News Content Writer"
    backstory: |
      As a News Content Writer, you specialize in transforming raw news data into detailed summaries, adding depth and clarity.
    goal: |
      Review the collected news data, verify its relevance, search for additional context, and create comprehensive summaries for each topic.
    verbose: true
    allow_delegation: true
    tools:
      - "RetrieveNewsTool"
      - "InternetSearchTool"
    tasks:
      analyze_and_write_news_task:
        description: |
          Identify each topic, verify relevance using the retrieved data, then conduct additional web searches to add depth. Summarize each topic in detail.
        expected_output: |
          An in-depth summary of each identified news topic, incorporating all relevant information retrieved from the database and online searches.
dependencies: []
"""
```

## Main


```python
import os
from praisonai import PraisonAI
from google.colab import userdata

# Create a PraisonAI instance with the agent_yaml content
praisonai = PraisonAI(agent_yaml=agent_yaml, tools=[NewsDBTool, RetrieveNewsTool, InternetSearchTool])

# Add OPENAI_API_KEY Secrets to Google Colab on the Left Hand Side 🔑 or Enter Manually Below
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY') or "ENTER OPENAI_API_KEY HERE"
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

# Run PraisonAI
result = praisonai.run()

# Print the result
print(result) # 10/10

```

    [1m[95m [2024-11-02 17:51:51][DEBUG]: == Working Agent: News Researcher[00m
    [1m[95m [2024-11-02 17:51:51][INFO]: == Starting Task: Search for the latest news articles based on a specified topic, process them, and generate key points for each article.
    [00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI need to conduct a search for the latest news articles on a specified topic to gather key points for further summarization. Since the topic is not specified yet, I will first define a general topic to search for. 
    
    For this exercise, I'll choose the topic "climate change" as it is currently a relevant and pressing issue. 
    
    Action: NewsDBTool  
    Action Input: {"query": "climate change"}  [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/langchain_core/_api/deprecation.py:119: LangChainDeprecationWarning: Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
      warn_deprecated(
    

    [95m 
    
    News articles processed and stored in the database.
    [00m
    [32;1m[1;3mThought: I have successfully stored news articles related to "climate change" in the database. Now, I will retrieve these articles to summarize the key points from each one. 
    
    Action: RetrieveNewsTool  
    Action Input: {"query": "climate change"}  [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    [{'content': 'I’m hearing so many stories like that from the survivors of the climate-intensified hurricanes that trashed western North Carolina, coastal Florida, and other parts of the Southeastern USA. From the victims of a climate-intensified catastrophe that has wrecked whole towns and torn out roads, flattened forests, washed away homes and put parts of Asheville underwater. I don’t want any more disasters like that, and I’m a climate activist to try to keep nature from getting more violent and destructive, which it will if we keep being violent and destructive toward the climate. But I do want us to know who we are, and how hungry we are for meaning, purpose, and connection, and sometimes disaster lets us see that.', 'metadata': {'language': 'en-US', 'source': 'https://lithub.com/rebecca-solnit-we-the-people-can-make-a-better-future-for-all/', 'title': 'Rebecca Solnit: We, the People, Can Make a Better Future ‹ Literary  Hub'}}, {'content': 'When it comes to the climate we want faith in the future, we want the symphony of life to continue with the harmonies, the beauties, the integration of the parts into one harmonious whole to continue. Part of what gives our lives meaning is the confidence or at least hope that these good things will persevere beyond us, that there will be bison grazing the prairies in the year 2124, that there will be whales migrating in the oceans, that wildflowers will bloom in spring and pollinators will come for the nectar and leave with the pollen, that the people we love who are one or six or seventeen or their grandchildren will have a chance to enjoy some of the things we have, that there will be joy and beauty and possibility in the year 2074 and after.', 'metadata': {'language': 'en-US', 'source': 'https://lithub.com/rebecca-solnit-we-the-people-can-make-a-better-future-for-all/', 'title': 'Rebecca Solnit: We, the People, Can Make a Better Future ‹ Literary  Hub'}}, {'content': 'Barriers to Environmental Sustainability', 'metadata': {'description': 'Investigate the role of AI in enhancing logistics efficiency and promoting environmental sustainability through advanced route optimization techniques.', 'language': 'en', 'source': 'https://dzone.com/articles/smart-routing-using-ai-for-efficient-logistics', 'title': 'Smart Routing Using AI for Efficient Logistics'}}, {'content': 'Polls offer the false promise of knowing what is going to happen, but what is going to happen in this election is what campaigners, activists, and the electorate make happen. It is not yet decided. We are deciding it with what we do, as voters, as organizers, as voices for truth, justice, inclusion, the reality of the climate crisis and the importance of acting on it. In June, I got to meet one of my heroes, Congressman Jamie Raskin when he gave a keynote for the Third Act chapters in DC, Virginia and Maryland. (Third Act is a climate group founded by Bill McKibben for US people over 60; I’m on its board.) He gave me his memoir of prosecuting the impeachment of Trump after January 6, right after his beloved son Tommy had died by suicide, and there’s a dazzling passage in it that reminds us of the power of participation.\nHe writes that, during his first campaign, there was', 'metadata': {'language': 'en-US', 'source': 'https://lithub.com/rebecca-solnit-we-the-people-can-make-a-better-future-for-all/', 'title': 'Rebecca Solnit: We, the People, Can Make a Better Future ‹ Literary  Hub'}}]
    [00m
    [32;1m[1;3mThought: I have retrieved several articles related to "climate change." Now, I will summarize the key points from each article to make the information available for further review.
    
    Final Answer:
    1. **Title: Rebecca Solnit: We, the People, Can Make a Better Future**  
       - **Source:** [Literary Hub](https://lithub.com/rebecca-solnit-we-the-people-can-make-a-better-future-for-all/)  
       - **Key Points:**
         - Survivors of climate-intensified hurricanes have shared their stories, revealing the devastation in places like western North Carolina and coastal Florida.
         - The author emphasizes the need to address climate violence to prevent future disasters and highlights the human desire for meaning and connection, especially in the face of catastrophe.
         - A hopeful vision for the future includes the persistence of life and beauty, such as bison on prairies and whales in oceans, which gives life meaning.
    
    2. **Title: Smart Routing Using AI for Efficient Logistics**  
       - **Source:** [DZone](https://dzone.com/articles/smart-routing-using-ai-for-efficient-logistics)  
       - **Key Points:**
         - The article discusses the role of AI in improving logistics efficiency.
         - It highlights advanced route optimization techniques that can promote environmental sustainability.
         - The integration of AI is presented as a crucial factor in overcoming barriers to environmental sustainability.
    
    3. **Title: Rebecca Solnit: We, the People, Can Make a Better Future** (continued)  
       - **Source:** [Literary Hub](https://lithub.com/rebecca-solnit-we-the-people-can-make-a-better-future-for-all/)  
       - **Key Points:**
         - The article stresses the importance of participation in elections and activism concerning climate change.
         - It references a keynote from Congressman Jamie Raskin, who inspires action and highlights the importance of organizing and participating in the democratic process.
         - The narrative suggests that the future of climate action is not predetermined; it is shaped by the actions of voters and activists.
    
    These key points provide an overview of the current discourse around climate change, touching on personal stories, technological advancements, and the importance of civic engagement.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-02 17:52:11][DEBUG]: == [News Researcher] Task output: 1. **Title: Rebecca Solnit: We, the People, Can Make a Better Future**  
       - **Source:** [Literary Hub](https://lithub.com/rebecca-solnit-we-the-people-can-make-a-better-future-for-all/)  
       - **Key Points:**
         - Survivors of climate-intensified hurricanes have shared their stories, revealing the devastation in places like western North Carolina and coastal Florida.
         - The author emphasizes the need to address climate violence to prevent future disasters and highlights the human desire for meaning and connection, especially in the face of catastrophe.
         - A hopeful vision for the future includes the persistence of life and beauty, such as bison on prairies and whales in oceans, which gives life meaning.
    
    2. **Title: Smart Routing Using AI for Efficient Logistics**  
       - **Source:** [DZone](https://dzone.com/articles/smart-routing-using-ai-for-efficient-logistics)  
       - **Key Points:**
         - The article discusses the role of AI in improving logistics efficiency.
         - It highlights advanced route optimization techniques that can promote environmental sustainability.
         - The integration of AI is presented as a crucial factor in overcoming barriers to environmental sustainability.
    
    3. **Title: Rebecca Solnit: We, the People, Can Make a Better Future** (continued)  
       - **Source:** [Literary Hub](https://lithub.com/rebecca-solnit-we-the-people-can-make-a-better-future-for-all/)  
       - **Key Points:**
         - The article stresses the importance of participation in elections and activism concerning climate change.
         - It references a keynote from Congressman Jamie Raskin, who inspires action and highlights the importance of organizing and participating in the democratic process.
         - The narrative suggests that the future of climate action is not predetermined; it is shaped by the actions of voters and activists.
    
    These key points provide an overview of the current discourse around climate change, touching on personal stories, technological advancements, and the importance of civic engagement.
    
    [00m
    [1m[95m [2024-11-02 17:52:11][DEBUG]: == Working Agent: News Content Writer[00m
    [1m[95m [2024-11-02 17:52:11][INFO]: == Starting Task: Identify each topic, verify relevance using the retrieved data, then conduct additional web searches to add depth. Summarize each topic in detail.
    [00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI will start by reviewing the collected news data to identify the main topics, verify their relevance, and then conduct additional web searches to gather more context. The topics identified are climate change activism and technology's role in logistics and environmental sustainability.
    
    Action: RetrieveNewsTool  
    Action Input: {"query": "climate change activism"}  [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    [{'content': 'I’m hearing so many stories like that from the survivors of the climate-intensified hurricanes that trashed western North Carolina, coastal Florida, and other parts of the Southeastern USA. From the victims of a climate-intensified catastrophe that has wrecked whole towns and torn out roads, flattened forests, washed away homes and put parts of Asheville underwater. I don’t want any more disasters like that, and I’m a climate activist to try to keep nature from getting more violent and destructive, which it will if we keep being violent and destructive toward the climate. But I do want us to know who we are, and how hungry we are for meaning, purpose, and connection, and sometimes disaster lets us see that.', 'metadata': {'language': 'en-US', 'source': 'https://lithub.com/rebecca-solnit-we-the-people-can-make-a-better-future-for-all/', 'title': 'Rebecca Solnit: We, the People, Can Make a Better Future ‹ Literary  Hub'}}, {'content': 'Polls offer the false promise of knowing what is going to happen, but what is going to happen in this election is what campaigners, activists, and the electorate make happen. It is not yet decided. We are deciding it with what we do, as voters, as organizers, as voices for truth, justice, inclusion, the reality of the climate crisis and the importance of acting on it. In June, I got to meet one of my heroes, Congressman Jamie Raskin when he gave a keynote for the Third Act chapters in DC, Virginia and Maryland. (Third Act is a climate group founded by Bill McKibben for US people over 60; I’m on its board.) He gave me his memoir of prosecuting the impeachment of Trump after January 6, right after his beloved son Tommy had died by suicide, and there’s a dazzling passage in it that reminds us of the power of participation.\nHe writes that, during his first campaign, there was', 'metadata': {'language': 'en-US', 'source': 'https://lithub.com/rebecca-solnit-we-the-people-can-make-a-better-future-for-all/', 'title': 'Rebecca Solnit: We, the People, Can Make a Better Future ‹ Literary  Hub'}}, {'content': 'Barriers to Environmental Sustainability', 'metadata': {'description': 'Investigate the role of AI in enhancing logistics efficiency and promoting environmental sustainability through advanced route optimization techniques.', 'language': 'en', 'source': 'https://dzone.com/articles/smart-routing-using-ai-for-efficient-logistics', 'title': 'Smart Routing Using AI for Efficient Logistics'}}, {'content': 'Rebecca Solnit\n\t\t\t\t\t\t\t\t\t\t\t\t\tWriter, historian, and activist Rebecca Solnit is the author of twenty-five books on feminism, environmental and urban history, popular power, social change and insurrection, wandering and walking, hope and catastrophe. She co-edited the 2023 anthology Not Too Late: Changing the Climate Story from Despair to Possibility. Her other books include Orwell’s Roses; Recollections of My Nonexistence; Hope in the Dark; Men Explain Things to Me; A Paradise Built in Hell: The Extraordinary Communities that Arise in Disaster; and A Field Guide to Getting Lost. A product of the California public education system from kindergarten to graduate school, she writes regularly for the Guardian, serves on the board of the climate group Oil Change International, and in 2022 launched the climate project Not Too Late (nottoolateclimate.com).\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n\nhttp://rebeccasolnit.net\n\n\n\n\n\n\n \n\n\n\nPrevious Article\nSuzanne Nossel is stepping down as PEN America CEO.', 'metadata': {'language': 'en-US', 'source': 'https://lithub.com/rebecca-solnit-we-the-people-can-make-a-better-future-for-all/', 'title': 'Rebecca Solnit: We, the People, Can Make a Better Future ‹ Literary  Hub'}}]
    [00m
    [32;1m[1;3mThought: I have retrieved relevant content related to climate change activism and the role of AI in logistics. The next step is to conduct additional web searches to gather more context and depth on these topics.
    
    Action: InternetSearchTool  
    Action Input: {"query": "climate change activism news 2023"}  [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    [{'title': "Thousands march in New York to demand that Biden 'end fossil fuels' - NPR", 'href': 'https://www.npr.org/2023/09/17/1200027693/thousands-expected-to-march-in-new-york-to-demand-that-biden-end-fossil-fuels', 'body': 'Thousands of activists, indigenous groups, students and others take to the streets of New York for the March to End Fossil Fuels protest on Sunday, Sept. 17, 2023 in New York. Spencer Platt/Getty ...'}, {'title': 'Judge sides with young activists in first-of-its-kind climate change ...', 'href': 'https://www.npr.org/2023/08/14/1193780700/montana-climate-change-trial-ruling', 'body': 'In a landmark climate change trial, a Montana judge sides with youth activists The ruling follows a first-of-its-kind trial in the U.S., adding to a small number of legal decisions around the ...'}, {'title': "The Most Important Climate Stories of 2023 Aren't All Bad News", 'href': 'https://www.scientificamerican.com/article/the-most-important-climate-stories-of-2023-arent-all-bad-news/', 'body': 'In 2023 climate news was a mixed bag: we saw unrelenting heat but also glimmers of hope of progress toward reducing greenhouse gas emissions. By Andrea Thompson. A person rides a bicycle as heat ...'}, {'title': 'Meet the Young Activists behind the New Youth Climate Lawsuit', 'href': 'https://www.scientificamerican.com/article/meet-the-young-activists-behind-the-new-youth-climate-lawsuit/', 'body': 'Like the other young activists who agreed to join in the lawsuit, Williams, a Los Angeles resident, said climate change is increasingly disrupting her life. She loves soccer, but smoke from ...'}, {'title': 'Climate Action 30 - Business Insider', 'href': 'https://www.businessinsider.com/climate-action-30-global-leaders-climate-solutions-2023-11?op=1', 'body': 'Climate Action 30 spotlights global activists, academics, founders, executives, and nonprofit and public sector leaders tackling the climate crisis. ... the Harvard Chan C-Change Youth Summit as ...'}]
    [00m
    [32;1m[1;3mThought: I have gathered relevant information regarding climate change activism from both the retrieved articles and the recent web search results. Now, I will also gather more context on the role of AI in logistics to comprehensively summarize both topics.
    
    Action: InternetSearchTool  
    Action Input: {"query": "AI in logistics 2023 news"}  [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    [{'title': 'How Global Companies Use AI to Prevent Supply Chain Disruptions', 'href': 'https://hbr.org/2023/11/how-global-companies-use-ai-to-prevent-supply-chain-disruptions', 'body': 'Our recent research at Walmart, Tyson Foods, Koch Industries, Maersk, Siemens, and Unilever revealed how these global companies are using advanced AI technologies to plan for and adapt to supply ...'}, {'title': 'AI in Logistics: A Sea of Potential Amidst Waves of Change', 'href': 'https://www.freightos.com/ai-in-logistics-a-sea-of-potential-amidst-waves-of-change/', 'body': 'A significant share (84%) believe AI might trim the workforce. Specifically, 53% anticipate a limited reduction, while 31% foresee substantial cuts. Enterprises, more than SMBs, sense this looming change, with 38% vs. 26% bracing for significant reductions. Notably, a handful of SMBs (7%) harbor hopes of AI birthing new roles, a sentiment ...'}, {'title': 'Logistics Trends 2023/2024: Which Direction for AI?', 'href': 'https://dhl-freight-connections.com/en/trends/logistics-trends-2023-2024/', 'body': 'Review 2023, Outlook 2024 - the Trends Carry On. The keyword for our 2023 trends was "resilience". After the supply chain problems caused by the coronavirus and the war in Ukraine, we felt that innovative technologies for the stabilization and robustness of supply chains should be a priority. And as it turned out in 2023, it was not just ...'}, {'title': 'The True Role Of AI In Logistics - Forbes', 'href': 'https://www.forbes.com/councils/forbestechcouncil/2023/08/17/the-true-role-of-ai-in-logistics/', 'body': 'True Artificial Intelligence: Implement Warehouse Automation And Robotics. The integration of AI and robotics has led to significant advancements in warehouse automation.'}, {'title': 'AI in Logistics: Top 23 Use Cases and Applications in 2023 - Markovate', 'href': 'https://markovate.com/blog/ai-applications-in-logistics/', 'body': "10. AI in Logistics for Supply Chain Transparency. In any supply chain, stuff happens. With AI, you can keep tabs on that stuff every step of the way, from the factory floor to the customer's door, making it easier to fix issues quickly. 11. AI in Logistics for Customer Service."}]
    [00m
    [32;1m[1;3mThought: I have successfully gathered extensive information on climate change activism and the role of AI in logistics. Now, I will synthesize this information into comprehensive summaries for each topic, ensuring that I cover both the retrieved data and the additional context from online searches.
    
    ### Climate Change Activism
    
    In recent discussions about climate change activism, notable voices, including Rebecca Solnit, emphasize the critical importance of civic participation and collective action in combating climate crises. Survivors of climate-intensified disasters, such as hurricanes affecting western North Carolina and coastal Florida, have shared harrowing accounts of the devastation they faced. These narratives highlight not only the physical destruction of homes and communities but also the emotional and psychological toll of such disasters. Solnit articulates a yearning for meaning and connection that often surfaces during catastrophic events, suggesting that these experiences can galvanize a deeper commitment to environmental activism.
    
    A pivotal aspect of contemporary climate activism is the role of organized movements and legal actions. For instance, a recent landmark ruling in Montana demonstrated the impact of youth-led initiatives, where a judge sided with young activists in a significant climate change trial. This case is part of a growing trend where the voices of younger generations are increasingly recognized in discussions surrounding climate policy and action.
    
    Moreover, the March to End Fossil Fuels, held in September 2023, saw thousands of activists, including indigenous groups and students, taking to the streets of New York to demand decisive action from political leaders, particularly President Biden. This demonstration underscores the urgency felt by many regarding the need for a swift transition away from fossil fuel dependency.
    
    In summary, the current climate activism landscape is marked by a blend of personal testimonies, collective demonstrations, and legal battles, all pushing for a transformative approach to environmental policy and action. The future of climate action is not predetermined; rather, it is shaped by the involvement of voters and activists who are committed to advocating for change.
    
    ### The Role of AI in Logistics and Environmental Sustainability
    
    As industries continue to evolve, the integration of Artificial Intelligence (AI) in logistics is emerging as a crucial factor for enhancing efficiency and promoting environmental sustainability. Recent advancements in AI technologies are being leveraged by global companies like Walmart, Tyson Foods, and Unilever to navigate and adapt to supply chain disruptions, particularly in light of challenges posed by the COVID-19 pandemic and geopolitical tensions.
    
    AI's application in logistics includes advanced route optimization techniques that can significantly reduce carbon footprints associated with transportation. By analyzing vast amounts of data, AI systems can identify the most efficient routes for delivery, minimize fuel consumption, and promote sustainable practices within supply chains. This not only benefits the environment but also enhances operational efficiency, allowing companies to respond more agilely to changing market demands.
    
    Furthermore, the discourse surrounding AI in logistics highlights concerns about workforce displacement, with many anticipating that automation may lead to job reductions. However, there is also optimism that AI could create new roles and opportunities, particularly in areas focused on sustainability and innovation.
    
    In light of these developments, the ongoing integration of AI in logistics represents a dual opportunity: to streamline operations while also addressing urgent environmental challenges. As businesses increasingly adopt AI-driven solutions, the potential for significant improvements in logistics efficiency and sustainability becomes more pronounced, paving the way for a greener future.
    
    Final Answer: The detailed summaries above encapsulate the current discourse surrounding climate change activism, highlighting personal stories, collective actions, and the pivotal role of civic engagement. Additionally, the summary on AI in logistics illustrates how technological advancements are being harnessed to enhance operational efficiency and promote environmental sustainability, reflecting a growing recognition of the interconnectedness of technology and ecological responsibility.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-02 17:52:30][DEBUG]: == [News Content Writer] Task output: The detailed summaries above encapsulate the current discourse surrounding climate change activism, highlighting personal stories, collective actions, and the pivotal role of civic engagement. Additionally, the summary on AI in logistics illustrates how technological advancements are being harnessed to enhance operational efficiency and promote environmental sustainability, reflecting a growing recognition of the interconnectedness of technology and ecological responsibility.
    
    [00m
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">### Task Output ###
The detailed summaries above encapsulate the current discourse surrounding climate change activism, highlighting 
personal stories, collective actions, and the pivotal role of civic engagement. Additionally, the summary on AI in 
logistics illustrates how technological advancements are being harnessed to enhance operational efficiency and 
promote environmental sustainability, reflecting a growing recognition of the interconnectedness of technology and 
ecological responsibility.
</pre>



    None
    


```python

```
