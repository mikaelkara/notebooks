# Instagram Post Agents
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MervinPraison/PraisonAI/blob/main/cookbooks/notebooks/instagram_post_agents.ipynb)

## Dependencies


```python
# Install dependencies without output
%pip install praisonai > /dev/null
%pip install praisonai[crewai] > /dev/null
%pip install duckduckgo_search > /dev/null
%pip install python-dotenv > /dev/null
```

## Tools


```python
# This program requires addition of a pending "InstagramSearchTool" which will be added soon!
# https://github.com/MervinPraison/PraisonAI-Tools/tree/develop/praisonai_tools/tools
from duckduckgo_search import DDGS
from langchain.tools import tool
from praisonai_tools import BaseTool, ScrapeWebsiteTool

class InternetSearchTool(BaseTool):
    name: str = "InternetSearchTool"
    description: str = "Search Internet for relevant information based on a query or latest news"

    def _run(self, query: str):
        ddgs = DDGS()
        results = ddgs.text(keywords=query, region='wt-wt', safesearch='moderate', max_results=5)
        return results

os.environ["SERPER_API_KEY"] = "ENTER SERPER_API_KEY HERE"
os.environ["BROWSERLESS_API_KEY"] = "ENTER BROWSERLESS_API_KEY HERE"

# import json
# import os

# import requests
# from crewai import Agent, Task
# from langchain.tools import tool
# from unstructured.partition.html import partition_html

# from langchain.llms import Ollama

# class BrowserTools():

#   @tool("Scrape website content")
#   def scrape_and_summarize_website(website):
#     """Useful to scrape and summarize a website content, just pass a string with
#     only the full url, no need for a final slash `/`, eg: https://google.com or https://clearbit.com/about-us"""
#     url = f"https://chrome.browserless.io/content?token={os.environ['BROWSERLESS_API_KEY']}"
#     payload = json.dumps({"url": website})
#     headers = {'cache-control': 'no-cache', 'content-type': 'application/json'}
#     response = requests.request("POST", url, headers=headers, data=payload)
#     elements = partition_html(text=response.text)
#     content = "\n\n".join([str(el) for el in elements])
#     content = [content[i:i + 8000] for i in range(0, len(content), 8000)]
#     summaries = []
#     for chunk in content:
#       agent = Agent(
#           role='Principal Researcher',
#           goal=
#           'Do amazing researches and summaries based on the content you are working with',
#           backstory=
#           "You're a Principal Researcher at a big company and you need to do a research about a given topic.",
#           llm=Ollama(model=os.environ['MODEL']),
#           allow_delegation=False)
#       task = Task(
#           agent=agent,
#           description=
#           f'Analyze and make a LONG summary the content bellow, make sure to include the ALL relevant information in the summary, return only the summary nothing else.\n\nCONTENT\n----------\n{chunk}'
#       )
#       summary = task.execute()
#       summaries.append(summary)
#       content = "\n\n".join(summaries)
#     return f'\nScrapped Content: {content}\n'


# import json
# import os

# import requests
# from langchain.tools import tool


# class SearchTools():

#   @tool("Search internet")
#   def search_internet(query):
#     """Useful to search the internet about a given topic and return relevant
#     results."""
#     return SearchTools.search(query)

#   @tool("Search instagram")
#   def search_instagram(query):
#     """Useful to search for instagram post about a given topic and return relevant
#     results."""
#     query = f"site:instagram.com {query}"
#     return SearchTools.search(query)

#   def search(query, n_results=5):
#     url = "https://google.serper.dev/search"
#     payload = json.dumps({"q": query})
#     headers = {
#         'X-API-KEY': os.environ['SERPER_API_KEY'],
#         'content-type': 'application/json'
#     }
#     response = requests.request("POST", url, headers=headers, data=payload)
#     results = response.json()['organic']
#     stirng = []
#     for result in results[:n_results]:
#       try:
#         stirng.append('\n'.join([
#             f"Title: {result['title']}", f"Link: {result['link']}",
#             f"Snippet: {result['snippet']}", "\n-----------------"
#         ]))
#       except KeyError:
#         next

#     content = '\n'.join(stirng)
#     return f"\nSearch result: {content}\n"


```

## YAML Prompt


```python
agent_yaml = """
framework: "crewai"
topic: "Marketing Analysis and Strategy Development"
roles:
  lead_market_analyst:
    role: "Lead Market Analyst"
    backstory: |
      As the Lead Market Analyst at a premier digital marketing firm, you specialize in dissecting online business landscapes.
    goal: |
      Conduct an in-depth analysis of products and competitors to guide marketing strategies with actionable insights.
    verbose: true
    allow_delegation: false
    tools:
      - "ScrapeWebsiteTool"
      - "InternetSearchTool"
    tasks:
      product_analysis:
        description: |
          Analyze product and competitor landscape for key trends, strengths, and areas of opportunity in the market.
        expected_output: "Detailed report with insights on competitors, market trends, and potential strategy alignments."
  chief_marketing_strategist:
    role: "Chief Marketing Strategist"
    backstory: |
      You are the Chief Marketing Strategist at a leading digital marketing agency, known for crafting bespoke strategies that drive success.
    goal: |
      Formulate comprehensive marketing strategies based on product analysis and market insights.
    verbose: true
    allow_delegation: false
    tools:
      - "ScrapeWebsiteTool"
      - "InternetSearchTool"
      - "InstagramSearchTool"
    tasks:
      strategy_formulation:
        description: |
          Synthesize insights from product analysis to develop impactful marketing strategies.
        expected_output: "Strategic recommendations for marketing based on product insights and competitor landscape."
  creative_content_creator:
    role: "Creative Content Creator"
    backstory: |
      As a Creative Content Creator at a top-tier digital marketing agency, you excel in crafting narratives that resonate with audiences on social media. Your expertise lies in turning strategies into engaging stories and visual content.
    goal: |
      Create compelling social media content for campaigns, focusing on high-impact Instagram ad copies.
    verbose: true
    allow_delegation: true
    tools:
      - "ScrapeWebsiteTool"
      - "InternetSearchTool"
      - "InstagramSearchTool"
    tasks:
      content_creation:
        description: |
          Develop social media content that aligns with the marketing strategy and captures the target audience’s attention.
        expected_output: "Engaging social media content, optimized for platforms like Instagram, to enhance campaign reach."
  senior_photographer:
    role: "Senior Photographer"
    backstory: |
      As a Senior Photographer at a leading digital marketing agency, you are an expert in capturing photographs that inspire and engage. You are currently working on an important campaign to take standout visuals.
    goal: |
      Capture high-quality photographs for Instagram ads that convey emotion and resonate with the audience.
    verbose: true
    allow_delegation: false
    tools:
      - "InstagramSearchTool"
    tasks:
      photography:
        description: |
          Produce visually compelling photographs that align with the campaign’s objectives and enhance the ad’s impact.
        expected_output: "High-quality photographs suitable for Instagram ads that capture audience attention and engagement."
  chief_creative_director:
    role: "Chief Creative Director"
    backstory: |
      You're the Chief Content Officer of a leading digital marketing agency specializing in product branding. You ensure that your team delivers top-tier content aligned with client goals.
    goal: |
      Oversee and refine content produced by the team, ensuring alignment with the brand’s objectives and quality standards.
    verbose: true
    allow_delegation: true
    tools:
      - "ScrapeWebsiteTool"
      - "InternetSearchTool"
      - "InstagramSearchTool"
    tasks:
      review_content:
        description: |
          Evaluate content produced by team members, provide feedback, and approve or request refinements as needed.
        expected_output: "Reviewed and refined content aligned with campaign and client goals."
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
print(result) # 2/10


```

    [1m[95m [2024-11-01 14:18:36][DEBUG]: == Working Agent: Lead Market Analyst[00m
    [1m[95m [2024-11-01 14:18:36][INFO]: == Starting Task: Analyze product and competitor landscape for key trends, strengths, and areas of opportunity in the market.
    [00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mTo begin the analysis of the product and competitor landscape, I need to gather information on key competitors and market trends in the digital marketing industry. I will start by searching for the latest trends and competitor information.
    
    Action: InternetSearchTool
    Action Input: {"query": "latest trends in digital marketing 2023"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    [{'title': 'The Top 14 Digital Marketing Trends of 2023, According to Experts', 'href': 'https://www.wordstream.com/blog/ws/2023/01/20/digital-marketing-trends', 'body': '2023 digital marketing trends & predictions, according to experts. To get a well-rounded perspective on how the 2023 digital marketing trends will impact consumers, marketers, and brands, we came up with some key questions and interviewed marketers, advertisers, and business owners across verticals. ... "AI offers new ideas and angles that we ...'}, {'title': '9 Cutting-Edge Digital Marketing Trends for 2023 - Marketing', 'href': 'https://elements.envato.com/learn/digital-marketing-trends-2023', 'body': 'We predict 2023 to be a year of embracing new technology, revisiting traditional marketing methods, and doing things differently. So what digital marketing trends will shape the coming year? From the "Gen Z effect" to AI domination, get ahead with these top digital marketing trends for 2023.'}, {'title': '2023 Digital Marketing Trends To Take Advantage Of Now - Forbes', 'href': 'https://www.forbes.com/councils/forbescoachescouncil/2022/11/15/2023-digital-marketing-trends-to-take-advantage-of-now/', 'body': 'Many businesses are already mapping out their digital marketing plans for 2023 and beyond. Smart businesses are looking for the latest trends to incorporate to maximize their results and impact.'}, {'title': "The Top Marketing Trends of 2024 & How They've Changed Since 2023 [Data ...", 'href': 'https://blog.hubspot.com/marketing/marketing-trends', 'body': "FREE 2023 STATE OF MARKETING REPORT. ... Keeping up with the latest marketing trends isn't always easy. But, to succeed in the fast-paced marketing world — and maintain a sense of relevance with your audience — it's vital to stay ahead of them. ... CMO and Co-Founder of NP Digital says. So, it's no surprise our survey found that 53% of ..."}, {'title': 'What are the Top Digital Marketing Trends for 2023?', 'href': 'https://digitalmarketinginstitute.com/blog/what-are-the-top-digital-marketing-trends-for-2023', 'body': "As customers become more discerning about the products and services they buy, there's also a global cost of living crisis that brands will need to plan for in 2023. Find out the top digital marketing trends for 2023 so you can plan your campaigns and tweak your strategy."}]
    [00m
    [32;1m[1;3mThought: I have found several sources that discuss the latest trends in digital marketing for 2023. I will read the content of one of these sources to gain detailed insights into the trends.
    
    Action: Read website content
    Action Input: {"website_url": "https://www.wordstream.com/blog/ws/2023/01/20/digital-marketing-trends"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    The Top 14 Digital Marketing Trends of 2023, According to Experts
    Blog
    Free Tools
    	Google AdsPerformance Grader
    		Find out if you're making costly mistakes—and how to fix them.
    Get your grade
    	Facebook AdsPerformance Grader
    		Get ready to improve your reach, results, and ROI—fast.
    Get your grade
    FreeKeyword Tool
    			Discover the best keywords for your PPC and SEO goals.
    Find your keywords
    FreeWebsite Grader
    		See how your website stacks up against the competition.
    Get your grade
    See More Free Tools
    ResourcesGuides135 Best Words & Phrases for Marketing with Emotion 
    PPC 101: Complete Guide to Google Ads
    25 Ways to Increase Traffic to Your Website
    Recommended ReadsGoogle Ads Benchmarks for Your Industry
    Facebook Ads Benchmarks for Every Industry
    15 Insanely Effective Sales Promotion Examples
    PPC 101What is PPC?
    The Last Guide to Account Structure You’ll Ever Need
    Intro to Social Media Marketing
    Recent Articles
     42 Veterans Day Greetings & Messages for Any Audience (+6 Quick Tips!) Want to Learn How the Pros Write CTAs? Click Here Now! 10 Key Insights to Inform Your Marketing Plans in 2025 
    See More Resources
    Solutions
    About Us
    Free Google Ads Grader
    Get a Demo
    Free Google Ads Grader
    Home — Blog — These 14 Digital Marketing Trends Will Have the Biggest Influence in 2023 
    These 14 Digital Marketing Trends Will Have the Biggest Influence in 2023 Author: Lisa Smith
     Last Updated: January 25, 2024
     |
     Business Planning & Strategy
    In case you missed it, we covered some of the top digital marketing trends of 2023, which will include a mix of both familiar tactics and emerging technologies and strategies.
    There’s blockchain, AR/VR, live streaming, short-form video, and more. Many of these trends have already made their debut in previous years but are picking up momentum and evolving in 2023. So in this post, we’re going to take a deeper dive into the patterns within these trends and how they will influence marketers throughout the year.
    2023 digital marketing trends & predictions, according to experts
    To get a well-rounded perspective on how the 2023 digital marketing trends will impact consumers, marketers, and brands,  we came up with some key questions and interviewed marketers, advertisers, and business owners across verticals. They include:
    How will social platforms adapt to TikTok?
    Will AI content generation help or hurt content marketing?
    What will DEI efforts focus on this year?
    Will automation and PPC learn to get along?
    How will the rise of Gen Z impact brand strategies?
    What changes will we make in privacy?
    Will we get a handle on the supply chain?
    Let’s take a look at what they said.
    How will social platforms adapt to TikTok?
    Imitation is the most sincere form of flattery, right?
    1. Some social platforms will imitate or partner with it
    Six-year-old TikTok is one of the most popular social media platforms in the world with more than one billion monthly active users. It’s especially popular with consumers born between 1997 and 2012, who are collectively known as Gen Z.
    As this generation grows in spending power, other social platforms are scrambling to keep up—and to stay relevant.
    Their most likely tactic in 2023 is imitation.
    For her part, Tetiana Sira, senior digital marketing executive at Energy Casino, noted social media platforms have already responded by either copying TikTok outright, such as Instagram with Reels, or by implementing more vertical videos, like Pinterest and Twitter.
    Image source
    “There’s clearly a much higher effort put into video consumption and trying to make it as seamless as possible,” she said.
    Shahnawaz Sadique, a blogger at CommonStupidMan.com, however, noted social platforms may also partner with TikTok to gain access to its content and users—or even acquire TikTok or similar platforms outright.
    But, according to Carly Campbell, a blogger at Mommy On Purpose, there’s more to TikTok’s success than just short-form videos. It also has an easy-to-navigate interface and “intense focus” on user-generated content. These elements have helped the platform stand out and allowed it to create an engaged community.
    “To effectively challenge TikTok, other social platforms will need to mimic the same user experience that has made it so popular,” she added.
    2. Others will distinguish themselves
    That being said, there are several ways social platforms could also distinguish themselves from TikTok in 2023.
    One is by focusing on other types of content, such as long-form videos or live streaming, Sadique said.
    Another is investing in AI to improve video capabilities and generate more engaging content, added Jessica Shee, manager at iBoysoft.
    Campbell pointed to Snapchat, which is focusing on messaging.
    “Similarly, Facebook is focusing heavily on its gaming and group features, which can also be seen as a way to set itself apart,” she added.
    What’s more, Averi Melcher, owner of OSO Content, said platforms may re-evaluate their creator monetization strategies to better compete with the TikTok Creator Fund, which compensates users who generate at least 100,000 authentic video views monthly, to encourage creators to stay on their platforms as well.
    “We are seeing this with Instagram subscriptions, Facebook Reels, Pinterest’s beta program, and YouTube Shorts,” Melcher said. “If these platforms are able to incentivize and better reward creators, I think they will successfully gain back some market share.”
    Image source
    In addition, Melcher said it’s possible social platforms may improve their creator campaign attribution models or even build out native communities to connect people with brands—and therefore better ways to monetize their content.
    Will AI content generation help or hurt content marketing?
    If you have a finely tuned content marketing funnel, you’ll want to tune into the perspectives on this 2023 digital marketing trend.
    📣 Free guide >> The 30 Best Ways to Promote Your Business (With or Without Money)
    3. The advantages will abound…
    AI content generation started to go mainstream in 2022—and it will play an important role in marketing in the year to come.
    For starters, there’s potential for AI content generation as a brainstorming tool for marketers.
    “At Delightful Ads, we are already using it as a starting point for creating assets needed for our clients’ advertising campaigns—such as ad copy, headlines, hooks, graphics and even videos,” said digital ads strategist Jenn Martin. “AI offers new ideas and angles that we may not have thought of on our own.”
    AI will also be increasingly used to create content tailored to specific audiences. That means taking data like demographics, past purchases, interests, and needs into account, added blog coach Becca Klein.
    “[The resulting content] could include personalized emails or targeted ads that would be more tailored to the individual needs of their customers,” Campbell added. “As they become increasingly good at utilizing natural language processing, marketers will be able to develop more dynamic and engaging campaigns.”
    Mailshake’s AI email generator SHAKEspeare
    As a result, relevance and effectiveness will also increase.
    And, with better content, brands are better poised to win market share.
    “With [the support of AI tools], content can be kept at a high standard over time and the brand’s reputation may be protected,” said Peter Drow, head of marketing at Nccttingtools. “Establishing a name for yourself in the marketplace is a surefire way to give yourself an edge over the competition.”
    And, with quality content comes other advantages.
    “AI content generation tools can help businesses to ensure the accuracy and consistency of their content, as the tools can be trained to adhere to specific guidelines and standards,” Sadique said.
    In turn, this can yield improved return on investment (ROI), Drow said.
    This may also help increase revenue.
    “Using AI-driven processes, businesses can crank out more and better blog and social media material,” Drow added. “Users’ confidence in these methods can be bolstered through these outlets [as] customers are more likely to buy from reputable companies.”
    What’s more, AI helps improve the speed of content creation, said Sugiya Oki, who runs reviews site We Recommend.
    “Inevitably, AI-powered writing tools will outpace their human competitors when it comes to speed,” Drow added. “To that end, they are a time-saver for businesses and content producers. This frees up valuable time that may be used for other important tasks, such as improving the quality of the content provided by AI.”
    Sadique agreed.
    “AI content generation tools can help businesses to create a high volume of content quickly and efficiently, potentially increasing productivity and allowing businesses to focus on other tasks,” he added.
    “This can significantly impact a digital marketer’s ability to quickly and effectively create content, as they no longer have to manually research and write blog articles, product descriptions, web content, or other forms of written marketing material,” Campbell said.
    Meanwhile, AI tools will increase competition on short-tail keywords with high volume. That’s according to James De Roche, managing partner at Lead Comet, who also noted, “While there can be initial anxiety around this for businesses, it will push content marketers to pursue more relevant, long-tail keywords with more conversion potential.
    AI may even come to replace some of the services currently offered by search engines.
    “This is likely to impact the way that companies approach the use of digital PR and link building,” said Phelan Gowing-Mikellides, business development manager at Digital Trails. “As a result, we are very likely to see a shift in the way that businesses approach their online presence and the strategies they use to reach and engage their target audiences, including changes to their backlink profiles [in 2023].”
    AI may also automate the tasks performed by PR professionals.
    “Think writing and distributing press releases, managing social media accounts or, at the very least, populating digital channels with high-quality, perfectly optimized AI-generated content,” he added. “This has the potential to displace many PR professionals, as AI is able to perform these tasks more quickly and efficiently than humans.”
    🛑 Get updated 2024 content marketing trends here!
    4. Human copywriters, don’t panic
    But human copywriters shouldn’t panic—they will still have a role to play.
    That’s in part because the quality and effectiveness of the content AI produces are only as good as the inputs it receives.
    “It still takes an experienced marketer/advertiser to determine not only what prompts to submit, but also what is useful in selecting and refining the finished product,” Martin said.
    Oki agreed, noting AI doesn’t understand emotions and therefore can’t grasp search intent.
    Free guide >> The 120 Best Words & Phrases for Marketing with Emotion 😌
    “There’s still a lot of room for unique creativity that only the human brain can conquer, at least for now,” added Dmitry Sokhach, founder of Admix Global.
    That will help distinguish copywriter-generated content from “a massive increase in generic blog content,” as De Roche called it.
    “[The latter] may address the query, but it will lack the advice needed to satisfy search intent,” he said. “To do that, you need expert insights and advice.”
    This also ties into first-hand experience, niche expertise, and author authority, which we discuss in our 2023 SEO trends.
    What will DEI efforts focus on in 2023?
    In recent years, diversity, equity, and inclusion (DEI) has become an increasingly important business initiative. Here’s what experts see as being the key focal points this year.
    5. An emphasis on equity & intersectionality
    In 2023, companies will continue to foster more equitable workplaces—with the key word being “equity.”
    Eleanor Fletcher, CEO of The Best Brisbane, explained, “Equity in the workplace refers to the right to fair representation and equal progression chances for people with various backgrounds and experiences.”
    Garit Boothe, CEO of Digital Honey, noted this can include efforts to increase underrepresented groups, as well as to implement training and education programs and to promote diversity in leadership and decision-making roles.
    “It is also possible that there will be a greater emphasis on intersectionality and addressing the specific needs and experiences of diverse individuals,” he added.
    This is something we mentioned in our Best Marketing Strategies for Small Businesses 2023 post.
    Image source
    6. More accountability through quantification
    We can also expect more focus on accountability as organizations set targets to regularly measure progress.
    That’s according to Sadique, who also expects to see more data and analytics—such as tracking diversity and inclusion metrics and identifying areas for improvement—along with the continued growth of employee resource groups and increased focus on diversity in leadership.
    Jeff Romero, founder of Octiv Digital, agreed.
    “This entails tracking and analyzing diversity and representation at all organizational levels, utilizing competency frameworks and other strategies to lessen bias in hiring and promoting employees, and coaching front-line managers on how to conduct team meetings and collaborative sessions to promote equal respect for all opinions,” he added.
    Meanwhile, more companies will take steps to address DEI in campaigns that focus on underrepresented groups or specific social causes relevant to those groups.
    “We can expect this trend of greater representation in digital marketing campaigns to continue into 2023,” said Tomasz Niezgoda, head of marketing at Surfer.
    The same is true of content more broadly.
    “Marketers are increasingly looking to create content that is inclusive and resonates with a broader range of audiences,” said David Ellis, founder of Teranga Digital Marketing.
    Will PPC and automation make friends?
    We are all too familiar with Google’s push for automation. We asked experts where they see things heading, sentiment-wise.
    🚨 Want 2024 Google Ads trends and predictions? We’ve got you covered here!
    7. More positive attitudes toward automation
    In 2023, expect to see attitudes toward automation become more positive.
    “There has always been a bit of an aversion to automation and AI in marketing and PPC—a lot of people thinking that it will take their jobs,” said Sean Nguyen, a director at Internet Advisor. “The reality of it is automation is there to assist us and help us refine our jobs. While it might take over some tasks, that actually helps us to become better at more complex tasks that aren’t tedious or repetitive.”
    Shee agreed, noting automation can help with keyword research, ad testing, and analytics tracking.
    This is especially true when it comes to Google Ads.
    8. Greater efficiency in Google Ads
    To that end, we can expect Google Ads automation to grow as advertisers optimize their campaigns and reach their target audiences more effectively. This automation technology includes bid optimization, as well as ad creation and testing.
    The result is improved efficiency and effectiveness, Boothe said.
    That is especially true since the release of Google Ads’ Performance Max campaigns in 2021.
    “They learn how to identify where someone is on their path to purchase and show the right ad at the right time on the right Google channel to help a prospect move further along the customer journey,” Martin said. “And they even provide feedback on what is working and what inputs to test in the future that we hadn’t even considered.”
    As they become more familiar with the benefits of automation, marketers will likely rely on it more to manage their Google Ads campaigns. However, Sadique warned, it does have limitations, such as the potential for errors and biases. Learn more in our Performance Max Dos, Don’ts, Watchouts and Workarounds.
    “Businesses may adopt hybrid approaches that combine automation with manual management in order to get the best of both worlds,” he added.
    Learn more about a hybrid approach to automation in Google Ads here.
    How will the rise of Gen Z impact brand strategies?
    As Gen Z becomes more influential, marketers are adjusting their strategies to engage them. Here’s how:
    9. Greater influence on authenticity, transparency & accessibility
    For starters, focus on authenticity and transparency.
    “Brands will need to be genuine and transparent in their marketing efforts in order to appeal to this group,” Sadique said.
    Fletcher agreed.
    “Gen Z is drawn to marketing that is honest… they prioritize word-of-mouth advertising from friends and influencers and social evidence,” she added.
    That means incorporating user-generated content featuring consumers who resemble them and being transparent about their environmental impact, dedication to diversity, and core values.
    “Gen Z is more diverse and socially conscious than previous generations, so brands will need to prioritize diversity and inclusion in their marketing efforts in order to appeal to this group,” Sadique said.
    From our post on how to market to Gen Z.
    Campbell agreed, noting this also includes accessibility.
    “Through storytelling and cause marketing, brands can develop narratives that align with the values and beliefs of Gen Z and that work to create an emotional connection with their target audience,” she added.
    10. Immersive experiences
    Gen Z also values experiential marketing and immersive experiences, which are key to engaging this cohort, along with omnichannel marketing.
    “One of the main challenges for brands when it comes to engaging with Gen Z is understanding the various platforms and digital tools that this generation uses,” Campbell said. “Many Gen Zers have grown up with technology and are highly engaged on social media and digital networks. This means that for brands to capture their attention effectively, they must be present and active on the platforms their target audience uses.”
    Facebook and Instagram Shops offer immersive, ideal-for-Gen-Z experiences on social media
    What changes will we make in privacy?
    In addition to automation, privacy is the other unstoppable force that is carving out new paths in marketing each year. Let’s see what directions they’re headed in.
    11. More transparent data practices
    Meanwhile, privacy-friendly reporting dictates marketers must embrace a more transparent approach to data in which users give explicit consent—and, as Home Grounds CEO Alex Mastin noted, they must strike the right balance between personalization and privacy.
    To do so, marketers must clearly communicate their privacy practices to clients and stakeholders and be transparent about the data they are collecting and how it is being used.
    They should also use consent management systems and provide clear opt-outs.
    12. Greater adoption of privacy-friendly tracking technology
    “Additionally, they should invest in security tools that allow them to measure performance while still keeping customer data safe from misuse or attack by third parties,” Niezgoda said.
    But, Sadique noted, marketers can implement privacy-friendly tracking methods, such as using anonymous identifiers or aggregated data, to track and report on key metrics without collecting personal data.
    Google’s Topics API, the Facebook Conversions API, Facebook’s Aggregated Event Measurement, and Google Ads’ Enhanced Conversions are all examples of privacy-friendly technology solutions you can embrace in 2023.
    Boothe agreed, noting it may also involve working with third-party organizations that have expertise in privacy-compliant data management.
    “Marketers will need to be proactive in finding solutions that balance the need for data with the importance of protecting consumers’ privacy,” he added.
    Will we get a handle on the supply chain?
    Inflation and supply chain shortages were the biggest culprits behind Google Ads cost per lead increasing in 2022. Here’s what experts predict on the supply chain front.
    13. More businesses will adopt supply chain automation technology…
    To protect against the supply chain shortages that plagued the holiday season, businesses can diversify their sourcing, which includes working more with local suppliers, and/or strengthen ties with existing vendors.
    They can also implement multi-modal transportation and invest in analytics for greater supply chain visibility and to better anticipate and manage risk.
    Image source
    “[Supply chain automation technology] can help to expedite the movement of goods, reduce delays in the supply chain and increase efficiency,” Campbell said. “Automation technologies such as robotics, machine learning, big data analysis and artificial intelligence can enable businesses to identify and respond to supply chain disruptions.”
    14. …and build prevention and contingency plans
    In addition, building up inventory levels, implementing contingency plans, and looking for opportunities to increase efficiency and reduce costs can help mitigate bottlenecks.
    “Some businesses may also be considering shifts in their operations, such as moving production to different locations or adopting new technologies, in an effort to reduce their reliance on certain suppliers or markets,” Boothe added.
    Inventory optimization, which identifies when and where shortages may occur, is another tactic.
    “Additionally, businesses may also look to build more robust safety nets, such as insurance policies and emergency funds, to protect them from unexpected economic changes,” Shee said.
    Keep an eye on these 2023 marketing trends
    We’ve covered a lot of trends, patterns, and perspectives in this post, on everything from AI content to supply chain automation. Here is the final list to sum up our top digital marketing trends and predictions for 2023:
    Some social platforms will imitate or partner with TikTok
    Others will distinguish themselves
    The advantages of AI content will abound
    But human copywriters need not panic
    A greater emphasis on equity & intersectionality in the DEI realm
    The use of metrics and KPIs for greater DEI accountability
    More positive attitudes toward automation
    Greater efficiency (but caution) in Google Ads
    Catering to Gen Z’s affinity for authenticity, transparency & accessibility
    Immersive experiences on social
    Increase of transparency in data practices
    Greater adoption of privacy-friendly tracking technology
    The need for automation technology remains
    Building of supply chain prevention and contingency plans
    Want to see digital marketing trends from other years? We’ve got you covered!
    2021 Digital Marketing Trends
    2022 Digital Marketing Trends
    2023 Digital Marketing Trends
    Are you wasting money in Google Ads?
    Find out now with the Free Google Ads Grader.
    Get my grade
    Join 713K other marketers that receive our weekly newsletter!
    Loading...
    Meet The Author
    Lisa Smith
    Lisa Smith is a freelance writer who has covered the digital marketing industry for more than a decade.
    Loading...
    Shares
    Recommended for you
    10 Key Insights to Inform Your Marketing Plans in 2025
    October 29, 2024
    What Is Competitive Positioning? (+How to Create Yours)
    August 02, 2024
    Free Google Ads Performance GraderFree ToolHow to Create a Unique Value Proposition From the Ground UpGuide 
    7 Competitive Insights You Need to Beat the Competition
    August 21, 2024
    3 Complete SWOT Analysis Examples to Learn From
    June 18, 2024
    Comments
    Please read our Comment Policy before commenting.
    Unlock your business potential
    We offer innovative technology and unparalleled expertise to move your business forward.
    Get a demo
    Free ToolsGoogle Ads Grader
    Facebook Ads Grader
    Free Keyword Tool
    Website Grader
    Explore TopicsCopywriting
    Facebook Ads
    Industry News & Insights
    Instagram
    Lead Generation
    Marketing Ideas
    Paid Search Marketing
    SEO
    Social Media
    Websites & Online Presence
    LearnBlog
    Top Blog Posts
    Guides
    PPC University
    Marketing 101
    Infographics
    Social Ads 101
    Marketing TipsGoogle Ads Industry Benchmarks
    Facebook Advertising Benchmarks
    What Is a Good Conversion Rate?
    Guide to Google Ads Account Structure
    24 Ad Copy Examples
    How Much Does Google Ads Cost?
    CompanyAbout WordStream
    Solutions
    Careers
    Contact Us855.967.3787
    +1.866.971.0933
    								© WordStream. All Rights Reserved. | Privacy Policy | Security | Your Privacy Choices | Terms & Conditions | Cookie Policy
    [00m
    [32;1m[1;3mThought: I have gathered comprehensive insights on the latest digital marketing trends for 2023. Now, I need to explore the competitive landscape by identifying key competitors in the digital marketing industry and analyzing their strengths and strategies.
    
    Action: InternetSearchTool
    Action Input: {"query": "top digital marketing companies 2023"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    [{'title': 'Top 30 Digital Marketing Agencies - Oct 2024 Rankings - DesignRush', 'href': 'https://www.designrush.com/agency/digital-marketing', 'body': 'Partner with the #1 Rated Digital Marketing Agency. Outsmart the competition with best-in-class digital marketing services. With over 1,000 5-star reviews online, SmartSites is America #1 rated digital marketing agency. Call (201) 870 6000 for a free [... view the full profile of SmartSites] Paramus, New Jersey.'}, {'title': 'Best Digital Marketing Agencies - Oct 2024 Reviews | Clutch.co', 'href': 'https://clutch.co/agencies/digital-marketing', 'body': 'The average hourly rate for Digital Marketing providers is $85.53/hr. The best rated Digital Marketing companies with the lowest hourly rates are: Intelus Agency (5 stars, 9 reviews) - < $25/hr. Ecommerce Ally (5 stars, 9 reviews) - < $25/hr. Creation Infoways Pvt Ltd (5 stars, 8 reviews) - < $25/hr.'}, {'title': 'Top 10 Best Digital Marketing Agencies - 2024 Review - QuickSprout', 'href': 'https://www.quicksprout.com/best-digital-marketing-agencies/', 'body': 'The 10 Best Digital Marketing Agencies. Top Picks: 97th Floor, DigitalMarketing.com, Location3, Mad Fish Digital, Max Audience, Rise Interactive, Split Reef, WebFX, NP Digital, Reach Digital. These are the most well-respected and trusted agencies on the market today. Four, in particular, stand out from the rest.'}, {'title': 'Best Digital Marketing Services in 2024: Compare Reviews on 6000 ... - G2', 'href': 'https://www.g2.com/categories/digital-marketing', 'body': "These insights are written by G2's Market Research team, using actual user reviews for OpenMoves, left between November 2022 and December 2023. OpenMoves is a New York-based digital marketing agency specializing in performance marketing and SEO, with a focus on ROAS-driven e-commerce and lead-generation brands."}, {'title': 'Top Digital Marketing Agencies in the United States 2024', 'href': 'https://influencermarketinghub.com/digital-marketing-agency-usa/', 'body': "2023 was a massively successful year for Fresh Content Society's clients. They benefited from the agency's expertise to see the following results: ... Directive is a top-tier digital marketing company that redefines the concept of client-agency partnership. Unlike traditional agencies, Directive offers a dedicated team of search marketing ..."}]
    [00m
    [32;1m[1;3mThought: I have identified a source that lists the top digital marketing agencies for 2023. To gain detailed insights about these competitors, I will read the content from one of these sources.
    
    Action: Read website content
    Action Input: {"website_url": "https://www.designrush.com/agency/digital-marketing"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    Attention Required! | Cloudflare
    Please enable cookies.
    Sorry, you have been blocked
    You are unable to access designrush.com
    Why have I been blocked?
    This website is using a security service to protect itself from online attacks. The action you just performed triggered the security solution. There are several actions that could trigger this block including submitting a certain word or phrase, a SQL command or malformed data.
    What can I do to resolve this?
    You can email the site owner to let them know you were blocked. Please include what you were doing when this page came up and the Cloudflare Ray ID found at the bottom of this page.
    Cloudflare Ray ID: 8dbc841c1f002058
    •
     Your IP:
     Click to reveal
    34.86.217.198
    •
    Performance & security by Cloudflare
    [00m
    [32;1m[1;3mI encountered a security block while trying to access the detailed content from DesignRush. I will attempt to access another source to gather insights on top digital marketing companies.
    
    Action: Read website content
    Action Input: {"website_url": "https://clutch.co/agencies/digital-marketing"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    Just a moment...Enable JavaScript and cookies to continue
    [00m
    [32;1m[1;3mI'm encountering access issues with the websites from which I intended to gather detailed information about top digital marketing companies. I will attempt to access another source to obtain this information.
    
    Action: Read website content
    Action Input: {"website_url": "https://www.quicksprout.com/best-digital-marketing-agencies/"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    Top 10 Best Digital Marketing Agencies - 2024 Review
    		Skip to content
    Creating Websites
    Blogging
    Starting a Business
    Marketing
    Operations
    About Us 
    Main Menu
    Creating Websites
    Blogging
    Starting a Business
    Marketing
    Operations
    About Us 
    Main Menu
    Creating Websites
    Blogging
    Starting a Business
    Marketing
    Operations
    About Us 
    Compare The Best Digital Marketing Agencies
    December 30, 2023
    Quick Sprout Editorial
    Disclosure: Our content is reader-supported, which means we earn commissions from links on Quick Sprout. Commissions do not affect our editorial evaluations or opinions.
    It’s no secret that every business needs to leverage digital marketing to stay competitive in the modern era. However, what is a secret to some businesses is that sometimes you need extra guidance to really dial in your digital marketing strategy. You may want to try and go it alone, but that’s not always in the best interest of your company. 
    Whether you need help with a new campaign or simply want general marketing consulting, a digital marketing agency provides you with the professional and proven marketing assistance you need to reach your goals.
    The 10 Best Digital Marketing Agencies
    Top Picks: 97th Floor, DigitalMarketing.com, Location3, Mad Fish Digital, Max Audience, Rise Interactive, Split Reef, WebFX, NP Digital, Reach Digital.
    These are the most well-respected and trusted agencies on the market today. Four, in particular, stand out from the rest.
    Our Favorite Digital Marketing Agencies:
    Reach Digital — Best for small businesses
    NP Digital — Best global digital marketing agency
    97th Floor — Best for enterprise B2B brands with bigger budgets
    Rise Interactive — Best for attracting ecommerce customers
    Check out the in-depth reviews for each agency below. I cover their service offerings, their specializations, the budgets they work with, and the business types that will benefit most from their services.
    Best Digital Marketing Agencies Reviews
    Reach Digital — Best for Small Businesses
    Reach Digital was built with one main purpose in mind: to help small businesses hit and surpass their marketing goals so they can succeed online.
    Reach Digital doesn’t want to be just another digital marketing agency. They strive to build long-term relationships with the businesses they work with based on three pillars: transparency, honestly, and exceptional customer service.
    They provide expert digital marketing services that are tailored to each business’ unique goals, with a strategy that is derived from marketing data that is specific to your industry allowing for more informed decisions.
    Although Reach Digital can do everything from web design to lead generation, their focus is in three key areas:
    SEO
    Facebook Lead Gen
    Google Ads
    Some of the main services include:
    No contract required
    Keyword research
    Technical SEO
    Full-service ad campaigns
    Social media ads
    Copywriting
    Lead nurturing
    Monthly reporting
    Overall, Reach Digital aims to generate you more leads through SEO strategies and ad campaigns that target your specific audience. The best part is, Reach Digital does all of the heavy lifting, so you can focus on growing your business.
    You’ll have to schedule a 30 minute call with Reach Digital to see if you are the right fit, and to generate a customized marketing strategy that is unique to your goals.
    For small businesses looking for the perfect digital marketing agency to grow with, Reach Digital is it.
    Schedule a free consultation to get started.
    NP Digital — Best Global Digital Marketing Agency
    NP Digital is a behemoth in the digital marketing arena. In fact, the agency is so big that it has nearly 20 expert agencies spread across more than a dozen countries, including Hong Kong, Japan, France, Italy, Brazil, and Columbia, just to name a few. This is all the proof you need to know that NP Digital is there for you, in person, when you’re ready.
    Being one of the largest global agencies, NP Digital specializes in innovative marketing solutions that are driven by data, and that adapt to the ever-changing landscape that is digital marketing. NP Digital lets its results speak for itself, boasting some of the biggest names in the world, like Western Union, ESPN, and Mitsubishi Motors.
    NP Digital also offers proprietary SEO and keyword research tools in the form of Ubersuggest and Answer The Public. These tools help you find the keywords that matter so you can design a data-led marketing strategy that actually converts.
    NP Digital develops its marketing solutions around four key areas—data and analytics, earned media, paid media, tech and tools. Here’s more of a peak of what that looks like:
    Dashboard development
    Conversion rate optimization
    Ad operations
    SEO
    Influencer marketing
    Organic social media
    Email marketing
    Paid serach
    Paid social
    Streaming
    Ubersuggest
    Answer The Public
    Google Ads grader
    You can reach out to NP Digital to set up a meeting with the team in your country. Whether you are from Malaysia, Chile, or Canada, NP Digital has a team there waiting to help you take your marketing strategy to the next level.
    NP Digital is built for large, enterprise companies, and is a true global agency.
    NP Digital claims to be the “alternative to the status quo,” and by putting customer results at the foreground of every decision it makes, NP Digital might just be a real standout. 
    97th Floor — Best for Enterprise B2B Brands with Bigger Budgets
    97th Floor is a full-service digital marketing agency that stands out for its impressive client roster, which includes Salesforce, Dell, Oatly, and even the NBA’s Utah Jazz. They’re no stranger to larger, enterprise-tier organizations.
    Where they stand out though is in their patent-pending Palomar software suite. This is a set of tools that will analyze relevant marketing data in real-time to help you develop a marketing strategy that’s optimized for your brand.
    Included in this suite are:
    Customer Search Intent. This is their keyword research tool that allows your team to see the ROI of relevant target keywords.
    Semantic Analysis. This tool analyzes the word choices for a large volume of web pages related to your brand’s topics — and furnishes you with valuable context and data about what topics and words people focus on so you can use them for your marketing.
    SERP Intelligence. Palomar crawls through competitors’ digital content and analyzes it for the best topics your team should focus on and target for keywords.
    97th Floor offers digital marketing services across four main categories—SEO, paid media, creative, and automation. Top services in those categories include:
    Link building
    Keyword research
    Technical SEO
    Retargeting
    Social media ads
    Audience testing
    Copywriting
    Graphic design
    Content marketing
    Email marketing
    Lead nurturing
    Analytics
    Overall, 97th Floor aims to reach current and prospective clients across every potential digital channel to maximize consumer interaction with your brand.
    After your initial consultation, 97th Floor will give you a customized game plan for your strategy. You’ll be provided with a dedicated marketing team based on factors like work experience, industry knowledge, area of expertise, and even common interests.
    But with that said, it’s a better option for larger businesses with substantial marketing budgets. Consider hiring 97th Floor if you have a monthly marketing budget of $10k-$25k. They even have plans for companies with marketing budgets of $100k+ per month.
    Rise Interactive — Best for Attracting Ecommerce Customers
    Rise Interactive is a full-service digital marketing agency that has offices based throughout the United States and in Mexico City and Buenos Aires.
    Where they stand out is their offerings when it comes to online marketplace listings and advertisements.
    For example, their Amazon advertising services help you capture more customers on Amazon through on-page ads and cross-channel integrations. They will also work with you to optimize your advertising game on other online marketplaces such as the Walmart Media Group.
    The marketplace listing services are perfect for ecommerce businesses that sell products on multiple platforms. As such, I highly recommend them to ecommerce websites looking to make a bigger splash online.
    Here’s a more comprehensive list of Rise’s offerings:
    Paid search
    SEO
    Amazon marketing
    Programmatic advertising
    Affiliate marketing
    Social media marketing
    Email marketing
    Marketplace listing
    Web design
    Content marketing
    Customer experience analytics
    This agency developed a proprietary multi-channel strategy called “Interactive Investment Management.” The idea behind this approach is to fluidly optimize media content to whichever tactics or channels drive the highest ROI.
    Rise is a top choice to consider for businesses that prefer an analytically-driven approach to digital marketing.
    You’ll also benefit from a suite of tools to provide you with real-time performance metrics, analysis, and optimization for your campaigns and strategy as a whole.
    Unlike some of the other options on our list, Rise does not have specific monthly budget requirements to be considered as a client. They work with companies of all different shapes and sizes.
    More Great Digital Marketing Agencies
    WebFX — Best for Dominating Local SEO
    WebFX is an award-winning digital marketing agency with over 20 years of experience.
    While they offer many of the typical services you expect from a full-service agency, WebFX is best known for its SEO services. In fact, they’ve driven clients to have over 255,000 items ranked on page one of Google.
    Where they really stand out is in their local SEO service. They’ll give you a dedicated local SEO expert, on-page local SEO services like product keyword research, citation management across in-network citations, review management, and custom reporting.
    Overall, they’re a great choice whether you already have a local SEO campaign, just getting started, or you just want someone to take care of the nuts and bolts for you.
    In addition to SEO services, WebFX also provides:
    PPC services
    Social media marketing services
    Web design services
    Content marketing services
    Conversion rate optimization (CRO)
    Amazon optimization
    Lead generation services
    They work with businesses across a wide range of industries, including education, food and beverage, finance, automotive, industrial, legal, home and repair, insurance, medical, professional services, nonprofits, real estate, retail, tech, tourism, and more.
    WebFX can accommodate budgets of all different sizes. Whether you’re looking to spend $300 per month or $10,000+ per month, this will be a top digital marketing agency for you to consider.
    DigitalMarketing.com — Best for Nailing Your Campaign the First Time
    DigitalMarketing.com offers all things related to—you guessed it—digital marketing.
    This full-service digital marketing firm has worked with some of the fastest growing and most well-established brands in the world. Some of their top clients include Ferrari, AT&T, Fox, CBS, Coca-Cola, Warner Brothers, Boeing, and Bank of America.
    Their most standout offering, though, is a tool used to beta test your marketing campaign. In fact, they have their own technology called “Capture it All” which uses data to simulate beta tests. That helps you plan the best possible course of action for your marketing campaigns.
    That can help you anticipate how your marketing campaign will ultimately perform. However, it should be taken with a grain of salt, as there are many outside factors that can ultimately determine the course of your campaign.
    However, this is a pretty incredible feature and one I haven’t seen other digital marketing agencies offer. It can help you ensure you launch the best marketing campaign for your brand on the first try—rather than investing a ton of money only for it to flop.
    They also offer a great PR management service that’s great for brands who need to put their best foot forward after a dent in their reputation. A more comprehensive list of their offerings includes:
    SEO
    Content marketing
    PPC management
    Social media advertising
    Public relations
    Reputation management
    Like most marketing firms, DigitalMarketing.com does not have prices listed on its website. They don’t require you to disclose your budget to receive a quote. Contact them today to schedule a free consultation.
    Mad Fish Digital — Best for Socially-Conscious Brands and Nonprofits
    Mad Fish Digital is a digital marketing firm that chooses to work with businesses that have a positive impact on their communities and the environment.
    Top Mad Fish Digital services include:
    Content strategy
    Email marketing
    Social media marketing
    Data-driven design
    Visual storytelling
    Branded video campaigns
    PPC
    Lead generation and CRO
    Brand awareness
    SEO
    Keyword mapping and on-site optimization
    Technical SEO
    As you can see, Mad Fish Digital covers all aspects of digital marketing. They specialize in certain industries, such as healthcare, higher education, sustainability and environment, SaaS, B2B, banking, hospitality, event planning, and more.
    Mad Fish Digital is innovative, genuine, and fully accountable for your campaigns. They’ll keep you informed through each step of the process.
    If you have a nonprofit, environmentally friendly, or sustainable organization, Mad Fish Digital will be a top agency for you to consider working with.
    Split Reef — Best for Climbing to the Top in the App Store
    Split Reef is a digital marketing agency with offices in Ohio and Florida.  They provide a wide range of digital marketing services, including other offerings like custom web applications, iOS and Android app development, progressive web apps, and more.
    If you’re also in the market for custom design and development, you can use Split Reef as an all-in-one solution.
    For example, let’s say you want to create an app designed for wearable devices, like a smartwatch. Split Reef can help you design the app, develop it, and then handle your digital marketing as well.
    Split Reef also specializes in responsive web designs and website speed optimization. Their overall list of offerings is a bit smaller than the other services:
    SEO
    CRO
    Social media marketing
    Reputation management
    Another unique niche carved out by this agency is the legal industry. Split Reef has extensive experience with digital marketing for law firms. They have a team of legal content writers, and other tools that are specifically designed to boost traffic and conversions for legal websites.
    While the two are definitely unrelated, Split Reef is a top choice for both mobile app marketing and law firm marketing.
    No matter what type of budget or project you have in mind, Split Reef can work with your budget. Services start in the $1,000-$5,000 range and go all the way up to over $50,000.
    Max Audience — Best for Creating An Automated B2B Marketing System
    Max Audience has offices on both coasts of the United States, with three locations in California and another in New York.
    While they offer a variety of marketing solutions—including content, PPC, and SEO marketing—they also offer a great marketing automation solution. This allows you to automate repetitive tasks so your system captures leads and customer data without you needing to constantly check in on it.
    They’ll work with you to set up a lead generation, lead nurturing, and segmentation system.
    They’ll even pair it with a CRM to really boost your ROI. This will help inform you and keep you connected with your customers no matter where they are in your funnel.
    Their CRM also gives you analytics to help translate data about your clients into insights that can help enrich their experience and turn them into customers.
    Overall, they leverage automation to maximize ROI in marketing campaigns. They follow the lead of Fortune 500 companies and B2B organizations by applying cost-effective and fail-proof automated campaigns to great success.
    A full list of their offerings can be found below:
    SEO
    SEM
    Email marketing
    CRM marketing
    Social media marketing
    Marketing automation
    Brand management
    Max Audience has worked with small organizations and large companies throughout the world. They prioritize ROI with all digital marketing campaigns.
    In addition to digital marketing services, Max Audience also provides web design services and messaging solutions for its clients. The messaging solutions will help you reach your customers across various digital touchpoints.
    Prices for the Max Audience digital marketing services are unavailable online. You’ll need to speak to their sales team for a custom quote.
    Location3 — Best for Brick-and-Mortar Franchises
    Location3 is a full-service digital marketing firm that specializes in marketing services for franchise brands. They work with national franchise chains like Honey Baked Ham, Advantage Rent a Car, and Edible Arrangements.
    As such, they’re especially great if you have a brick-and-mortar presence.
    They have a property platform called “LOCALACT” that is designed for franchise systems and companies with a significant brick-and-mortar footprint. These custom online dashboards allow Location3’s clients to scale and power their digital marketing campaigns in real-time on a national and local level.
    A more comprehensive look at their offerings includes:
    Local listings management
    Paid search
    Paid media
    SEO
    Content creation
    Local SEO
    Web analytics
    Franchise development
    If you own a franchise and need assistance marketing your locations online, Location3 will be the best option for you.
    How to Find the Best Digital Marketing Agency For You
    With so many options out there, finding the best digital marketing agency for your business can be a challenge. To find a firm that will deliver the best results for your company’s needs, there are certain factors you should take into consideration.
    This is the methodology that we use and recommend here at Quick Sprout to evaluate the best digital marketing firms. It isn’t a one-size-fits-all approach, though. Some factors will be more important than others, depending on your business needs.
    Outcomes and Outputs
    Be wary of services that promise to simply give you “more customers.” Not only is that incredibly vague, but it should already be an obvious goal of every single digital marketing service.
    Instead, focus on services that offer specific services like increasing traffic or getting qualified leads.
    Beware of any marketing agency that promises you the moon. Any goals you and the agency set should be realistic and attainable while still being ambitious.
    Claims promising a 200% engagement increase or a #1 ranking on Google should be treated with suspicion. This is not to say that those goals won’t happen or the business hasn’t helped other brands achieve such success. It’s just a red flag if they overpromise a specific outcome and underdeliver. 
    Industry and Niche Experience
    As you narrow down your list, keep an eye out for digital marketing agencies with experience that’s relevant to your type of business. They’ll be able to draw from experience to tailor their services to your business and target audience.
    Small businesses should go with an agency like Page Prosper. Location3 is better suited for franchises and Fortune 500s. If you run a non-profit organization, consider using a digital marketing service like Mad Fish Digital. 
    Looking at an agency’s client list will give you a good idea of the types of businesses and industries where they’ve delivered proven results.
    If you are having trouble finding an agency that can handle your specific industry, consider sorting your candidates into broader categories first. For example, if you are a B2B company, you can eliminate agencies that focus on B2C. 
    Further sort the remaining agencies on your list by whether they have expertise with the type of customer you’re seeking. For example, marketers that focus on clients in service industries will offer different skills than those that focus on clients that sell products.
    Team Scope and Capacity
    Digital marketing agencies come in diverse shapes and sizes—ranging from agencies with less than a dozen employees to those with several hundred people working on digital marketing efforts.
    If you want a bit more focus on you and your brand, a smaller agency will typically be able to provide that. If the agency only has a dozen people and you’re one of their hand-picked clients, the entire staff will be aware of your brand and its marketing efforts.
    Conversely, smaller agencies may not offer all of the services you’re looking for. So you should stick with larger firms if you want a wide range of digital marketing solutions under one roof. 
    For example, NP Digital is a standout as a true global digital marketing agency, with teams spread across the globe.
    The best agencies also provide you with a dedicated point of contact or account manager that’s readily accessible when you have questions or concerns. 
    Marketing Channel Focus and Methods
    While there is definitely some overlap from company to company, some marketing agencies specialize in one area over another. 
    For example, 97th Floor might specialize in technical SEO and link building, whereas WebFX puts more focus on local search listings or marketplace optimization.
    Beyond basic digital marketing services, some agencies offer specialty solutions for things like web design, online reputation management, app development, and other needs in the digital space.
    For example, Spit Reef stands out from the pack for its custom web application, iOS, and Android app development. DigitalMarketing.com is also excellent because they offer a powerful, technology-driven approach. Rise Interactive is great for ecommerce businesses because they help market your brand on different online marketplaces and listings.
    Experience
    The right digital marketing service will have extensive experience in the areas that are important to you. For example, you don’t want to hire a provider to help with creating social media posts if that’s not something they have proven experience in. 
    Use reviews from real clients to find A DIGITAL MARKETING SERVICE with EXPERIENCE that aligns with your business’s SPECIFIC NEEDS.
    Do your research to find the providers’ past clients and portfolio to see if they’re able to provide the services you’re looking for. 
    If you can’t find any information about past projects, then chances are good this provider may not have enough experience to serve your company’s needs. 
    Best Digital Marketing Agencies: Your Top Questions Answered
    What is the best company for digital marketing?NP Digital is the best digital marketing company for most businesses because of its vast service offerings and proven results. They provide a more personalized service compared to other digital marketing agencies, and they won’t force you to sign a long-term contract.
    How much does digital marketing cost?Average digital marketing services cost around $500 to $5,000 per month. But the cost varies significantly based on the company you’re hiring and the services you need. The cost structure will also vary by company and service.For example, a complete website redesign or marketing video creation would have a fixed rate for the job. But ongoing ad management would be billed as a percentage of your monthly or quarterly advertising budget. PPC ad agencies typically charge 10% to 20% of your monthly ad spending.
    Is it worth paying for digital marketing?It’s worth it to pay for digital marketing if you don’t have the in-house marketing expertise to deliver results. It’s also worth paying for digital marketing if you need to generate revenue quickly. For example, having a digital marketing agency manage your ads can deliver quicker results than managing an in-house content creation strategy for long-term SEO. You can use digital marketing services to bridge the gap between your resources and goals. Maybe you’re really good at organic social media marketing. But you need help with technical SEO. In this case, it would be worth paying for digital marketing to bridge that gap.
    What are examples of digital marketing services?Examples of digital marketing services include social media ads, PPC search ads, content marketing, link building, email marketing, graphic design, affiliate marketing, lead generation, and conversion rate optimization. You’ll also find digital marketing services that specialize in SEO, including technical SEO, keyword research, and on-page SEO. Some digital marketing services focus more on consultancy, planning, and big-picture goal alignment. Others are better at implementing specific campaigns for fast results within your budget.
    The Top Digital Marketing Agencies in Summary
    NP Digital is the premiere global marketing agency, while Reach Digital is perfect for small businesses. They each deliver excellent SEO services, paid media, email marketing, and everything else you need to take your marketing strategy to the next level.
    Of course, all of the choices on this list are full-service digital marketing firms. They each have unique standout features and services that appeal to different business types and marketing objectives. The one that’s right for you is the one that meets your unique needs.
    Come with a Question.
    Leave with a Solution.
    Every day, we help people discover products and services that can help their business bloom.
    The Quick Sprout Promise
    Quick Sprout publishes original work by writers in the B2B space. All reviews are edited and fact-checked internally before publication. We will never gate an article or ask our readers to pay for our content.
    Quick Sprout publishes original work by writers in the B2B space. All reviews are edited and fact-checked internally before publication. We will never gate an article or ask our readers to pay for our content.
    Creating Websites
    Blogging
    Starting a Business
    Marketing
    Operations
    About Us
    Work with Us | Privacy Policy | Terms of Service
    Copyright © 2024 • Quick Sprout
     Scroll to Top
    [00m
    [32;1m[1;3mThought: I have gathered detailed information about the latest digital marketing trends for 2023 and insights into top digital marketing agencies and their specialties. With this information, I can provide a comprehensive analysis of the product and competitor landscape.
    
    Final Answer:
    
    **Detailed Report on Digital Marketing Trends and Competitor Insights for 2023**
    
    **Digital Marketing Trends 2023:**
    
    1. **Social Media Adaptation to TikTok:**
       - Social platforms are imitating TikTok's short-form video style. Instagram Reels and Twitter's vertical videos are examples. Some platforms may even partner with or acquire TikTok-like platforms.
       - Platforms are also distinguishing themselves by focusing on long-form videos, live streaming, and improving AI video capabilities.
    
    2. **AI Content Generation:**
       - AI is becoming mainstream, offering tools for brainstorming and creating content tailored to specific audiences. This includes personalized emails and targeted ads.
       - AI helps maintain content quality and can automate tasks traditionally handled by PR professionals. However, human expertise remains crucial for refining AI outputs.
    
    3. **Diversity, Equity, and Inclusion (DEI):**
       - Companies are emphasizing equity and intersectionality, with efforts to increase underrepresented groups and promote diversity in leadership.
       - There is a focus on accountability through data and analytics, tracking diversity metrics, and measuring progress.
    
    4. **Automation in PPC:**
       - Positive attitudes toward automation are increasing. Google Ads automation, including Performance Max campaigns, is improving efficiency and effectiveness.
    
    5. **Gen Z's Influence:**
       - Brands are focusing on authenticity, transparency, and accessibility to appeal to Gen Z. User-generated content and environmental transparency are key.
       - Experiential marketing and immersive experiences are important for engaging Gen Z.
    
    6. **Privacy Practices:**
       - Marketers are adopting transparent data practices and privacy-friendly tracking technologies, balancing data needs with consumer privacy.
    
    7. **Supply Chain Automation:**
       - Businesses are adopting supply chain automation technologies to increase efficiency and manage risks, along with building prevention and contingency plans.
    
    **Top Digital Marketing Agencies 2023:**
    
    1. **Reach Digital:**
       - Best for small businesses, offering SEO, Facebook Lead Gen, and Google Ads services without contracts.
    
    2. **NP Digital:**
       - A global digital marketing agency with teams in over a dozen countries. Specializes in data-driven marketing solutions and offers tools like Ubersuggest for SEO and keyword research.
    
    3. **97th Floor:**
       - Known for working with enterprise B2B brands. Offers services across SEO, paid media, creative, and automation, and uses its Palomar software suite for marketing strategy optimization.
    
    4. **Rise Interactive:**
       - Specializes in attracting ecommerce customers with services like Amazon marketing and multi-channel strategy optimization.
    
    5. **WebFX:**
       - Renowned for local SEO services and accommodating various budget sizes.
    
    6. **DigitalMarketing.com:**
       - Offers a unique tool for beta testing marketing campaigns and PR management services.
    
    7. **Mad Fish Digital:**
       - Works with socially conscious brands and nonprofits, offering a wide range of digital marketing services.
    
    8. **Split Reef:**
       - Focuses on app marketing and law firm marketing, offering SEO, CRO, and reputation management services.
    
    9. **Max Audience:**
       - Specializes in marketing automation solutions, helping businesses set up lead generation and CRM systems.
    
    10. **Location3:**
        - Best for franchise brands, providing local listings management and franchise development services.
    
    **Conclusion:**
    The digital marketing landscape in 2023 is characterized by technological advancements, increased focus on DEI, and the influence of Gen Z. Competitors in the industry are leveraging these trends to offer specialized and comprehensive services. The key to success lies in adapting to new technologies, maintaining transparency in data practices, and engaging diverse audiences effectively.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-01 14:19:00][DEBUG]: == [Lead Market Analyst] Task output: **Detailed Report on Digital Marketing Trends and Competitor Insights for 2023**
    
    **Digital Marketing Trends 2023:**
    
    1. **Social Media Adaptation to TikTok:**
       - Social platforms are imitating TikTok's short-form video style. Instagram Reels and Twitter's vertical videos are examples. Some platforms may even partner with or acquire TikTok-like platforms.
       - Platforms are also distinguishing themselves by focusing on long-form videos, live streaming, and improving AI video capabilities.
    
    2. **AI Content Generation:**
       - AI is becoming mainstream, offering tools for brainstorming and creating content tailored to specific audiences. This includes personalized emails and targeted ads.
       - AI helps maintain content quality and can automate tasks traditionally handled by PR professionals. However, human expertise remains crucial for refining AI outputs.
    
    3. **Diversity, Equity, and Inclusion (DEI):**
       - Companies are emphasizing equity and intersectionality, with efforts to increase underrepresented groups and promote diversity in leadership.
       - There is a focus on accountability through data and analytics, tracking diversity metrics, and measuring progress.
    
    4. **Automation in PPC:**
       - Positive attitudes toward automation are increasing. Google Ads automation, including Performance Max campaigns, is improving efficiency and effectiveness.
    
    5. **Gen Z's Influence:**
       - Brands are focusing on authenticity, transparency, and accessibility to appeal to Gen Z. User-generated content and environmental transparency are key.
       - Experiential marketing and immersive experiences are important for engaging Gen Z.
    
    6. **Privacy Practices:**
       - Marketers are adopting transparent data practices and privacy-friendly tracking technologies, balancing data needs with consumer privacy.
    
    7. **Supply Chain Automation:**
       - Businesses are adopting supply chain automation technologies to increase efficiency and manage risks, along with building prevention and contingency plans.
    
    **Top Digital Marketing Agencies 2023:**
    
    1. **Reach Digital:**
       - Best for small businesses, offering SEO, Facebook Lead Gen, and Google Ads services without contracts.
    
    2. **NP Digital:**
       - A global digital marketing agency with teams in over a dozen countries. Specializes in data-driven marketing solutions and offers tools like Ubersuggest for SEO and keyword research.
    
    3. **97th Floor:**
       - Known for working with enterprise B2B brands. Offers services across SEO, paid media, creative, and automation, and uses its Palomar software suite for marketing strategy optimization.
    
    4. **Rise Interactive:**
       - Specializes in attracting ecommerce customers with services like Amazon marketing and multi-channel strategy optimization.
    
    5. **WebFX:**
       - Renowned for local SEO services and accommodating various budget sizes.
    
    6. **DigitalMarketing.com:**
       - Offers a unique tool for beta testing marketing campaigns and PR management services.
    
    7. **Mad Fish Digital:**
       - Works with socially conscious brands and nonprofits, offering a wide range of digital marketing services.
    
    8. **Split Reef:**
       - Focuses on app marketing and law firm marketing, offering SEO, CRO, and reputation management services.
    
    9. **Max Audience:**
       - Specializes in marketing automation solutions, helping businesses set up lead generation and CRM systems.
    
    10. **Location3:**
        - Best for franchise brands, providing local listings management and franchise development services.
    
    **Conclusion:**
    The digital marketing landscape in 2023 is characterized by technological advancements, increased focus on DEI, and the influence of Gen Z. Competitors in the industry are leveraging these trends to offer specialized and comprehensive services. The key to success lies in adapting to new technologies, maintaining transparency in data practices, and engaging diverse audiences effectively.
    
    [00m
    [1m[95m [2024-11-01 14:19:00][DEBUG]: == Working Agent: Chief Marketing Strategist[00m
    [1m[95m [2024-11-01 14:19:00][INFO]: == Starting Task: Synthesize insights from product analysis to develop impactful marketing strategies.
    [00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mThought: To develop strategic marketing recommendations based on product insights and competitor landscape, it's essential to gather detailed information about the current digital marketing trends and the competitive landscape. I already have a detailed report on the trends and competitor insights for 2023. I will now synthesize these insights to formulate comprehensive marketing strategies.
    
    Thought: I have all the necessary information to synthesize insights and develop strategic marketing recommendations.
    
    Final Answer:
    
    **Strategic Recommendations for Marketing Based on Product Insights and Competitor Landscape**
    
    **Digital Marketing Strategy Recommendations:**
    
    1. **Leverage Short-form Video Content:**
       - Integrate short-form video content into marketing strategies by utilizing platforms like Instagram Reels and Twitter's vertical videos. Experiment with partnerships or acquisitions with TikTok-like platforms to expand reach and engagement.
    
    2. **Embrace AI Content Generation:**
       - Implement AI tools for content creation to enhance personalization and efficiency in marketing campaigns. Focus on combining AI-generated content with human expertise to maintain quality and relevance.
    
    3. **Enhance Diversity, Equity, and Inclusion (DEI) Efforts:**
       - Prioritize DEI in marketing strategies by incorporating diverse voices and perspectives. Use data analytics to track diversity metrics and measure progress to ensure accountability and transparency.
    
    4. **Utilize Automation in Pay-Per-Click (PPC) Campaigns:**
       - Adopt Google Ads automation features, such as Performance Max campaigns, to optimize PPC advertising efforts. Monitor and adjust campaigns based on performance data to maximize efficiency.
    
    5. **Target Gen Z with Authentic and Transparent Marketing:**
       - Focus on authenticity, transparency, and accessibility to engage Gen Z audiences. Incorporate user-generated content and highlight environmental transparency in marketing messages.
    
    6. **Implement Privacy-friendly Data Practices:**
       - Adopt transparent data practices and privacy-friendly tracking technologies to build trust with consumers. Balance data collection needs with consumer privacy concerns.
    
    7. **Adopt Supply Chain Automation Technologies:**
       - Implement supply chain automation technologies to enhance efficiency and manage risks. Develop prevention and contingency plans to ensure business continuity.
    
    **Competitor Landscape Analysis and Recommendations:**
    
    1. **Reach Digital:**
       - Focus on small businesses by offering no-contract services like SEO, Facebook Lead Gen, and Google Ads. Highlight flexibility and cost-effectiveness to attract small business clients.
    
    2. **NP Digital:**
       - Emphasize data-driven marketing solutions and tools like Ubersuggest for SEO and keyword research. Position as a leader in global digital marketing with a strong focus on analytics.
    
    3. **97th Floor:**
       - Target enterprise B2B brands by offering comprehensive services across SEO, paid media, creative, and automation. Leverage the Palomar software suite to optimize marketing strategies.
    
    4. **Rise Interactive:**
       - Specialize in ecommerce customer attraction with services like Amazon marketing and multi-channel strategy optimization. Highlight expertise in ecommerce strategies to attract clients.
    
    5. **WebFX:**
       - Focus on local SEO services and cater to various budget sizes. Position as a versatile agency that can accommodate different client needs.
    
    6. **DigitalMarketing.com:**
       - Offer beta testing marketing campaigns and PR management services. Highlight unique tools and capabilities to attract innovative clients.
    
    7. **Mad Fish Digital:**
       - Work with socially conscious brands and nonprofits. Emphasize a commitment to social responsibility and a wide range of digital marketing services.
    
    8. **Split Reef:**
       - Target app marketing and law firm marketing by offering SEO, CRO, and reputation management services. Position as a specialized agency for niche markets.
    
    9. **Max Audience:**
       - Specialize in marketing automation solutions, including lead generation and CRM systems. Highlight automation expertise to attract clients seeking efficiency.
    
    10. **Location3:**
        - Focus on franchise brands by providing local listings management and franchise development services. Position as an expert in franchise marketing solutions.
    
    **Conclusion:**
    In 2023, success in digital marketing requires adapting to technological advancements, maintaining transparency in data practices, and engaging diverse audiences effectively. By leveraging current trends and analyzing competitor strategies, businesses can craft impactful marketing strategies that resonate with their target audiences and drive success.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-01 14:19:17][DEBUG]: == [Chief Marketing Strategist] Task output: **Strategic Recommendations for Marketing Based on Product Insights and Competitor Landscape**
    
    **Digital Marketing Strategy Recommendations:**
    
    1. **Leverage Short-form Video Content:**
       - Integrate short-form video content into marketing strategies by utilizing platforms like Instagram Reels and Twitter's vertical videos. Experiment with partnerships or acquisitions with TikTok-like platforms to expand reach and engagement.
    
    2. **Embrace AI Content Generation:**
       - Implement AI tools for content creation to enhance personalization and efficiency in marketing campaigns. Focus on combining AI-generated content with human expertise to maintain quality and relevance.
    
    3. **Enhance Diversity, Equity, and Inclusion (DEI) Efforts:**
       - Prioritize DEI in marketing strategies by incorporating diverse voices and perspectives. Use data analytics to track diversity metrics and measure progress to ensure accountability and transparency.
    
    4. **Utilize Automation in Pay-Per-Click (PPC) Campaigns:**
       - Adopt Google Ads automation features, such as Performance Max campaigns, to optimize PPC advertising efforts. Monitor and adjust campaigns based on performance data to maximize efficiency.
    
    5. **Target Gen Z with Authentic and Transparent Marketing:**
       - Focus on authenticity, transparency, and accessibility to engage Gen Z audiences. Incorporate user-generated content and highlight environmental transparency in marketing messages.
    
    6. **Implement Privacy-friendly Data Practices:**
       - Adopt transparent data practices and privacy-friendly tracking technologies to build trust with consumers. Balance data collection needs with consumer privacy concerns.
    
    7. **Adopt Supply Chain Automation Technologies:**
       - Implement supply chain automation technologies to enhance efficiency and manage risks. Develop prevention and contingency plans to ensure business continuity.
    
    **Competitor Landscape Analysis and Recommendations:**
    
    1. **Reach Digital:**
       - Focus on small businesses by offering no-contract services like SEO, Facebook Lead Gen, and Google Ads. Highlight flexibility and cost-effectiveness to attract small business clients.
    
    2. **NP Digital:**
       - Emphasize data-driven marketing solutions and tools like Ubersuggest for SEO and keyword research. Position as a leader in global digital marketing with a strong focus on analytics.
    
    3. **97th Floor:**
       - Target enterprise B2B brands by offering comprehensive services across SEO, paid media, creative, and automation. Leverage the Palomar software suite to optimize marketing strategies.
    
    4. **Rise Interactive:**
       - Specialize in ecommerce customer attraction with services like Amazon marketing and multi-channel strategy optimization. Highlight expertise in ecommerce strategies to attract clients.
    
    5. **WebFX:**
       - Focus on local SEO services and cater to various budget sizes. Position as a versatile agency that can accommodate different client needs.
    
    6. **DigitalMarketing.com:**
       - Offer beta testing marketing campaigns and PR management services. Highlight unique tools and capabilities to attract innovative clients.
    
    7. **Mad Fish Digital:**
       - Work with socially conscious brands and nonprofits. Emphasize a commitment to social responsibility and a wide range of digital marketing services.
    
    8. **Split Reef:**
       - Target app marketing and law firm marketing by offering SEO, CRO, and reputation management services. Position as a specialized agency for niche markets.
    
    9. **Max Audience:**
       - Specialize in marketing automation solutions, including lead generation and CRM systems. Highlight automation expertise to attract clients seeking efficiency.
    
    10. **Location3:**
        - Focus on franchise brands by providing local listings management and franchise development services. Position as an expert in franchise marketing solutions.
    
    **Conclusion:**
    In 2023, success in digital marketing requires adapting to technological advancements, maintaining transparency in data practices, and engaging diverse audiences effectively. By leveraging current trends and analyzing competitor strategies, businesses can craft impactful marketing strategies that resonate with their target audiences and drive success.
    
    [00m
    [1m[95m [2024-11-01 14:19:17][DEBUG]: == Working Agent: Creative Content Creator[00m
    [1m[95m [2024-11-01 14:19:17][INFO]: == Starting Task: Develop social media content that aligns with the marketing strategy and captures the target audience’s attention.
    [00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mTo develop compelling social media content, particularly for Instagram, I need to focus on the strategic recommendations provided, especially those concerning short-form video content, AI content generation, and targeting Gen Z with authenticity. I will start by asking the Chief Marketing Strategist for insights on how we can best incorporate these recommendations into our Instagram campaign strategy. This will ensure that the content aligns with the overall marketing strategy and effectively captures the target audience's attention.
    
    Action: Ask question to coworker
    Action Input: {
      "coworker": "Chief Marketing Strategist",
      "question": "Given our strategic recommendations, how should we incorporate short-form video content, AI content generation, and authentic marketing to target Gen Z on Instagram? We aim to align with the strategy that focuses on leveraging Instagram Reels, embracing AI for personalization, and engaging Gen Z with authentic, transparent content. Your insights will help shape our social media campaign effectively."
    }
    [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mTo provide a comprehensive strategy that incorporates short-form video content, AI content generation, and authentic marketing targeting Gen Z on Instagram, I need to gather insights on current trends and successful strategies in these areas. I will start by searching for the latest information on Instagram Reels, AI personalization, and authentic marketing targeted at Gen Z.
    
    Action: InternetSearchTool
    Action Input: {"query": "Instagram Reels strategy Gen Z AI personalization authentic marketing 2023"}
    [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    [{'title': 'Tap into 2023 trends on Instagram to help drive results', 'href': 'https://business.instagram.com/blog/2023-instagram-trend-report', 'body': 'Instagram has revealed its 2023 Trend Report, your Gen Z guide to cultural trends across the platform. From the financial renaissance to historic movements in politics, the survey features topics, issues and creators at the forefront of culture. 1. At the heart, Gen Z consumers are seeking community and connection more than ever.'}, {'title': 'Marketing to Gen Z: Do it Right with AI-Powered Social Media Strategy', 'href': 'https://www.perfectcorp.com/business/blog/general/marketing-to-gen-z-do-it-right-with-ai-powered-social-media-strategy', 'body': "Understanding Gen Z Preferences in Marketing. Gen Z, born between the mid-1990s and mid-2010s, has unique characteristics and values that shape their consumer behavior. They are known for their digital nativism, social consciousness, and desire for authentic, personalized experiences. Technology is at the core of Gen Z's daily lives."}, {'title': 'The 2023 Instagram Trend Report', 'href': 'https://about.instagram.com/blog/announcements/instagram-trends-2023', 'body': 'The 2023 Trend Report was produced alongside a sweeping study of Gen Z users across Instagram. Created in partnership with WGSN* during October 2022, Instagram measured the topics, issues, and trends that matter most to teens across the platform. Reporter Darian Symoné Harvin (@darian) also sat down with a handful of creators to talk about ...'}, {'title': 'Gen Z Stats & Trends: How to Market to Gen Z in 2025 - Later', 'href': 'https://later.com/blog/gen-z-trends/', 'body': 'Gen Z, born between 1997 and 2021, is a generation who have grown up with technology and social media. According to McKinsey, Gen Z will account for 27% of the global workforce by 2025, significantly increasing their economic and cultural influence and making them more powerful than ever. To reach them, marketers need to understand their unique ...'}, {'title': '5 Gen Z Marketing Strategies To Implement in 2024 | Later', 'href': 'https://later.com/blog/gen-z-marketing/', 'body': "Building a digital community is integral to your Gen Z marketing strategy. According to surveys, Gen Z is the loneliest generation in America, so they're actively looking for ways to engage and connect with like-minded individuals. Brands can help facilitate authentic connection and conversation, but it goes beyond working with mega-influencers."}]
    [00m
    [32;1m[1;3mThought: I have found several resources that discuss Instagram trends, AI personalization, and authentic marketing strategies for Gen Z. To provide a detailed and accurate strategic recommendation, I will read the content from a few selected sources to understand the insights they offer. I will start with the Instagram 2023 Trend Report as it seems to be directly relevant.
    
    Action: Read website content
    Action Input: {"website_url": "https://business.instagram.com/blog/2023-instagram-trend-report"}
    [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    ErrorSorry, something went wrong.We're working on getting this fixed as soon as we can.Go back Meta © 2024 · Help
    [00m
    [32;1m[1;3mThought: The initial attempt to access the Instagram Trend Report was unsuccessful. I will try another source that discusses marketing strategies for Gen Z, specifically using AI and authenticity on social media platforms like Instagram.
    
    Action: Read website content
    Action Input: {"website_url": "https://www.perfectcorp.com/business/blog/general/marketing-to-gen-z-do-it-right-with-ai-powered-social-media-strategy"}
    [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    Marketing to Gen Z: Do it Right with AI-Powered Social Media Strategy | PERFECTYou need to enable JavaScript to run this app.
    [00m
    [32;1m[1;3mThought: Since my previous attempts to access detailed content from certain sources were unsuccessful, I should try reading content from another relevant source that may provide insights on Gen Z marketing strategies using AI and authenticity.
    
    Action: Read website content
    Action Input: {"website_url": "https://later.com/blog/gen-z-marketing/"}
    [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    5 Gen Z Marketing Strategies To Implement in 2024 | LaterSkip to main contentSocial Media ManagementInfluencer MarketingLink in BioSocial Media ManagementSolutionsSocial Media ToolsWhat We DoEvery social tool in one appPublish & ScheduleNever forget to post againSocial ListeningCustomer insights, ready in secondsSocial Media AnalyticsGet impactful data & reportsContent CalendarPlan & visualize across platformsContent CreationCreate better content fasterCaption WriterGenerate Instagram captions in seconds Later Mobile AppSchedule & post on the goLater Social UsersSocial Media ManagersWork faster & get better resultsAgenciesManage social campaigns for clientsCreatorsFind brand collabs and get paidSmall Business OwnersPost consistently & drive salesNew Later Social FeatureAI Caption WriterWrite Instagram captions faster (opens a new tab)ProductSocial PlatformsPlatforms OverviewPost on every social channelInstagramPlan your ideal feedInstagram submenuVisual PlannerPreview & curate your feedStories SchedulingCreate, edit, & scheduleCaption WriterGenerate Instagram captionsBest Time to PostPost at the perfect timeAnalyticsGet the data that mattersAuto PublishJust set it & forget itHashtag SuggestionsGet personalized hashtagsReelsSchedule Reels in advanceDesktop PostingPost on Instagram from your desktopFacebookBuild your brand easilyTikTokPost more & grow your brandLinkedInGet down to businessPinterestCreate pin-worthy postsX (Twitter)Tweet, tweet, repeatYouTubeStart sharing your ShortsIntro to Later SocialNew to Later? Watch this video!Later makes social media easy (opens a new tab)PricingResourcesMarketing TrainingMarketing ResourcesFree guides to help you growCase StudiesLearn from Later customersBeyond Influence PodcastListen & learn from real influencersSocial Media GlossaryAll the terms from A-ZInstagram Captions LibraryFind captions for every topicEmoji DictionaryEmoji meanings explainedHow to Use LaterGet Later CertifiedBecome a Later pro for free (opens a new tab)Product TrainingGet the most out of LaterLater WebinarsLive product demos and Q&AsHelp CenterSearch & find answers (opens a new tab)CommunityConnect, learn, & engage (opens a new tab)New Plan PickerFind Your Later Social PlanTake the guesswork out with our 1-minute quizBlogLog InCreate Free AccountContinue asCreate Free AccountLater BlogLatestPlatformsInstagramTikTokPinterestLinkedInFacebookTwitterTopicsSocial Media NewsSocial CommerceDesignVideo StrategyAnalytics & ReportingInfluencer MarketingLater Tools & FeaturesBy RoleInfluencers & CreatorsSmall Businesses & BrandsSocial Media ManagersFree ResourcesSocial Media Reporting Template (opens a new tab)Influencer Marketing Benchmarks in 2024Social Media Holidays Calendar (opens a new tab)Instagram Industry Benchmarks Report (opens a new tab)VideosPodcastSearchMenuMenuCopied URL to clipboard!Social Media Marketing Blog5 Gen Z Marketing Strategies To Implement in 2024By Monique ThomasUpdated on December 12, 20236 minute readAccording to a Gen Zer 😎Published December 12, 2023ShareContentSide Nav Table of ContentsIntroductionWho is Gen Z?5 Gen Z Marketing StrategiesShareGen Z marketing requires more than simply selling a product. Born between 1996 and 2010, this generation is value-conscious and drawn to brands that are authentic.They’re creative, bold, and the creators of many trends on TikTok, Instagram Reels, and other platforms. And they require a slightly different approach than their millennial counterparts. So how do you reach and connect with the Gen Z market in 2024? We're covering it all, below.Never Miss a Trend AgainJoin over 1 million marketers to get social news, trends, and tips right to your inbox!Email AddressWho is Gen Z?Young, tech savvy, and socially-minded, Gen Z is the most racially and ethnically diverse generation with a projected spending power of over $360B. Having grown up with smartphones, the internet, and social media, these digital natives are more likely to buy from brands that have established clear values, are inclusive, and have a strong online community. So, if you haven’t started thinking about this younger generation, 2024 is the perfect time to start. With a few strategies tailored specifically to Gen Z, you can tap into their audience and create content they’ll engage with — without alienating your current customers. Ready to create a solid Gen Z marketing strategy? Learn from Gen Z expert, Larry Milstein, in this free video workshop (or keep reading, below): 5 Gen Z Marketing Strategies for 2024In order to connect with this generation of digital experts, you'll need to keep these five strategies in mind:Establish Clear Values & MissionBe Transparent & AccountableEstablish Your Brand’s PersonalityBe EntertainingBuild a Community#1: Establish Clear Values & MissionBefore marketing to Gen Z, it’s important to establish your brand's values and mission. Why?“Gen Zers are much more inclined to vote with our dollars, and believe a brand’s values are a reflection of our own,” explains Gen Z expert (and Gen Zer!) Larry Milstein. Some topics that matter to Gen Z include: LGBTQ+ rights: 60% of Gen Zers think same-sex couples should be able to adopt childrenDiversity: 60% of Gen Zers say increased racial and ethnic diversity is good for societySocial responsibility: 70% of Gen Zers try to purchase from companies they consider ethical“We’re 3x more likely than older generations to believe a company has a role in improving society,” adds Larry. Take underwear brand Huha for example:Their content not only helps to normalize stretch marks and body hair, but they are vocal advocates for body positivity and dismantling myths about sexual health.The takeaway? Establishing your brand’s values and communicating them effectively is key to connecting with Gen Z.Manage your social media content like a pro: schedule your posts, videos, and stories all in one place with Later. Get started today! Hit your social goals faster with more features — try them on any paid plan.Explore Plans#2: Be Transparent & AccountableThe next step in your Gen Z marketing strategy is to be transparent and publicly take accountability for any missteps. According to Larry, Gen Z has no qualms about doing their research. They’ll do a deep dive into a brand’s website, scroll through their social media accounts, and read comments and reviews. “Brand trust is only second to price in terms of determining what brands we’re going to support. You’ve seen brands trip up because they’ve not upheld the standards that they’ve been communicating,” explains Larry. Company culture matters to Gen Z, and your brand should hold the same values externally and internally.But it’s not enough to solely cast and work with diverse creators and influencers. Larry adds: “It needs to be built into the fabric of the brand.”#3: Establish Your Brand’s PersonalityIn order to engage and connect with Gen Z, say goodbye to the millennial-focused aesthetic of perfectly curated content. No more sleek and minimal imagery — Gen Zers want brands that are bold, have a strong voice, and a personality. Starface sells pimple-protectors and their website and social media content is full of bright yellow, smiley faces, and stars:They’re able to promote their products in a light-hearted and silly way, and the strategy has seen massive success. They have over 2M followers on TikTok with videos full of skincare routines, fun transformations, and catchy music. For more established brands, Larry points to a Crocs and KFC collaboration as an example of targeting a younger generation:“Love it or hate it, what they’ve done well is tap into an irony that Gen Zers gravitate towards, especially in a world where there’s massive uncertainty and unrest.” The takeaway? The zany and weird can work too — these Crocs sold out.Plan, manage & analyze TikTok videos all in one spot with Later!Create Account#4: Be EntertainingGen Zers have an uncanny ability to filter content. With a short attention span, “you have around 8 seconds to essentially tell us why we should pay attention before we move on.”Larry says makeup brand Fenty Beauty has done this effectively with both their TikTok and Instagram Reels content:Creating fun and quick tutorials featuring various beauty influencers lets Fenty Beauty spotlight their products in an easily digestible way.TIP: If entertaining isn't your style, consider going the more educational route — but keep your language conversational. #5: Build a CommunityBuilding a digital community is integral to your Gen Z marketing strategy.According to surveys, Gen Z is the loneliest generation in America, so they’re actively looking for ways to engage and connect with like-minded individuals. Brands can help facilitate authentic connection and conversation, but it goes beyond working with mega-influencers. The answer? Seek real people who have interesting platforms and embody your brand’s values. “It could be a creative, an artist, a thought-leader, or an activist — find people you believe can serve as ambassadors and tap into their networks in a way that feels less transactional and more authentic,” says Larry. Another way to build a community is to either solicit feedback from Gen Zers during product development or to celebrate those who’ve been loyal to your brand for years. So, there you have it — Gen Z marketing is a great way to connect with a new type of consumer: one that is value-conscious, loves bold personalities, and seeking community. Time to ideate. ICYMI: With Later's social media management tools, you can plan, manage, and analyze your posts — all in one spot. Create an account today. Plan, manage, & analyze posts for every social channel in one spot with Later.Sign UpNever Miss a Trend AgainJoin over 1 million marketers to get social news, trends, and tips right to your inbox!Email AddressShareAbout the AuthorMonique ThomasContent Marketing ManagerMonique is a Content Marketing Manager with over 7 years of experience working in communications and social media. You can usually find her sipping matcha or adding another book to her TBR list.Plan, schedule, and automatically publish your social media posts with Later.Get StartedRelated ArticlesCategory Social Media Marketing Blog10 min readGen Z vs Millennial: What Marketers Need to Know on Social MediaBy Amanda DemekuCategory TikTok Blogs & Resources17 min read28 Creative Brands on TikTok to Inspire Your FeedBy Erica SalvalaggioCategory Social Media Blogs & Tips for Small Businesses5 min readWhy Pop Culture Should Be Part of Your Social Media StrategyBy Jessica WorbThe best social media marketing app, influencer marketing management platform & Link in Bio tool.Later SocialLater Social submenuPlatformsInstagramFacebookTikTokLinkedInPinterestX (Twitter)YouTubeAll PlatformsSolutionsSocial Media ToolsPublish & ScheduleSocial ListeningSocial Media AnalyticsContent CalendarContent CreationAI Caption WriterLater Mobile AppInstagram FeaturesVisual PlannerStories SchedulingCaption WriterBest Time to PostAnalyticsAuto PublishHashtag SuggestionsReels SchedulingDesktop PostingLater UsersSocial Media ManagersAgenciesCreatorsSmall Business OwnersPricingSee PlansLater InfluenceLater Influence submenuSolutionsLater Influencer ProgramAffiliate MarketingReputation ManagementToolsInfluencer DatabaseFind InfluencersInfluencer OutreachInfluencer CampaignsManage InfluencersPay InfluencersSyndicate ContentInfluencer AnalyticsEMV + ROIShopify IntegrationAll ToolsInfluencersMicro InfluencersBrand AmbassadorsUGC CreatorsYouTubersFashion & BeautyServicesIn-House Influencer Marketing ServicesSocial Media Content Creation ServicesInfluencer Marketing for AgenciesUse CasesCPGRetailFood & BeverageE-CommerceDemoBook a DemoLink in BioLink in Bio submenuSolutionsLink in Bio by LaterCompanyAboutCareersNewsroomResourcesBlogMarketing ResourcesCase StudiesBeyond Influence PodcastLearn with LaterGet Later CertifiedHelp CenterCommunityDownloadApp StoreGoogle PlayFollow UsJoin Our NewsletterPartnershipsBecome a PartnerPinterest Marketing Partners© 2024 Later. All Rights Reserved.SitemapTermsPrivacy PolicyCookie Settings
    [00m
    [32;1m[1;3mThought: I now have enough information to provide a comprehensive strategy for incorporating short-form video content, AI content generation, and authentic marketing to target Gen Z on Instagram. The insights from the Later article outline key strategies that align with our goals.
    
    Final Answer: To effectively incorporate short-form video content, AI content generation, and authentic marketing to target Gen Z on Instagram, we should focus on the following strategies:
    
    1. **Establish Clear Values & Mission**: Clearly communicate your brand's values and mission, as Gen Z seeks brands whose values mirror their own. Highlight topics important to them, such as diversity and social responsibility.
    
    2. **Be Transparent & Accountable**: Build trust by being transparent and accountable. Gen Z values brand trust and will research to ensure a brand's values align with their own. Ensure your brand's culture reflects externally and internally.
    
    3. **Establish Your Brand's Personality**: Move away from perfectly curated content and embrace a bold, distinct brand personality. Use humor, creativity, and relatable content to engage Gen Z.
    
    4. **Be Entertaining**: Capture Gen Z's attention quickly with entertaining and engaging content. Use Instagram Reels for fun, quick tutorials or entertaining narratives that promote your products.
    
    5. **Build a Community**: Foster a sense of community by engaging with your audience authentically. Collaborate with influencers and creators who align with your brand values, and engage in meaningful conversations with your community.
    
    6. **Utilize AI for Personalization**: Leverage AI to create personalized experiences and content for your audience. Use AI tools to analyze engagement data and tailor content to meet the preferences and interests of your Gen Z audience.
    
    By implementing these strategies, we can create a social media campaign on Instagram that resonates with Gen Z, leveraging short-form video content, AI-driven personalization, and authentic engagement.[0m
    
    [1m> Finished chain.[0m
    [95m 
    
    To effectively incorporate short-form video content, AI content generation, and authentic marketing to target Gen Z on Instagram, we should focus on the following strategies:
    
    1. **Establish Clear Values & Mission**: Clearly communicate your brand's values and mission, as Gen Z seeks brands whose values mirror their own. Highlight topics important to them, such as diversity and social responsibility.
    
    2. **Be Transparent & Accountable**: Build trust by being transparent and accountable. Gen Z values brand trust and will research to ensure a brand's values align with their own. Ensure your brand's culture reflects externally and internally.
    
    3. **Establish Your Brand's Personality**: Move away from perfectly curated content and embrace a bold, distinct brand personality. Use humor, creativity, and relatable content to engage Gen Z.
    
    4. **Be Entertaining**: Capture Gen Z's attention quickly with entertaining and engaging content. Use Instagram Reels for fun, quick tutorials or entertaining narratives that promote your products.
    
    5. **Build a Community**: Foster a sense of community by engaging with your audience authentically. Collaborate with influencers and creators who align with your brand values, and engage in meaningful conversations with your community.
    
    6. **Utilize AI for Personalization**: Leverage AI to create personalized experiences and content for your audience. Use AI tools to analyze engagement data and tailor content to meet the preferences and interests of your Gen Z audience.
    
    By implementing these strategies, we can create a social media campaign on Instagram that resonates with Gen Z, leveraging short-form video content, AI-driven personalization, and authentic engagement.
    [00m
    [32;1m[1;3mThought: With the insights from the Chief Marketing Strategist, I can now create engaging social media content for Instagram that aligns with the marketing strategy and effectively targets Gen Z. The focus will be on leveraging short-form video content, AI personalization, and authentic engagement.
    
    Final Answer:
    
    **Instagram Campaign Content**
    
    *Theme: Embrace Your Bold Self*
    
    **Instagram Reels & Posts:**
    
    1. **Reel 1: "Values Unleashed"**
       - Visual: A vibrant montage of diverse individuals expressing what they stand for.
       - Caption: "Our values define us. What do yours say about you? 🌟 #EmbraceYourBoldSelf"
       - Call to Action: "Share your values with us in the comments!"
    
    2. **Reel 2: "Behind the Brand"**
       - Visual: A day-in-the-life at the company, showcasing transparency and accountability.
       - Caption: "We believe in walking the talk. Take a peek behind our curtains! 👀 #TransparentTogether"
       - Call to Action: "Got questions? Drop them below!"
    
    3. **Reel 3: "Fun with AI: Personalized Just for You"**
       - Visual: A playful animation of AI creating personalized content for users.
       - Caption: "Not just any content—it's YOUR content. 💡 #AIPersonalized"
       - Call to Action: "Tell us your favorite type of content, and we'll make it happen!"
    
    **Instagram Stories:**
    
    1. **Story 1: "Meet the Team"**
       - Visual: Short clips introducing team members with fun facts.
       - Interactive Element: Poll - "Which team member shares your passion for sustainability?"
    
    2. **Story 2: "Community Shoutout"**
       - Visual: Featuring content from followers who use the hashtag #EmbraceYourBoldSelf.
       - Interactive Element: Question Sticker - "How do you embrace your bold self?"
    
    3. **Story 3: "Quick Tips & Tricks"**
       - Visual: Quick tutorials or tips related to the product, delivered in a light-hearted manner.
       - Interactive Element: Quiz - "Did you know this product hack?"
    
    **Post: "Influencer Collaboration"**
    - Visual: A carousel featuring clips and photos from a popular influencer using the product in everyday life.
    - Caption: "Join [Influencer Name] as they embrace their bold self with [Product]! 🎉 #BoldCollab"
    - Call to Action: "See how [Influencer Name] uses [Product] in their day-to-day. Swipe to learn more!"
    
    By implementing these content pieces, we aim to successfully capture the attention of Gen Z on Instagram, enhance brand engagement, and foster a community centered around shared values and authenticity.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-01 14:19:58][DEBUG]: == [Creative Content Creator] Task output: **Instagram Campaign Content**
    
    *Theme: Embrace Your Bold Self*
    
    **Instagram Reels & Posts:**
    
    1. **Reel 1: "Values Unleashed"**
       - Visual: A vibrant montage of diverse individuals expressing what they stand for.
       - Caption: "Our values define us. What do yours say about you? 🌟 #EmbraceYourBoldSelf"
       - Call to Action: "Share your values with us in the comments!"
    
    2. **Reel 2: "Behind the Brand"**
       - Visual: A day-in-the-life at the company, showcasing transparency and accountability.
       - Caption: "We believe in walking the talk. Take a peek behind our curtains! 👀 #TransparentTogether"
       - Call to Action: "Got questions? Drop them below!"
    
    3. **Reel 3: "Fun with AI: Personalized Just for You"**
       - Visual: A playful animation of AI creating personalized content for users.
       - Caption: "Not just any content—it's YOUR content. 💡 #AIPersonalized"
       - Call to Action: "Tell us your favorite type of content, and we'll make it happen!"
    
    **Instagram Stories:**
    
    1. **Story 1: "Meet the Team"**
       - Visual: Short clips introducing team members with fun facts.
       - Interactive Element: Poll - "Which team member shares your passion for sustainability?"
    
    2. **Story 2: "Community Shoutout"**
       - Visual: Featuring content from followers who use the hashtag #EmbraceYourBoldSelf.
       - Interactive Element: Question Sticker - "How do you embrace your bold self?"
    
    3. **Story 3: "Quick Tips & Tricks"**
       - Visual: Quick tutorials or tips related to the product, delivered in a light-hearted manner.
       - Interactive Element: Quiz - "Did you know this product hack?"
    
    **Post: "Influencer Collaboration"**
    - Visual: A carousel featuring clips and photos from a popular influencer using the product in everyday life.
    - Caption: "Join [Influencer Name] as they embrace their bold self with [Product]! 🎉 #BoldCollab"
    - Call to Action: "See how [Influencer Name] uses [Product] in their day-to-day. Swipe to learn more!"
    
    By implementing these content pieces, we aim to successfully capture the attention of Gen Z on Instagram, enhance brand engagement, and foster a community centered around shared values and authenticity.
    
    [00m
    [1m[95m [2024-11-01 14:19:58][DEBUG]: == Working Agent: Senior Photographer[00m
    [1m[95m [2024-11-01 14:19:58][INFO]: == Starting Task: Produce visually compelling photographs that align with the campaign’s objectives and enhance the ad’s impact.
    [00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mTo create visually compelling photographs that align with the campaign's objectives of capturing audience attention and engagement on Instagram, I need to gather information on current trends and strategies for Instagram ads that resonate with Gen Z. This will help ensure that the photographs I take are tailored to effectively convey emotion and engage this specific audience.
    
    Action: InternetSearchTool
    Action Input: {"query": "current Instagram ad trends for Gen Z 2023"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    [{'title': 'Tap into 2023 trends on Instagram to help drive results', 'href': 'https://business.instagram.com/blog/2023-instagram-trend-report', 'body': 'The 2023 Instagram trend report is your guide to the trends that matter most to Gen Z. Learn what brands can do to elevate their sustainability, authenticity and global culture. Goals Create a business account Create successful posts Advertise on Instagram Boost my posts Set up a shop on Instagram Inspire people to take action Optimize your ...'}, {'title': 'The 2023 Instagram Trend Report', 'href': 'https://about.instagram.com/blog/announcements/instagram-trends-2023', 'body': 'Scroll on to explore our 2023 Instagram Trend Report, and may the trends (and trendsetters) on the pages that follow inspire you all year long. Download the report *In October 2022, WGSN conducted a survey of 1,200 Gen Z social media users aged 16-24 across the United States.'}, {'title': '2024 Instagram Trend Talk', 'href': 'https://about.instagram.com/blog/announcements/instagram-trend-talk/', 'body': "Today, Instagram is unveiling the 2024 Instagram Trend Talk, Gen Z's take on trends that will drive global culture in the year ahead.. 2023 has been a year on socials to remember. From Barbie-core painting the world pink, to the rise of the 'girl dinner', viral moments and trends on Instagram and beyond have been driving global culture and influencing Gen Z's next move."}, {'title': "Gen Z Is Shaping Our Culture and Instagram's 2023 Trend Report ... - INTO", 'href': 'https://www.intomore.com/culture/gen-z-shaping-culture-instagrams-2023-trend-report-proves/', 'body': 'Instagram partnered with WGSN to survey 1200 Gen Z social media users on topics, issues, and behaviors that drive culture on the platform. The report covers fashion, activism, beauty, music, dating, and more, but does not mention screen time.'}, {'title': 'Instagram Offers Its Predictions on Generation Z Trends for 2023 - Adweek', 'href': 'https://www.adweek.com/media/instagram-offers-its-predictions-on-generation-z-trends-for-2023/', 'body': 'Return of the Rave and Rise of Global Beats: Instagram and WGSN found that 68% of Gen-Z social media users plan to attend or want to attend a rave in 2023, while more than one-half intend to non ...'}]
    [00m
    [32;1m[1;3mThought: I have found relevant sources containing insights on Instagram trends for Gen Z in 2023. I will now read through one of the detailed reports to gain a deeper understanding of these trends, which will inform my photography strategy for the campaign.
    
    Action: Read website content
    Action Input: {"website_url": "https://business.instagram.com/blog/2023-instagram-trend-report"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    ErrorSorry, something went wrong.We're working on getting this fixed as soon as we can.Go back Meta © 2024 · Help
    [00m
    [32;1m[1;3mThought: Since the first source was not accessible, I will try another relevant source that might provide the necessary insights on Instagram trends for Gen Z in 2023.
    
    Action: Read website content
    Action Input: {"website_url": "https://about.instagram.com/blog/announcements/instagram-trends-2023"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    ErrorSorry, something went wrong.We're working on getting this fixed as soon as we can.Go back Meta © 2024 · Help
    [00m
    [32;1m[1;3mThought: Since the initial websites I attempted to access are not available, I should try another source from the search results to gain insights on Instagram trends for Gen Z in 2023.
    
    Action: Read website content
    Action Input: {"website_url": "https://www.intomore.com/culture/gen-z-shaping-culture-instagrams-2023-trend-report-proves/"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    Attention Required! | Cloudflare
    Please enable cookies.
    Sorry, you have been blocked
    You are unable to access intomore.com
    Why have I been blocked?
    This website is using a security service to protect itself from online attacks. The action you just performed triggered the security solution. There are several actions that could trigger this block including submitting a certain word or phrase, a SQL command or malformed data.
    What can I do to resolve this?
    You can email the site owner to let them know you were blocked. Please include what you were doing when this page came up and the Cloudflare Ray ID found at the bottom of this page.
    Cloudflare Ray ID: 8dbc8647ea4c3920
    •
     Your IP:
     Click to reveal
    34.86.217.198
    •
    Performance & security by Cloudflare
    [00m
    [32;1m[1;3mSince the previous websites were not accessible, I should try another source from the search results to gather insights on Instagram trends for Gen Z in 2023.
    
    Action: Read website content
    Action Input: {"website_url": "https://www.adweek.com/media/instagram-offers-its-predictions-on-generation-z-trends-for-2023/"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    Instagram Offers Its Predictions on Generation Z Trends for 2023
    Media Audio & Podcasting Blockchain Mediaweek Women in Sports Newsletters Platforms Publishing
    				Media
    Media Audio & Podcasting Blockchain Mediaweek Women in Sports Newsletters Platforms Publishing
     Platforms
     Instagram Offers Its Predictions on Generation Z Trends for 2023
    The platform teamed up with WGSN
    Highlights from the findings by Instagram and WGSN were broken out into 10 categoriesasbe/iStock
    By David Cohen  
    Instagram teamed up with WGSN on a study of Generation Z users to help determine the issues, topics and trends that matter to teens on its platform, and it shared the results in its 2023 Trend Report Tuesday.The platform said in the introduction to the report, “From the financial renaissance to historic trends in political participation, we survey the topics, issues and creators who’ll drive culture in the new year. At the heart of it all: community and connection.
    WORK SMARTER - LEARN, GROW AND BE INSPIRED.
    Subscribe today!
    To Read the Full Story Become an Adweek+ Subscriber
    View Subscription Options
    Already a member? Sign in
    David Cohen
    @9Number9 davidjcohen9@gmail.com 
    David Cohen is a freelance writer for ADWEEK, covering breaking news.
     Recommended articles
    Popular Now
    Here Are the Cable News Ratings for October 2024
    Watch Kellyanne and Claudia Conway’s Mother-Daughter Moment
    Havas Media’s First Shell Campaign Is Under Fire With More Than 70 Complaints for Greenwashing
    Here Are the Cable News Ratings for September 2024
    Craft a Quality AI Marketing Strategy That Goes Beyond the Hype
    Upcoming Webinars
     Create Seamless Shopping Experiences With AI Wednesday, November 6, 2024
     The Secrets to Successful Creator Partnerships Thursday, November 7, 2024
     Maximize the Impact of Your Big Game Ad Spend Tuesday, November 12, 2024
     The 2025 Consumer Trends Report Wednesday, November 13, 2024
     Create Emotional Connections with CTV Viewers Thursday, November 14, 2024
    Adweek is the leading source of news and insight serving the brand marketing ecosystem.
    About
    About UsContact UsMedia KitEditorial CalendarEvent CalendarSponsor ContentADWEEK CareersIndustry Job Openings 
    Subscriptions
    Subscription OptionsFAQsNewslettersCustomer Service 
    Events
    AwardsADWEEK EventsWebinarsOn-Demand WebinarsTrophies / Awards / SealsOn-Demand Event Videos 
    Publications
    ADWEEK Blog NetworkADWEEK Podcast NetworkRSSBack IssuesReprints / E-Prints 
    ©2024 Adweek - All Rights Reserved.
    Do not sell my personal information
    Terms of Use
    Privacy Policy
    [00m
    [32;1m[1;3mIt seems I am unable to access detailed content from the sources that might provide insights about Instagram trends for Gen Z in 2023. However, based on the search results, it is clear that the current trends emphasize sustainability, authenticity, community, and personalized content.
    
    To capture high-quality photographs for the Instagram campaign that resonate with Gen Z:
    
    1. **Emphasize Authenticity and Diversity:** Focus on capturing real and diverse individuals expressing their unique values and personalities. This aligns with the "Values Unleashed" reel, showcasing people in natural, candid moments that reflect their genuine selves.
    
    2. **Incorporate Elements of Fun and Innovation:** For the "Fun with AI" reel, create playful and dynamic visuals that highlight the innovative use of technology, perhaps using vibrant colors or dynamic compositions to convey excitement and creativity.
    
    3. **Showcase Behind-the-Scenes and Transparency:** For the "Behind the Brand" reel, capture genuine, behind-the-scenes moments that highlight the company’s transparency and commitment to authenticity, using natural lighting and unposed interactions.
    
    4. **Highlight Influencer Collaboration with Real-Life Contexts:** In the influencer collaboration post, ensure the visuals feel personal and relatable, showing the influencer using the product in everyday situations, thus creating a connection with the audience.
    
    By focusing on these elements, the photographs will effectively convey emotion, engage the audience, and align with the campaign's objectives of enhancing brand engagement and fostering a community centered around shared values and authenticity.
    
    Final Answer: To capture high-quality photographs for the Instagram campaign targeting Gen Z, focus on authenticity, diversity, and personalization. Capture real individuals expressing their values for "Values Unleashed," use vibrant and playful visuals for "Fun with AI," showcase genuine behind-the-scenes moments for "Behind the Brand," and highlight relatable, real-life uses of products in influencer collaborations. These approaches align with Gen Z's preference for authenticity, innovation, and community connection.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-01 14:20:27][DEBUG]: == [Senior Photographer] Task output: To capture high-quality photographs for the Instagram campaign targeting Gen Z, focus on authenticity, diversity, and personalization. Capture real individuals expressing their values for "Values Unleashed," use vibrant and playful visuals for "Fun with AI," showcase genuine behind-the-scenes moments for "Behind the Brand," and highlight relatable, real-life uses of products in influencer collaborations. These approaches align with Gen Z's preference for authenticity, innovation, and community connection.
    
    [00m
    [1m[95m [2024-11-01 14:20:27][DEBUG]: == Working Agent: Chief Creative Director[00m
    [1m[95m [2024-11-01 14:20:27][INFO]: == Starting Task: Evaluate content produced by team members, provide feedback, and approve or request refinements as needed.
    [00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mTo evaluate and refine the content produced by team members for the Instagram campaign targeting Gen Z, I need to gather the current content available for review. Since I need to assess the quality of the photographs, which are part of this content, I'll start by asking the Senior Photographer about the current photographs captured for the campaign. This will help ensure they align with the campaign's themes of authenticity, diversity, and personalization.
    
    Action: Ask question to coworker
    Action Input: {"question": "Can you provide the current photographs captured for the Instagram campaign targeting Gen Z? The campaign focuses on themes like 'Values Unleashed,' 'Fun with AI,' 'Behind the Brand,' and influencer collaborations. I'm looking to evaluate these to ensure they align with our themes of authenticity, diversity, and personalization.", "context": "To capture high-quality photographs for the Instagram campaign targeting Gen Z, focus on authenticity, diversity, and personalization. Capture real individuals expressing their values for 'Values Unleashed,' use vibrant and playful visuals for 'Fun with AI,' showcase genuine behind-the-scenes moments for 'Behind the Brand,' and highlight relatable, real-life uses of products in influencer collaborations. These approaches align with Gen Z's preference for authenticity, innovation, and community connection.", "coworker": "Senior Photographer"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mTo provide the current photographs captured for the Instagram campaign targeting Gen Z, I need to gather information on the themes we've been focusing on: 'Values Unleashed,' 'Fun with AI,' 'Behind the Brand,' and influencer collaborations. I'll start by searching for the latest trends and examples related to these themes to ensure they align with Gen Z's preferences for authenticity, diversity, and personalization.
    
    Action: InternetSearchTool
    Action Input: {"query": "latest photography trends for Gen Z Instagram campaigns 2023"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    [{'title': 'Tap into 2023 trends on Instagram to help drive results', 'href': 'https://business.instagram.com/blog/2023-instagram-trend-report', 'body': 'Instagram has revealed its 2023 Trend Report, your Gen Z guide to cultural trends across the platform. From the financial renaissance to historic movements in politics, the survey features topics, issues and creators at the forefront of culture. 1. At the heart, Gen Z consumers are seeking community and connection more than ever.'}, {'title': 'The 2023 Instagram Trend Report', 'href': 'https://about.instagram.com/blog/announcements/instagram-trends-2023', 'body': 'The 2023 Trend Report was produced alongside a sweeping study of Gen Z users across Instagram. Created in partnership with WGSN* during October 2022, Instagram measured the topics, issues, and trends that matter most to teens across the platform. Reporter Darian Symoné Harvin (@darian) also sat down with a handful of creators to talk about ...'}, {'title': 'Instagram Reveals Top Trends That Gen Z Will Drive in 2024', 'href': 'https://petapixel.com/2023/12/04/instagram-reveals-top-trends-that-gen-z-will-drive-in-2024/', 'body': 'Today, Instagram unveiled the "2024 Instagram Trend Talk" which shares the results of a sweeping survey looking at Generation Z trends across the U.S., U.K., Brazil, India, and South Korea.'}, {'title': "Gen Z's trend predictions for 2023: Instagram | CTV News", 'href': 'https://www.ctvnews.ca/lifestyle/here-s-what-gen-z-wants-in-2023-according-to-instagram-1.6188894', 'body': 'Instagram has published a new report highlighting the top rising trends for the coming year among Gen Z users across the platform. The report, which includes a survey of 1,200 Gen Z users across ...'}, {'title': 'New Photo Trends on Instagram: 2024 Edition - Hootsuite', 'href': 'https://blog.hootsuite.com/instagram-photo-editing-trends/', 'body': "9. Carousel panoramas. You can't post a photo on Instagram with dimensions wider than 1080 by 566 pixels… that is, unless you're sneaky about it (which, of course, we are). Carousel panoramas are single photos that line up with each other to create one long image. They're irresistibly scrollable."}]
    [00m
    [32;1m[1;3mThought: I have found some relevant links that discuss Instagram trends and Gen Z preferences. To get a more detailed understanding, I will read the content of one of the websites that specifically deals with Instagram trends for Gen Z. 
    
    Action: Read website content
    Action Input: {"website_url": "https://business.instagram.com/blog/2023-instagram-trend-report"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    ErrorSorry, something went wrong.We're working on getting this fixed as soon as we can.Go back Meta © 2024 · Help
    [00m
    [32;1m[1;3mThought: Since the previous attempt to read the website content resulted in an error, I need to select another source to gather the necessary information on Instagram trends for Gen Z.
    
    Action: Read website content
    Action Input: {"website_url": "https://about.instagram.com/blog/announcements/instagram-trends-2023"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    ErrorSorry, something went wrong.We're working on getting this fixed as soon as we can.Go back Meta © 2024 · Help
    [00m
    [32;1m[1;3m[0m[32;1m[1;3mThought: I now can give a great answer
    Final Answer: To capture high-quality photographs for the Instagram campaign targeting Gen Z, focus on authenticity, diversity, and personalization. Capture real individuals expressing their values for 'Values Unleashed,' use vibrant and playful visuals for 'Fun with AI,' showcase genuine behind-the-scenes moments for 'Behind the Brand,' and highlight relatable, real-life uses of products in influencer collaborations. These approaches align with Gen Z's preference for authenticity, innovation, and community connection.[0m
    
    [1m> Finished chain.[0m
    [95m 
    
    To capture high-quality photographs for the Instagram campaign targeting Gen Z, focus on authenticity, diversity, and personalization. Capture real individuals expressing their values for 'Values Unleashed,' use vibrant and playful visuals for 'Fun with AI,' showcase genuine behind-the-scenes moments for 'Behind the Brand,' and highlight relatable, real-life uses of products in influencer collaborations. These approaches align with Gen Z's preference for authenticity, innovation, and community connection.
    [00m
    [32;1m[1;3mThought: It seems there was a misunderstanding, as the observation did not provide the current photographs. I need to ask the Senior Photographer again, ensuring clarity in my request for the specific photographs captured for the campaign.
    
    Action: Ask question to coworker
    Action Input: {"question": "Can you provide me with the actual photographs captured for the Instagram campaign targeting Gen Z? I need to evaluate them to ensure they align with our themes of authenticity, diversity, and personalization. The campaign includes themes like 'Values Unleashed,' 'Fun with AI,' 'Behind the Brand,' and influencer collaborations.", "context": "The Instagram campaign targeting Gen Z focuses on capturing high-quality photographs that highlight authenticity, diversity, and personalization. We need images that show real individuals expressing their values, vibrant and playful visuals, genuine behind-the-scenes moments, and relatable, real-life product uses in influencer collaborations.", "coworker": "Senior Photographer"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI need to locate the actual photographs captured for the Instagram campaign targeting Gen Z. To do this, I'll need to find information about the campaign, possibly from the agency's website or other sources that may have the images or detailed descriptions.
    
    Action: InternetSearchTool
    Action Input: {"query": "Gen Z Instagram campaign Values Unleashed Fun with AI Behind the Brand influencer collaborations photographs"}
    [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    [{'title': 'Top 10 AI Influencers Making Waves on Instagram - Influencer Marketing Hub', 'href': 'https://influencermarketinghub.com/ai-influencers-instagram/', 'body': 'According to IMH\'s AI Influencer Marketing Benchmark Report, nearly half of the respondents who collaborated with AI influencers for their brand campaign reported a "very positive" experience. In addition, 52.8% of respondents believe that the versatility of AI influencers will have a major effect on the future of marketing and entertainment.'}, {'title': 'AI And Influencer Marketing: How Businesses Can Navigate The ... - Forbes', 'href': 'https://www.forbes.com/sites/esade/2024/10/30/ai-and-influencer-marketing-how-businesses-can-navigate-the-future/', 'body': 'Influencer marketing has evolved from a trend into a multimillion-dollar industry reshaping brand strategy worldwide. Now, with the rise of artificial intelligence (AI) making way for virtual and ...'}, {'title': 'The Ultimate List Of The Top 30 Gen Z Influencers For Epic ...', 'href': 'https://billo.app/gen-z-influencers/', 'body': 'TikTok: 56.2 Million Followers; Instagram: 398 Million Followers; Kylie Kenner is one of the most influential influencers and social media personalities out there. The Gen Z socialite and businesswoman has made a fortune for herself by modeling the latest fashionable clothing and taking her audience on an exploration of her luxe lifestyle.'}, {'title': 'How the Best Brand-Influencer Partnerships Reach Gen Z', 'href': 'https://hbr.org/2023/06/how-the-best-brand-influencer-partnerships-reach-gen-z', 'body': "Authenticity is among Gen Z's most important values. They feel empowered to ask and answer their own questions in a variety of social forums on any topic — from beauty to health to home ..."}, {'title': 'Five brands that rock Gen Z marketing on Instagram - Gen Z Insights', 'href': 'https://www.genzinsights.com/five-brands-that-rock-gen-z-marketing-on-instagram', 'body': "3. Everlane's use of behind-the-scenes stories. Gen Z wants marketing that's realistic and relatable, which means they love to peek behind-the-scenes of a popular brand. Online retailer Everlane knows their Gen Z audience, and they're also known for providing customers with unprecedented transparency into their supply chain practices. So ..."}]
    [00m
    [32;1m[1;3mThought: The search results provided information related to Gen Z influencers, AI in influencer marketing, and general strategies for reaching Gen Z. However, I did not find the actual photographs for the Instagram campaign. I need to find a direct source that might have the images, such as the agency's website or a specific page dedicated to the campaign.
    
    Action: InternetSearchTool
    Action Input: {"query": "agency Instagram campaign Gen Z photographs Values Unleashed Fun with AI Behind the Brand influencer collaborations"}
    [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    [{'title': 'Top 10 AI Influencers Making Waves on Instagram - Influencer Marketing Hub', 'href': 'https://influencermarketinghub.com/ai-influencers-instagram/', 'body': "Top 10 AI Influencers on Instagram. Now that we have gone over what AI influencers are, how they are created, and why they can be a better option against regular influencers for influencer marketing, let's head to the list of the top 10 AI influencers on Instagram. 1. Lu do Magalu. 7.2M followers."}, {'title': 'Understanding Gen Z Psychographics: Insights Into Values & Behaviors', 'href': 'https://thecampusagency.com/understanding-gen-z-psychographics-insights-into-values-behaviors/', 'body': "Gen Z values family, respect, friendship, honesty, trust, loyalty, sincerity, and empathy. These principles shape their worldview and influence their decision-making. Social media platforms like Instagram, TikTok, and YouTube play a significant role in molding their values and identities. Gen Z's ethical convictions are reflected in their ..."}, {'title': '17 Expert Insights Into The Rise Of AI Influencers - Forbes', 'href': 'https://www.forbes.com/councils/forbesagencycouncil/2024/05/23/17-expert-insights-into-the-rise-of-ai-influencers/', 'body': "16. Most Established Brands Will Rely On Human-Based Marketing. We're already seeing the rise of AI influencers on Instagram and TikTok, with virtual personas getting millions of followers. I ..."}, {'title': '6 Innovative Gen Z Marketing Campaigns To Inspire Your Strategy', 'href': 'https://thecampusagency.com/innovative-gen-z-marketing-campaigns-to-inspire-your-strategy/', 'body': 'Heinz\'s "Absolutely Heinz". Heinz\'s "Absolutely Heinz" slogan became a hit, taking over social media by storm. This campaign used an interaction strategy that connected deeply with Gen Z consumers. They found the phrase "Absolutely Heinz. Ridiculously Good" both catchy and relatable, making it spread quickly online.'}, {'title': 'Influencer Marketing Gets An Upgrade: How AI Is Changing The Game - Forbes', 'href': 'https://www.forbes.com/sites/renaegregoire/2024/04/27/influencer-marketing-gets-an-upgrade-how-ai-is-changing-the-game/', 'body': 'Influencer marketing is evolving fast. AI is transforming how we find influencers, craft campaigns, and measure success. But even with all the advancements, the best campaigns will always depend ...'}]
    [00m
    [32;1m[1;3mThought: The search did not yield the specific photographs for the Instagram campaign. It might be more effective to directly access the website of the agency responsible for the campaign to find the images or relevant details.
    
    Action: Read website content
    Action Input: {"website_url": "https://theagencywebsite.com/gen-z-instagram-campaign-photographs"}
    [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    404Page not found!
    [00m
    [32;1m[1;3m[0m[32;1m[1;3mThought: I now can give a great answer
    Final Answer: Unfortunately, I was unable to locate the actual photographs captured for the Instagram campaign targeting Gen Z using the available tools. Despite searching for the campaign images and attempting to access the agency's website, the specific photographs needed for evaluation were not found. To proceed, I recommend directly contacting the team responsible for the campaign within the agency, as they should have access to the images and can provide them for evaluation. This will ensure that the photographs align with the themes of authenticity, diversity, and personalization, as outlined in the campaign goals.[0m
    
    [1m> Finished chain.[0m
    [95m 
    
    Unfortunately, I was unable to locate the actual photographs captured for the Instagram campaign targeting Gen Z using the available tools. Despite searching for the campaign images and attempting to access the agency's website, the specific photographs needed for evaluation were not found. To proceed, I recommend directly contacting the team responsible for the campaign within the agency, as they should have access to the images and can provide them for evaluation. This will ensure that the photographs align with the themes of authenticity, diversity, and personalization, as outlined in the campaign goals.
    [00m
    [32;1m[1;3mIt seems there was a repeated misunderstanding or issue in acquiring the photographs from the Senior Photographer. Without the ability to visually evaluate the photographs, it's challenging to provide a thorough review or feedback. However, I can suggest an alternative approach to ensure the content aligns with the campaign goals:
    
    Thought: I will delegate the task of obtaining the photographs to the Creative Content Creator, as they might have direct access or know how to acquire the images from the campaign team. This will allow me to evaluate and refine the content effectively.
    
    Action: Delegate work to coworker
    Action Input: {"task": "Please obtain the current photographs captured for the Instagram campaign targeting Gen Z. I need these images to evaluate and ensure they align with our themes of authenticity, diversity, and personalization. The campaign includes themes like 'Values Unleashed,' 'Fun with AI,' 'Behind the Brand,' and influencer collaborations.", "context": "The Instagram campaign targeting Gen Z focuses on capturing high-quality photographs that highlight authenticity, diversity, and personalization. We need images that show real individuals expressing their values, vibrant and playful visuals, genuine behind-the-scenes moments, and relatable, real-life product uses in influencer collaborations.", "coworker": "Creative Content Creator"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mTo obtain the current photographs captured for the Instagram campaign targeting Gen Z, I need to search for relevant information or resources that might have these images available. 
    
    Action: InternetSearchTool
    Action Input: {"query": "current photographs for Instagram campaign targeting Gen Z 'Values Unleashed', 'Fun with AI', 'Behind the Brand', influencer collaborations"}
    [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    [{'title': 'The Ultimate List Of The Top 30 Gen Z Influencers For Epic ...', 'href': 'https://billo.app/gen-z-influencers/', 'body': 'Fortunately, this post has answers. We explore the top-rated Gen Z influencers and why they stand out for marketers. 1. Kylie Jenner - Lifestyle. TikTok: 56.2 Million Followers. Instagram: 398 Million Followers. Kylie Kenner is one of the most influential influencers and social media personalities out there.'}, {'title': 'Gen Z Stats & Trends: How to Market to Gen Z in 2025 - Later', 'href': 'https://later.com/blog/gen-z-trends/', 'body': 'Social Media Marketing Blog. Gen Z, born between 1997 and 2021, is a generation who have grown up with technology and social media. According to McKinsey, Gen Z will account for 27% of the global workforce by 2025, significantly increasing their economic and cultural influence and making them more powerful than ever.'}, {'title': 'How To Refresh 17 Tried-And-True Marketing Strategies For Gen-Z - Forbes', 'href': 'https://www.forbes.com/councils/forbesagencycouncil/2024/03/11/how-to-refresh-17-tried-and-true-marketing-strategies-for-gen-z/', 'body': '6. Focus On Micro-Influencers Rather Than Celebrities. One example of adapting strategies would be influencer marketing, which has been adapted for Gen-Z by focusing on micro-influencers who are ...'}, {'title': 'How the Best Brand-Influencer Partnerships Reach Gen Z', 'href': 'https://hbr.org/2023/06/how-the-best-brand-influencer-partnerships-reach-gen-z', 'body': 'This young influencer — who just graduated from the University of Miami — has more than 5 million followers on TikTok and was recently signed by United Talent Agency (UTA), one of the top ...'}, {'title': 'How Marketers Can Target Gen Z: Tapping Into Social Media And ... - Forbes', 'href': 'https://www.forbes.com/sites/meganpoinski/2023/11/15/how-marketers-can-target-gen-z-tapping-into-social-media-and-tiktok/', 'body': "According to Horizon Media's 2023 Gen Z Field Guide, shared exclusively with Forbes, 74% of the people in this age group use TikTok as a search platform. More than half—51%—prefer it to ..."}]
    [00m
    [32;1m[1;3mThought: The search did not provide direct access to the current photographs for the Instagram campaign targeting Gen Z. I need to find a specific resource or website that might host these campaign images.
    
    Action: InternetSearchTool
    Action Input: {"query": "Instagram campaign photographs for Gen Z 'Values Unleashed', 'Fun with AI', 'Behind the Brand'"}
    [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    [{'title': 'Navigating the Social Media Landscape: Top Trends for Gen Z Engagement', 'href': 'https://www.theatticism.com/post/navigating-the-social-media-landscape-top-trends-for-gen-z-engagement', 'body': "Learn how to build trust with this audience by showcasing your brand's values and mission through your social media presence. How to Tip 1: Behind-the-Scenes Sneak Peeks - Share candid glimpses of your team, workspace, or product creation process to give Gen Z a sense of your brand's authenticity. How to Tip 2: Live Q&A Sessions - Host live ..."}, {'title': 'Instagram: The Ultimate Platform For Gen Z Engagement - Nerds Collective', 'href': 'https://nerdscollective.com/instagram-gen-z-engagement/', 'body': "Instagram has evolved into the ultimate playground for Gen Z engagement. With its visual allure and youth-centric appeal, it has become the go-to platform for the digitally savvy generation. In this blog, we will dive into the captivating world of Instagram, exploring why it has become Gen Z's virtual hangout of choice."}, {'title': '6 Innovative Gen Z Marketing Campaigns To Inspire Your Strategy', 'href': 'https://thecampusagency.com/innovative-gen-z-marketing-campaigns-to-inspire-your-strategy/', 'body': 'Heinz\'s "Absolutely Heinz". Heinz\'s "Absolutely Heinz" slogan became a hit, taking over social media by storm. This campaign used an interaction strategy that connected deeply with Gen Z consumers. They found the phrase "Absolutely Heinz. Ridiculously Good" both catchy and relatable, making it spread quickly online.'}, {'title': 'Instagram celebrates Gen Z connections in short film-led campaign', 'href': 'https://www.marketingdive.com/news/instagram-connections-campaign-gen-z-marketing/689862/', 'body': 'The "Instagram Connections" brand campaign shows the Meta platform making a pitch to and about Gen Z using content that resembles short films more than traditional ads. The effort is grounded in a sense of realism and authenticity that is in line with the cohort\'s preferences and attitudes. Both shorts document the ups and downs and FOMO ...'}, {'title': "The Evolution of Marketing Funnel: Gen Z's Impact on Brand Engagement", 'href': 'https://read.crowdfireapp.com/2024/04/22/the-evolution-of-marketing-funnel-gen-zs-impact-on-brand-engagement/', 'body': "In conclusion, Generation Z's influence on the marketing funnel is undeniable, reshaping the way brands engage with consumers in the digital age. As the first generation to grow up entirely in the era of smartphones, social media, and instant communication, Gen Z brings a unique set of preferences, behaviors, and expectations to the table."}]
    [00m
    [32;1m[1;3mThought: The search results did not provide direct links to the actual photographs from the Instagram campaign targeting Gen Z. It seems I need to directly access a specific resource or website that hosts these images or potentially reach out to a team member who might have access to the campaign assets.
    
    Final Answer: I was unable to directly access the current photographs for the Instagram campaign targeting Gen Z using the available tools. To obtain these images, it might be best to contact the team member or department responsible for managing the campaign assets, as they likely have the images stored in a shared drive or internal database.[0m
    
    [1m> Finished chain.[0m
    [95m 
    
    I was unable to directly access the current photographs for the Instagram campaign targeting Gen Z using the available tools. To obtain these images, it might be best to contact the team member or department responsible for managing the campaign assets, as they likely have the images stored in a shared drive or internal database.
    [00m
    [32;1m[1;3mThought: Since my previous attempts to obtain the photographs from the Senior Photographer and Creative Content Creator were unsuccessful, I should try delegating the task to someone else. I will delegate the task to the Chief Marketing Strategist, who might have access to the campaign assets or know who to contact within the agency to retrieve the photographs.
    
    Action: Delegate work to coworker
    Action Input: {"task": "Please obtain the current photographs captured for the Instagram campaign targeting Gen Z. I need these images to evaluate and ensure they align with our themes of authenticity, diversity, and personalization. The campaign includes themes like 'Values Unleashed,' 'Fun with AI,' 'Behind the Brand,' and influencer collaborations.", "context": "The Instagram campaign targeting Gen Z focuses on capturing high-quality photographs that highlight authenticity, diversity, and personalization. We need images that show real individuals expressing their values, vibrant and playful visuals, genuine behind-the-scenes moments, and relatable, real-life product uses in influencer collaborations.", "coworker": "Chief Marketing Strategist"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mThought: To evaluate the photographs captured for the Instagram campaign targeting Gen Z, I need to access Instagram to view the images. Since I don't have a direct tool to access Instagram content, I will perform an internet search to find relevant information or links to the campaign images.
    
    Action: InternetSearchTool
    Action Input: {"query": "Gen Z Instagram campaign photographs Values Unleashed Fun with AI Behind the Brand influencer collaborations"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    [{'title': 'Top 10 AI Influencers Making Waves on Instagram - Influencer Marketing Hub', 'href': 'https://influencermarketinghub.com/ai-influencers-instagram/', 'body': "Top 10 AI Influencers on Instagram. Now that we have gone over what AI influencers are, how they are created, and why they can be a better option against regular influencers for influencer marketing, let's head to the list of the top 10 AI influencers on Instagram. 1. Lu do Magalu. 7.2M followers."}, {'title': 'How the Best Brand-Influencer Partnerships Reach Gen Z', 'href': 'https://hbr.org/2023/06/how-the-best-brand-influencer-partnerships-reach-gen-z', 'body': "How the Best Brand-Influencer Partnerships Reach Gen Z. Summary. Authenticity is among Gen Z's most important values. They feel empowered to ask and answer their own questions in a variety of ..."}, {'title': 'Gen Z has a surprising opinion about AI influencers on social media ...', 'href': 'https://nypost.com/2024/05/08/tech/gen-z-has-a-surprising-opinion-about-ai-influencers-on-social-media-study-finds/', 'body': "Only 35% of Gen Z consumers considered authenticity in influencers to be a top trait they care about, according to Sprout Social's 2024 Influencer Marketing Report. Social media influencers were ..."}, {'title': 'The Ultimate List Of The Top 30 Gen Z Influencers For Epic ...', 'href': 'https://billo.app/gen-z-influencers/', 'body': 'Fortunately, this post has answers. We explore the top-rated Gen Z influencers and why they stand out for marketers. 1. Kylie Jenner - Lifestyle. TikTok: 56.2 Million Followers. Instagram: 398 Million Followers. Kylie Kenner is one of the most influential influencers and social media personalities out there.'}, {'title': 'Influencer Marketing Gets An Upgrade: How AI Is Changing The Game - Forbes', 'href': 'https://www.forbes.com/sites/renaegregoire/2024/04/27/influencer-marketing-gets-an-upgrade-how-ai-is-changing-the-game/', 'body': 'Influencer marketing is evolving fast. AI is transforming how we find influencers, craft campaigns, and measure success. But even with all the advancements, the best campaigns will always depend ...'}]
    [00m
    [32;1m[1;3mThought: The internet search did not yield direct results or links to the specific Instagram campaign photographs. I need to refine my search query to better target the specific campaign themes like 'Values Unleashed,' 'Fun with AI,' 'Behind the Brand,' and influencer collaborations for Gen Z.
    
    Action: InternetSearchTool
    Action Input: {"query": "Gen Z Instagram campaign photos Values Unleashed Fun with AI Behind the Brand influencer collaborations"}[0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    [{'title': 'Gen Z has a surprising opinion about AI influencers on social media ...', 'href': 'https://nypost.com/2024/05/08/tech/gen-z-has-a-surprising-opinion-about-ai-influencers-on-social-media-study-finds/', 'body': "Only 35% of Gen Z consumers considered authenticity in influencers to be a top trait they care about, according to Sprout Social's 2024 Influencer Marketing Report. Social media influencers were ..."}, {'title': 'How the Best Brand-Influencer Partnerships Reach Gen Z', 'href': 'https://hbr.org/2023/06/how-the-best-brand-influencer-partnerships-reach-gen-z', 'body': "How the Best Brand-Influencer Partnerships Reach Gen Z. Summary. Authenticity is among Gen Z's most important values. They feel empowered to ask and answer their own questions in a variety of ..."}, {'title': 'The Ultimate List Of The Top 30 Gen Z Influencers For Epic ...', 'href': 'https://billo.app/gen-z-influencers/', 'body': 'Fortunately, this post has answers. We explore the top-rated Gen Z influencers and why they stand out for marketers. 1. Kylie Jenner - Lifestyle. TikTok: 56.2 Million Followers. Instagram: 398 Million Followers. Kylie Kenner is one of the most influential influencers and social media personalities out there.'}, {'title': 'The best Gen Z marketing campaigns - Khoros', 'href': 'https://khoros.com/blog/top-gen-z-campaigns', 'body': '7. Buy better, wear longer by Levi\'s. 8. #AerieREAL. 9. "Supergucci" by Gucci. Gen Z Marketing Bottom Line. Generation Z are the youngest consumers, but do not be mistaken; they are not a demographic to ignore. In fact, they make up 40% of all consumers and their share of retail spend is expected to increase to 48% by 2030.'}, {'title': 'Instagram: The Ultimate Platform For Gen Z Engagement - Nerds Collective', 'href': 'https://nerdscollective.com/instagram-gen-z-engagement/', 'body': "Instagram has evolved into the ultimate playground for Gen Z engagement. With its visual allure and youth-centric appeal, it has become the go-to platform for the digitally savvy generation. In this blog, we will dive into the captivating world of Instagram, exploring why it has become Gen Z's virtual hangout of choice."}]
    [00m
    [32;1m[1;3mThought: The refined search did not yield the specific photographs from the Instagram campaign. There might not be a direct online source where these images are collected or shared publicly. Given the tools at my disposal, I need to consider other ways to obtain these images.
    
    Final Answer: I attempted to locate the current photographs captured for the Instagram campaign targeting Gen Z using available tools, but I couldn't find specific images online. I recommend reaching out directly to the team responsible for the Instagram campaign or the content creator managing the campaign to request access to the images. Alternatively, if there's a private database or internal portal where these images are stored, accessing it would be the next step.[0m
    
    [1m> Finished chain.[0m
    [95m 
    
    I attempted to locate the current photographs captured for the Instagram campaign targeting Gen Z using available tools, but I couldn't find specific images online. I recommend reaching out directly to the team responsible for the Instagram campaign or the content creator managing the campaign to request access to the images. Alternatively, if there's a private database or internal portal where these images are stored, accessing it would be the next step.
    [00m
    [32;1m[1;3m[0m[32;1m[1;3mThought: I now can give a great answer
    Final Answer: Unfortunately, despite my efforts to obtain the photographs captured for the Instagram campaign targeting Gen Z, I was unable to access them using the available tools or through direct communication with the team members. 
    
    To proceed, I recommend directly contacting the team responsible for the campaign within the agency, as they should have access to the images and can provide them for evaluation. Once you have the photographs, you can review and refine the content to ensure it aligns with the campaign's themes of authenticity, diversity, and personalization, as outlined in the campaign goals. Focus on capturing real individuals expressing their values for "Values Unleashed," using vibrant and playful visuals for "Fun with AI," showcasing genuine behind-the-scenes moments for "Behind the Brand," and highlighting relatable, real-life uses of products in influencer collaborations. These approaches are designed to resonate with Gen Z's preference for authenticity, innovation, and community connection.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-01 14:21:56][DEBUG]: == [Chief Creative Director] Task output: Unfortunately, despite my efforts to obtain the photographs captured for the Instagram campaign targeting Gen Z, I was unable to access them using the available tools or through direct communication with the team members. 
    
    To proceed, I recommend directly contacting the team responsible for the campaign within the agency, as they should have access to the images and can provide them for evaluation. Once you have the photographs, you can review and refine the content to ensure it aligns with the campaign's themes of authenticity, diversity, and personalization, as outlined in the campaign goals. Focus on capturing real individuals expressing their values for "Values Unleashed," using vibrant and playful visuals for "Fun with AI," showcasing genuine behind-the-scenes moments for "Behind the Brand," and highlighting relatable, real-life uses of products in influencer collaborations. These approaches are designed to resonate with Gen Z's preference for authenticity, innovation, and community connection.
    
    [00m
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">### Task Output ###
Unfortunately, despite my efforts to obtain the photographs captured for the Instagram campaign targeting Gen Z, I 
was unable to access them using the available tools or through direct communication with the team members. 

To proceed, I recommend directly contacting the team responsible for the campaign within the agency, as they should
have access to the images and can provide them for evaluation. Once you have the photographs, you can review and 
refine the content to ensure it aligns with the campaign's themes of authenticity, diversity, and personalization, 
as outlined in the campaign goals. Focus on capturing real individuals expressing their values for <span style="color: #008000; text-decoration-color: #008000">"Values </span>
<span style="color: #008000; text-decoration-color: #008000">Unleashed,"</span> using vibrant and playful visuals for <span style="color: #008000; text-decoration-color: #008000">"Fun with AI,"</span> showcasing genuine behind-the-scenes moments for 
<span style="color: #008000; text-decoration-color: #008000">"Behind the Brand,"</span> and highlighting relatable, real-life uses of products in influencer collaborations. These 
approaches are designed to resonate with Gen Z's preference for authenticity, innovation, and community connection.
</pre>



    None
    


```python

```
