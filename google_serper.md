# Google Serper

This notebook goes over how to use the `Google Serper` component to search the web. First you need to sign up for a free account at [serper.dev](https://serper.dev) and get your api key.


```python
%pip install --upgrade --quiet  langchain-community
```


```python
import os
import pprint

os.environ["SERPER_API_KEY"] = ""
```


```python
from langchain_community.utilities import GoogleSerperAPIWrapper
```


```python
search = GoogleSerperAPIWrapper()
```


```python
search.run("Obama's first name?")
```




    'Barack Hussein Obama II'



## As part of a Self Ask With Search Chain


```python
os.environ["OPENAI_API_KEY"] = ""
```


```python
from langchain.agents import AgentType, initialize_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)
self_ask_with_search.run(
    "What is the hometown of the reigning men's U.S. Open champion?"
)
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m Yes.
    Follow up: Who is the reigning men's U.S. Open champion?[0m
    Intermediate answer: [36;1m[1;3mCurrent champions Carlos Alcaraz, 2022 men's singles champion.[0m
    [32;1m[1;3mFollow up: Where is Carlos Alcaraz from?[0m
    Intermediate answer: [36;1m[1;3mEl Palmar, Spain[0m
    [32;1m[1;3mSo the final answer is: El Palmar, Spain[0m
    
    [1m> Finished chain.[0m
    




    'El Palmar, Spain'



## Obtaining results with metadata
If you would also like to obtain the results in a structured way including metadata. For this we will be using the `results` method of the wrapper.


```python
search = GoogleSerperAPIWrapper()
results = search.results("Apple Inc.")
pprint.pp(results)
```

    {'searchParameters': {'q': 'Apple Inc.',
                          'gl': 'us',
                          'hl': 'en',
                          'num': 10,
                          'type': 'search'},
     'knowledgeGraph': {'title': 'Apple',
                        'type': 'Technology company',
                        'website': 'http://www.apple.com/',
                        'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQwGQRv5TjjkycpctY66mOg_e2-npacrmjAb6_jAWhzlzkFE3OTjxyzbA&s=0',
                        'description': 'Apple Inc. is an American multinational '
                                       'technology company headquartered in '
                                       'Cupertino, California. Apple is the '
                                       "world's largest technology company by "
                                       'revenue, with US$394.3 billion in 2022 '
                                       'revenue. As of March 2023, Apple is the '
                                       "world's biggest...",
                        'descriptionSource': 'Wikipedia',
                        'descriptionLink': 'https://en.wikipedia.org/wiki/Apple_Inc.',
                        'attributes': {'Customer service': '1 (800) 275-2273',
                                       'CEO': 'Tim Cook (Aug 24, 2011–)',
                                       'Headquarters': 'Cupertino, CA',
                                       'Founded': 'April 1, 1976, Los Altos, CA',
                                       'Founders': 'Steve Jobs, Steve Wozniak, '
                                                   'Ronald Wayne, and more',
                                       'Products': 'iPhone, iPad, Apple TV, and '
                                                   'more'}},
     'organic': [{'title': 'Apple',
                  'link': 'https://www.apple.com/',
                  'snippet': 'Discover the innovative world of Apple and shop '
                             'everything iPhone, iPad, Apple Watch, Mac, and Apple '
                             'TV, plus explore accessories, entertainment, ...',
                  'sitelinks': [{'title': 'Support',
                                 'link': 'https://support.apple.com/'},
                                {'title': 'iPhone',
                                 'link': 'https://www.apple.com/iphone/'},
                                {'title': 'Site Map',
                                 'link': 'https://www.apple.com/sitemap/'},
                                {'title': 'Business',
                                 'link': 'https://www.apple.com/business/'},
                                {'title': 'Mac',
                                 'link': 'https://www.apple.com/mac/'},
                                {'title': 'Watch',
                                 'link': 'https://www.apple.com/watch/'}],
                  'position': 1},
                 {'title': 'Apple Inc. - Wikipedia',
                  'link': 'https://en.wikipedia.org/wiki/Apple_Inc.',
                  'snippet': 'Apple Inc. is an American multinational technology '
                             'company headquartered in Cupertino, California. '
                             "Apple is the world's largest technology company by "
                             'revenue, ...',
                  'attributes': {'Products': 'AirPods; Apple Watch; iPad; iPhone; '
                                             'Mac; Full list',
                                 'Founders': 'Steve Jobs; Steve Wozniak; Ronald '
                                             'Wayne; Mike Markkula'},
                  'sitelinks': [{'title': 'History',
                                 'link': 'https://en.wikipedia.org/wiki/History_of_Apple_Inc.'},
                                {'title': 'Timeline of Apple Inc. products',
                                 'link': 'https://en.wikipedia.org/wiki/Timeline_of_Apple_Inc._products'},
                                {'title': 'Litigation involving Apple Inc.',
                                 'link': 'https://en.wikipedia.org/wiki/Litigation_involving_Apple_Inc.'},
                                {'title': 'Apple Store',
                                 'link': 'https://en.wikipedia.org/wiki/Apple_Store'}],
                  'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRvmB5fT1LjqpZx02UM7IJq0Buoqt0DZs_y0dqwxwSWyP4PIN9FaxuTea0&s',
                  'position': 2},
                 {'title': 'Apple Inc. | History, Products, Headquarters, & Facts '
                           '| Britannica',
                  'link': 'https://www.britannica.com/topic/Apple-Inc',
                  'snippet': 'Apple Inc., formerly Apple Computer, Inc., American '
                             'manufacturer of personal computers, smartphones, '
                             'tablet computers, computer peripherals, and computer '
                             '...',
                  'attributes': {'Related People': 'Steve Jobs Steve Wozniak Jony '
                                                   'Ive Tim Cook Angela Ahrendts',
                                 'Date': '1976 - present'},
                  'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS3liELlhrMz3Wpsox29U8jJ3L8qETR0hBWHXbFnwjwQc34zwZvFELst2E&s',
                  'position': 3},
                 {'title': 'AAPL: Apple Inc Stock Price Quote - NASDAQ GS - '
                           'Bloomberg.com',
                  'link': 'https://www.bloomberg.com/quote/AAPL:US',
                  'snippet': 'AAPL:USNASDAQ GS. Apple Inc. COMPANY INFO ; Open. '
                             '170.09 ; Prev Close. 169.59 ; Volume. 48,425,696 ; '
                             'Market Cap. 2.667T ; Day Range. 167.54170.35.',
                  'position': 4},
                 {'title': 'Apple Inc. (AAPL) Company Profile & Facts - Yahoo '
                           'Finance',
                  'link': 'https://finance.yahoo.com/quote/AAPL/profile/',
                  'snippet': 'Apple Inc. designs, manufactures, and markets '
                             'smartphones, personal computers, tablets, wearables, '
                             'and accessories worldwide. The company offers '
                             'iPhone, a line ...',
                  'position': 5},
                 {'title': 'Apple Inc. (AAPL) Stock Price, News, Quote & History - '
                           'Yahoo Finance',
                  'link': 'https://finance.yahoo.com/quote/AAPL',
                  'snippet': 'Find the latest Apple Inc. (AAPL) stock quote, '
                             'history, news and other vital information to help '
                             'you with your stock trading and investing.',
                  'position': 6}],
     'peopleAlsoAsk': [{'question': 'What does Apple Inc do?',
                        'snippet': 'Apple Inc. (Apple) designs, manufactures and '
                                   'markets smartphones, personal\n'
                                   'computers, tablets, wearables and accessories '
                                   'and sells a range of related\n'
                                   'services.',
                        'title': 'AAPL.O - | Stock Price & Latest News - Reuters',
                        'link': 'https://www.reuters.com/markets/companies/AAPL.O/'},
                       {'question': 'What is the full form of Apple Inc?',
                        'snippet': '(formerly Apple Computer Inc.) is an American '
                                   'computer and consumer electronics\n'
                                   'company famous for creating the iPhone, iPad '
                                   'and Macintosh computers.',
                        'title': 'What is Apple? An products and history overview '
                                 '- TechTarget',
                        'link': 'https://www.techtarget.com/whatis/definition/Apple'},
                       {'question': 'What is Apple Inc iPhone?',
                        'snippet': 'Apple Inc (Apple) designs, manufactures, and '
                                   'markets smartphones, tablets,\n'
                                   'personal computers, and wearable devices. The '
                                   'company also offers software\n'
                                   'applications and related services, '
                                   'accessories, and third-party digital content.\n'
                                   "Apple's product portfolio includes iPhone, "
                                   'iPad, Mac, iPod, Apple Watch, and\n'
                                   'Apple TV.',
                        'title': 'Apple Inc Company Profile - Apple Inc Overview - '
                                 'GlobalData',
                        'link': 'https://www.globaldata.com/company-profile/apple-inc/'},
                       {'question': 'Who runs Apple Inc?',
                        'snippet': 'Timothy Donald Cook (born November 1, 1960) is '
                                   'an American business executive\n'
                                   'who has been the chief executive officer of '
                                   'Apple Inc. since 2011. Cook\n'
                                   "previously served as the company's chief "
                                   'operating officer under its co-founder\n'
                                   'Steve Jobs. He is the first CEO of any Fortune '
                                   '500 company who is openly gay.',
                        'title': 'Tim Cook - Wikipedia',
                        'link': 'https://en.wikipedia.org/wiki/Tim_Cook'}],
     'relatedSearches': [{'query': 'Who invented the iPhone'},
                         {'query': 'Apple iPhone'},
                         {'query': 'History of Apple company PDF'},
                         {'query': 'Apple company history'},
                         {'query': 'Apple company introduction'},
                         {'query': 'Apple India'},
                         {'query': 'What does Apple Inc own'},
                         {'query': 'Apple Inc After Steve'},
                         {'query': 'Apple Watch'},
                         {'query': 'Apple App Store'}]}
    

## Searching for Google Images
We can also query Google Images using this wrapper. For example:


```python
search = GoogleSerperAPIWrapper(type="images")
results = search.results("Lion")
pprint.pp(results)
```

    {'searchParameters': {'q': 'Lion',
                          'gl': 'us',
                          'hl': 'en',
                          'num': 10,
                          'type': 'images'},
     'images': [{'title': 'Lion - Wikipedia',
                 'imageUrl': 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Lion_waiting_in_Namibia.jpg/1200px-Lion_waiting_in_Namibia.jpg',
                 'imageWidth': 1200,
                 'imageHeight': 900,
                 'thumbnailUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRye79ROKwjfb6017jr0iu8Bz2E1KKuHg-A4qINJaspyxkZrkw&amp;s',
                 'thumbnailWidth': 259,
                 'thumbnailHeight': 194,
                 'source': 'Wikipedia',
                 'domain': 'en.wikipedia.org',
                 'link': 'https://en.wikipedia.org/wiki/Lion',
                 'position': 1},
                {'title': 'Lion | Characteristics, Habitat, & Facts | Britannica',
                 'imageUrl': 'https://cdn.britannica.com/55/2155-050-604F5A4A/lion.jpg',
                 'imageWidth': 754,
                 'imageHeight': 752,
                 'thumbnailUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS3fnDub1GSojI0hJ-ZGS8Tv-hkNNloXh98DOwXZoZ_nUs3GWSd&amp;s',
                 'thumbnailWidth': 225,
                 'thumbnailHeight': 224,
                 'source': 'Encyclopedia Britannica',
                 'domain': 'www.britannica.com',
                 'link': 'https://www.britannica.com/animal/lion',
                 'position': 2},
                {'title': 'African lion, facts and photos',
                 'imageUrl': 'https://i.natgeofe.com/n/487a0d69-8202-406f-a6a0-939ed3704693/african-lion.JPG',
                 'imageWidth': 3072,
                 'imageHeight': 2043,
                 'thumbnailUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTPlTarrtDbyTiEm-VI_PML9VtOTVPuDXJ5ybDf_lN11H2mShk&amp;s',
                 'thumbnailWidth': 275,
                 'thumbnailHeight': 183,
                 'source': 'National Geographic',
                 'domain': 'www.nationalgeographic.com',
                 'link': 'https://www.nationalgeographic.com/animals/mammals/facts/african-lion',
                 'position': 3},
                {'title': 'Saint Louis Zoo | African Lion',
                 'imageUrl': 'https://optimise2.assets-servd.host/maniacal-finch/production/animals/african-lion-01-01.jpg?w=1200&auto=compress%2Cformat&fit=crop&dm=1658933674&s=4b63f926a0f524f2087a8e0613282bdb',
                 'imageWidth': 1200,
                 'imageHeight': 1200,
                 'thumbnailUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTlewcJ5SwC7yKup6ByaOjTnAFDeoOiMxyJTQaph2W_I3dnks4&amp;s',
                 'thumbnailWidth': 225,
                 'thumbnailHeight': 225,
                 'source': 'St. Louis Zoo',
                 'domain': 'stlzoo.org',
                 'link': 'https://stlzoo.org/animals/mammals/carnivores/lion',
                 'position': 4},
                {'title': 'How to Draw a Realistic Lion like an Artist - Studio '
                          'Wildlife',
                 'imageUrl': 'https://studiowildlife.com/wp-content/uploads/2021/10/245528858_183911853822648_6669060845725210519_n.jpg',
                 'imageWidth': 1431,
                 'imageHeight': 2048,
                 'thumbnailUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTmn5HayVj3wqoBDQacnUtzaDPZzYHSLKUlIEcni6VB8w0mVeA&amp;s',
                 'thumbnailWidth': 188,
                 'thumbnailHeight': 269,
                 'source': 'Studio Wildlife',
                 'domain': 'studiowildlife.com',
                 'link': 'https://studiowildlife.com/how-to-draw-a-realistic-lion-like-an-artist/',
                 'position': 5},
                {'title': 'Lion | Characteristics, Habitat, & Facts | Britannica',
                 'imageUrl': 'https://cdn.britannica.com/29/150929-050-547070A1/lion-Kenya-Masai-Mara-National-Reserve.jpg',
                 'imageWidth': 1600,
                 'imageHeight': 1085,
                 'thumbnailUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSCqaKY_THr0IBZN8c-2VApnnbuvKmnsWjfrwKoWHFR9w3eN5o&amp;s',
                 'thumbnailWidth': 273,
                 'thumbnailHeight': 185,
                 'source': 'Encyclopedia Britannica',
                 'domain': 'www.britannica.com',
                 'link': 'https://www.britannica.com/animal/lion',
                 'position': 6},
                {'title': "Where do lions live? Facts about lions' habitats and "
                          'other cool facts',
                 'imageUrl': 'https://www.gannett-cdn.com/-mm-/b2b05a4ab25f4fca0316459e1c7404c537a89702/c=0-0-1365-768/local/-/media/2022/03/16/USATODAY/usatsports/imageForEntry5-ODq.jpg?width=1365&height=768&fit=crop&format=pjpg&auto=webp',
                 'imageWidth': 1365,
                 'imageHeight': 768,
                 'thumbnailUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTc_4vCHscgvFvYy3PSrtIOE81kNLAfhDK8F3mfOuotL0kUkbs&amp;s',
                 'thumbnailWidth': 299,
                 'thumbnailHeight': 168,
                 'source': 'USA Today',
                 'domain': 'www.usatoday.com',
                 'link': 'https://www.usatoday.com/story/news/2023/01/08/where-do-lions-live-habitat/10927718002/',
                 'position': 7},
                {'title': 'Lion',
                 'imageUrl': 'https://i.natgeofe.com/k/1d33938b-3d02-4773-91e3-70b113c3b8c7/lion-male-roar_square.jpg',
                 'imageWidth': 3072,
                 'imageHeight': 3072,
                 'thumbnailUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQqLfnBrBLcTiyTZynHH3FGbBtX2bd1ScwpcuOLnksTyS9-4GM&amp;s',
                 'thumbnailWidth': 225,
                 'thumbnailHeight': 225,
                 'source': 'National Geographic Kids',
                 'domain': 'kids.nationalgeographic.com',
                 'link': 'https://kids.nationalgeographic.com/animals/mammals/facts/lion',
                 'position': 8},
                {'title': "Lion | Smithsonian's National Zoo",
                 'imageUrl': 'https://nationalzoo.si.edu/sites/default/files/styles/1400_scale/public/animals/exhibit/africanlion-005.jpg?itok=6wA745g_',
                 'imageWidth': 1400,
                 'imageHeight': 845,
                 'thumbnailUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSgB3z_D4dMEOWJ7lajJk4XaQSL4DdUvIRj4UXZ0YoE5fGuWuo&amp;s',
                 'thumbnailWidth': 289,
                 'thumbnailHeight': 174,
                 'source': "Smithsonian's National Zoo",
                 'domain': 'nationalzoo.si.edu',
                 'link': 'https://nationalzoo.si.edu/animals/lion',
                 'position': 9},
                {'title': "Zoo's New Male Lion Explores Habitat for the First Time "
                          '- Virginia Zoo',
                 'imageUrl': 'https://virginiazoo.org/wp-content/uploads/2022/04/ZOO_0056-scaled.jpg',
                 'imageWidth': 2560,
                 'imageHeight': 2141,
                 'thumbnailUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTDCG7XvXRCwpe_-Vy5mpvrQpVl5q2qwgnDklQhrJpQzObQGz4&amp;s',
                 'thumbnailWidth': 246,
                 'thumbnailHeight': 205,
                 'source': 'Virginia Zoo',
                 'domain': 'virginiazoo.org',
                 'link': 'https://virginiazoo.org/zoos-new-male-lion-explores-habitat-for-thefirst-time/',
                 'position': 10}]}
    

## Searching for Google News
We can also query Google News using this wrapper. For example:


```python
search = GoogleSerperAPIWrapper(type="news")
results = search.results("Tesla Inc.")
pprint.pp(results)
```

    {'searchParameters': {'q': 'Tesla Inc.',
                          'gl': 'us',
                          'hl': 'en',
                          'num': 10,
                          'type': 'news'},
     'news': [{'title': 'ISS recommends Tesla investors vote against re-election '
                        'of Robyn Denholm',
               'link': 'https://www.reuters.com/business/autos-transportation/iss-recommends-tesla-investors-vote-against-re-election-robyn-denholm-2023-05-04/',
               'snippet': 'Proxy advisory firm ISS on Wednesday recommended Tesla '
                          'investors vote against re-election of board chair Robyn '
                          'Denholm, citing "concerns on...',
               'date': '5 mins ago',
               'source': 'Reuters',
               'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcROdETe_GUyp1e8RHNhaRM8Z_vfxCvdfinZwzL1bT1ZGSYaGTeOojIdBoLevA&s',
               'position': 1},
              {'title': 'Global companies by market cap: Tesla fell most in April',
               'link': 'https://www.reuters.com/markets/global-companies-by-market-cap-tesla-fell-most-april-2023-05-02/',
               'snippet': 'Tesla Inc was the biggest loser among top companies by '
                          'market capitalisation in April, hit by disappointing '
                          'quarterly earnings after it...',
               'date': '1 day ago',
               'source': 'Reuters',
               'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4u4CP8aOdGyRFH6o4PkXi-_eZDeY96vLSag5gDjhKMYf98YBER2cZPbkStQ&s',
               'position': 2},
              {'title': 'Tesla Wanted an EV Price War. Ford Showed Up.',
               'link': 'https://www.bloomberg.com/opinion/articles/2023-05-03/tesla-wanted-an-ev-price-war-ford-showed-up',
               'snippet': 'The legacy automaker is paring back the cost of its '
                          'Mustang Mach-E model after Tesla discounted its '
                          'competing EVs, portending tighter...',
               'date': '6 hours ago',
               'source': 'Bloomberg.com',
               'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS_3Eo4VI0H-nTeIbYc5DaQn5ep7YrWnmhx6pv8XddFgNF5zRC9gEpHfDq8yQ&s',
               'position': 3},
              {'title': 'Joby Aviation to get investment from Tesla shareholder '
                        'Baillie Gifford',
               'link': 'https://finance.yahoo.com/news/joby-aviation-investment-tesla-shareholder-204450712.html',
               'snippet': 'This comes days after Joby clinched a $55 million '
                          'contract extension to deliver up to nine air taxis to '
                          'the U.S. Air Force,...',
               'date': '4 hours ago',
               'source': 'Yahoo Finance',
               'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQO0uVn297LI-xryrPNqJ-apUOulj4ohM-xkN4OfmvMOYh1CPdUEBbYx6hviw&s',
               'position': 4},
              {'title': 'Tesla resumes U.S. orders for a Model 3 version at lower '
                        'price, range',
               'link': 'https://finance.yahoo.com/news/tesla-resumes-us-orders-model-045736115.html',
               'snippet': '(Reuters) -Tesla Inc has resumed taking orders for its '
                          'Model 3 long-range vehicle in the United States, the '
                          "company's website showed late on...",
               'date': '19 hours ago',
               'source': 'Yahoo Finance',
               'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTIZetJ62sQefPfbQ9KKDt6iH7Mc0ylT5t_hpgeeuUkHhJuAx2FOJ4ZTRVDFg&s',
               'position': 5},
              {'title': 'The Tesla Model 3 Long Range AWD Is Now Available in the '
                        'U.S. With 325 Miles of Range',
               'link': 'https://www.notateslaapp.com/news/1393/tesla-reopens-orders-for-model-3-long-range-after-months-of-unavailability',
               'snippet': 'Tesla has reopened orders for the Model 3 Long Range '
                          'RWD, which has been unavailable for months due to high '
                          'demand.',
               'date': '7 hours ago',
               'source': 'Not a Tesla App',
               'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSecrgxZpRj18xIJY-nDHljyP-A4ejEkswa9eq77qhMNrScnVIqe34uql5U4w&s',
               'position': 6},
              {'title': 'Tesla Cybertruck alpha prototype spotted at the Fremont '
                        'factory in new pics and videos',
               'link': 'https://www.teslaoracle.com/2023/05/03/tesla-cybertruck-alpha-prototype-interior-and-exterior-spotted-at-the-fremont-factory-in-new-pics-and-videos/',
               'snippet': 'A Tesla Cybertruck alpha prototype goes to Fremont, '
                          'California for another round of testing before going to '
                          'production later this year (pics...',
               'date': '14 hours ago',
               'source': 'Tesla Oracle',
               'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRO7M5ZLQE-Zo4-_5dv9hNAQZ3wSqfvYCuKqzxHG-M6CgLpwPMMG_ssebdcMg&s',
               'position': 7},
              {'title': 'Tesla putting facility in new part of country - Austin '
                        'Business Journal',
               'link': 'https://www.bizjournals.com/austin/news/2023/05/02/tesla-leases-building-seattle-area.html',
               'snippet': 'Check out what Puget Sound Business Journal has to '
                          "report about the Austin-based company's real estate "
                          'footprint in the Pacific Northwest.',
               'date': '22 hours ago',
               'source': 'The Business Journals',
               'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR9kIEHWz1FcHKDUtGQBS0AjmkqtyuBkQvD8kyIY3kpaPrgYaN7I_H2zoOJsA&s',
               'position': 8},
              {'title': 'Tesla (TSLA) Resumes Orders for Model 3 Long Range After '
                        'Backlog',
               'link': 'https://www.bloomberg.com/news/articles/2023-05-03/tesla-resumes-orders-for-popular-model-3-long-range-at-47-240',
               'snippet': 'Tesla Inc. has resumed taking orders for its Model 3 '
                          'Long Range edition with a starting price of $47240, '
                          'according to its website.',
               'date': '5 hours ago',
               'source': 'Bloomberg.com',
               'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTWWIC4VpMTfRvSyqiomODOoLg0xhoBf-Tc1qweKnSuaiTk-Y1wMJZM3jct0w&s',
               'position': 9}]}
    

If you want to only receive news articles published in the last hour, you can do the following:


```python
search = GoogleSerperAPIWrapper(type="news", tbs="qdr:h")
results = search.results("Tesla Inc.")
pprint.pp(results)
```

    {'searchParameters': {'q': 'Tesla Inc.',
                          'gl': 'us',
                          'hl': 'en',
                          'num': 10,
                          'type': 'news',
                          'tbs': 'qdr:h'},
     'news': [{'title': 'Oklahoma Gov. Stitt sees growing foreign interest in '
                        'investments in ...',
               'link': 'https://www.reuters.com/world/us/oklahoma-gov-stitt-sees-growing-foreign-interest-investments-state-2023-05-04/',
               'snippet': 'T)), a battery supplier to electric vehicle maker Tesla '
                          'Inc (TSLA.O), said on Sunday it is considering building '
                          'a battery plant in Oklahoma, its third in...',
               'date': '53 mins ago',
               'source': 'Reuters',
               'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSSTcsXeenqmEKdiekvUgAmqIPR4nlAmgjTkBqLpza-lLfjX1CwB84MoNVj0Q&s',
               'position': 1},
              {'title': 'Ryder lanza solución llave en mano para vehículos '
                        'eléctricos en EU',
               'link': 'https://www.tyt.com.mx/nota/ryder-lanza-solucion-llave-en-mano-para-vehiculos-electricos-en-eu',
               'snippet': 'Ryder System Inc. presentó RyderElectric+ TM como su '
                          'nueva solución llave en mano ... Ryder también tiene '
                          'reservados los semirremolques Tesla y continúa...',
               'date': '56 mins ago',
               'source': 'Revista Transportes y Turismo',
               'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJhXTQQtjSUZf9YPM235WQhFU5_d7lEA76zB8DGwZfixcgf1_dhPJyKA1Nbw&s',
               'position': 2},
              {'title': '"I think people can get by with $999 million," Bernie '
                        'Sanders tells American Billionaires.',
               'link': 'https://thebharatexpressnews.com/i-think-people-can-get-by-with-999-million-bernie-sanders-tells-american-billionaires-heres-how-the-ultra-rich-can-pay-less-income-tax-than-you-legally/',
               'snippet': 'The report noted that in 2007 and 2011, Amazon.com Inc. '
                          'founder Jeff Bezos “did not pay a dime in federal ... '
                          'If you want to bet on Musk, check out Tesla.',
               'date': '11 mins ago',
               'source': 'THE BHARAT EXPRESS NEWS',
               'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR_X9qqSwVFBBdos2CK5ky5IWIE3aJPCQeRYR9O1Jz4t-MjaEYBuwK7AU3AJQ&s',
               'position': 3}]}
    

Some examples of the `tbs` parameter:

`qdr:h` (past hour)
`qdr:d` (past day)
`qdr:w` (past week)
`qdr:m` (past month)
`qdr:y` (past year)

You can specify intermediate time periods by adding a number:
`qdr:h12` (past 12 hours)
`qdr:d3` (past 3 days)
`qdr:w2` (past 2 weeks)
`qdr:m6` (past 6 months)
`qdr:m2` (past 2 years)

For all supported filters simply go to [Google Search](https://google.com), search for something, click on "Tools", add your date filter and check the URL for "tbs=".


## Searching for Google Places
We can also query Google Places using this wrapper. For example:


```python
search = GoogleSerperAPIWrapper(type="places")
results = search.results("Italian restaurants in Upper East Side")
pprint.pp(results)
```

    {'searchParameters': {'q': 'Italian restaurants in Upper East Side',
                          'gl': 'us',
                          'hl': 'en',
                          'num': 10,
                          'type': 'places'},
     'places': [{'position': 1,
                 'title': "L'Osteria",
                 'address': '1219 Lexington Ave',
                 'latitude': 40.777154599999996,
                 'longitude': -73.9571363,
                 'thumbnailUrl': 'https://lh5.googleusercontent.com/p/AF1QipNjU7BWEq_aYQANBCbX52Kb0lDpd_lFIx5onw40=w92-h92-n-k-no',
                 'rating': 4.7,
                 'ratingCount': 91,
                 'category': 'Italian'},
                {'position': 2,
                 'title': "Tony's Di Napoli",
                 'address': '1081 3rd Ave',
                 'latitude': 40.7643567,
                 'longitude': -73.9642373,
                 'thumbnailUrl': 'https://lh5.googleusercontent.com/p/AF1QipNbNv6jZkJ9nyVi60__8c1DQbe_eEbugRAhIYye=w92-h92-n-k-no',
                 'rating': 4.5,
                 'ratingCount': 2265,
                 'category': 'Italian'},
                {'position': 3,
                 'title': 'Caravaggio',
                 'address': '23 E 74th St',
                 'latitude': 40.773412799999996,
                 'longitude': -73.96473379999999,
                 'thumbnailUrl': 'https://lh5.googleusercontent.com/p/AF1QipPDGchokDvppoLfmVEo6X_bWd3Fz0HyxIHTEe9V=w92-h92-n-k-no',
                 'rating': 4.5,
                 'ratingCount': 276,
                 'category': 'Italian'},
                {'position': 4,
                 'title': 'Luna Rossa',
                 'address': '347 E 85th St',
                 'latitude': 40.776593999999996,
                 'longitude': -73.950351,
                 'thumbnailUrl': 'https://lh5.googleusercontent.com/p/AF1QipNPCpCPuqPAb1Mv6_fOP7cjb8Wu1rbqbk2sMBlh=w92-h92-n-k-no',
                 'rating': 4.5,
                 'ratingCount': 140,
                 'category': 'Italian'},
                {'position': 5,
                 'title': "Paola's",
                 'address': '1361 Lexington Ave',
                 'latitude': 40.7822019,
                 'longitude': -73.9534096,
                 'thumbnailUrl': 'https://lh5.googleusercontent.com/p/AF1QipPJr2Vcx-B6K-GNQa4koOTffggTePz8TKRTnWi3=w92-h92-n-k-no',
                 'rating': 4.5,
                 'ratingCount': 344,
                 'category': 'Italian'},
                {'position': 6,
                 'title': 'Come Prima',
                 'address': '903 Madison Ave',
                 'latitude': 40.772124999999996,
                 'longitude': -73.965012,
                 'thumbnailUrl': 'https://lh5.googleusercontent.com/p/AF1QipNrX19G0NVdtDyMovCQ-M-m0c_gLmIxrWDQAAbz=w92-h92-n-k-no',
                 'rating': 4.5,
                 'ratingCount': 176,
                 'category': 'Italian'},
                {'position': 7,
                 'title': 'Botte UES',
                 'address': '1606 1st Ave.',
                 'latitude': 40.7750785,
                 'longitude': -73.9504801,
                 'thumbnailUrl': 'https://lh5.googleusercontent.com/p/AF1QipPPN5GXxfH3NDacBc0Pt3uGAInd9OChS5isz9RF=w92-h92-n-k-no',
                 'rating': 4.4,
                 'ratingCount': 152,
                 'category': 'Italian'},
                {'position': 8,
                 'title': 'Piccola Cucina Uptown',
                 'address': '106 E 60th St',
                 'latitude': 40.7632468,
                 'longitude': -73.9689825,
                 'thumbnailUrl': 'https://lh5.googleusercontent.com/p/AF1QipPifIgzOCD5SjgzzqBzGkdZCBp0MQsK5k7M7znn=w92-h92-n-k-no',
                 'rating': 4.6,
                 'ratingCount': 941,
                 'category': 'Italian'},
                {'position': 9,
                 'title': 'Pinocchio Restaurant',
                 'address': '300 E 92nd St',
                 'latitude': 40.781453299999995,
                 'longitude': -73.9486788,
                 'thumbnailUrl': 'https://lh5.googleusercontent.com/p/AF1QipNtxlIyEEJHtDtFtTR9nB38S8A2VyMu-mVVz72A=w92-h92-n-k-no',
                 'rating': 4.5,
                 'ratingCount': 113,
                 'category': 'Italian'},
                {'position': 10,
                 'title': 'Barbaresco',
                 'address': '843 Lexington Ave #1',
                 'latitude': 40.7654332,
                 'longitude': -73.9656873,
                 'thumbnailUrl': 'https://lh5.googleusercontent.com/p/AF1QipMb9FbPuXF_r9g5QseOHmReejxSHgSahPMPJ9-8=w92-h92-n-k-no',
                 'rating': 4.3,
                 'ratingCount': 122,
                 'locationHint': 'In The Touraine',
                 'category': 'Italian'}]}
    
