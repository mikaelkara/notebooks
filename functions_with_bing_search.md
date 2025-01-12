# Function calling with Bing Search

In this notebook, we'll show how you can use the [Bing Search APIs](https://www.microsoft.com/bing/apis/llm) and [function calling](https://learn.microsoft.com/azure/ai-services/openai/how-to/function-calling?tabs=python) to ground Azure OpenAI models on data from the web. This is a great way to give the model access to up to date data from the web.

You'll need to create a [Bing Search resouce](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/create-bing-search-service-resource) before you begin.


```python
import json
import os
import requests
from openai import AzureOpenAI

# Load config values
with open(r'config.json') as config_file:
    config_details = json.load(config_file)
    


client = AzureOpenAI(
    azure_endpoint=config_details["AZURE_OPENAI_ENDPOINT"],  # The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
    api_key=os.getenv("AZURE_OPENAI_KEY"),  # The API key for your Azure OpenAI resource.
    api_version=config_details["OPENAI_API_VERSION"],  # This version supports function calling
)

model_name = config_details['MODEL_NAME'] # You need to ensure the version of the model you are using supports the function calling feature

bing_search_subscription_key = config_details['BING_SEARCH_SUBSCRIPTION_KEY']
bing_search_url = "https://api.bing.microsoft.com/v7.0/search"
```

## 1.0 Define a function to call the Bing Search APIs

 To learn more about using the Bing Search APIs with Azure OpenAI, see [Bing Search APIs, with your LLM](https://learn.microsoft.com/bing/search-apis/bing-web-search/use-display-requirements-llm).


```python
def search(query):
    headers = {"Ocp-Apim-Subscription-Key": bing_search_subscription_key}
    params = {"q": query, "textDecorations": False }
    response = requests.get(bing_search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    output = []

    for result in search_results['webPages']['value']:
        output.append({
            'title': result['name'],
            'link': result['url'],
            'snippet': result['snippet']
        })

    return json.dumps(output)

```


```python
search("where will the 2032 olymbics be held?")
```




    '[{"title": "2032 Summer Olympics - Wikipedia", "link": "https://en.wikipedia.org/wiki/2032_Summer_Olympics", "snippet": "The 2032 Summer Olympics, officially known as the Games of the XXXV Olympiad and also known as Brisbane 2032 ( Yagara: Meanjin 2032 ), [1] is an upcoming international multi-sport event scheduled to take place between 23 July to 8 August 2032, in Brisbane, Queensland, Australia. [2]"}, {"title": "Venues of the 2032 Summer Olympics and Paralympics", "link": "https://en.wikipedia.org/wiki/Venues_of_the_2032_Summer_Olympics_and_Paralympics", "snippet": "Stadium Australia North Queensland Stadium Barlow Park Toowoomba Sports Ground class=notpageimage| 2032 Olympic and Paralympic venues outside South East Queensland Sporting venues The Gabba Brisbane Convention & Exhibition Centre South Bank Piazza Anna Meares Velodrome Sleeman BMX SuperCross Track Brisbane Aquatics Centre Barrambin / Victoria Park"}, {"title": "The Next Olympics Location: Every Host City Through 2032 | Time", "link": "https://time.com/5063566/next-olympics-location/", "snippet": "Mountain events will take place at two locations about 50 and 100 miles outside of Beijing, with natural snowfall topping out at one of them at only five centimeters on average. 2024 Summer..."}, {"title": "Here\'s where the 2024, 2026 2028, 2032 Olympic Games will be - The Scotsman", "link": "https://www.scotsman.com/sport/other-sport/next-olympics-olympic-games-hosts-britain-london-who-latest-news-3321075", "snippet": "Looking ahead to 2032, Brisbane in Queensland, Australia, has been announced as the winning host location for the 2032 Olympic Games \\u2013 which will mark the 34th Olympic Games since records..."}, {"title": "Where is the next Olympics? Explaining where the Summer and Winter ...", "link": "https://www.cbssports.com/olympics/news/where-is-the-next-olympics-explaining-where-the-summer-and-winter-games-will-be-held-through-2032/", "snippet": "The opening and closing ceremonies will take place in SoFi Stadium, home of the Los Angeles Rams and Los Angeles Chargers and site of Super Bowl LVI. The Los Angeles Coliseum will once again hold..."}, {"title": "Brisbane 2032 Olympic venues announced | Austadiums", "link": "https://www.austadiums.com/news/921/brisbane-2032-olympic-games-venues-revealed", "snippet": "The Brisbane 2032 Masterplan includes 32 venues within South-East Queensland for the 28 Olympic sports, located in three primary zones. Not only will Brisbane\\u2019s Olympics expand to the entire south-east Queensland, football will also be played in North Queensland as well as Sydney and Melbourne."}, {"title": "Brisbane 2032 Summer Olympics - Summer Olympic Games in Australia", "link": "https://olympics.com/en/olympic-games/brisbane-2032", "snippet": "Brisbane 2032 23 July - 8 August 3185 days Australia Official website Brisbane 2032 Annual Report 2022-23 Brisbane 2032 | Olympic Games Countdown Begins: Brisbane Celebrates Nine-Year Mark to 2032 Summer Olympics Brisbane 2032 | Olympic Games 01:01 Brisbane 2032 Olympics Marks Nine-Year Milestone with Grand Celebrations"}, {"title": "2032 Games: Brisbane confirmed as Olympic and Paralympic host", "link": "https://www.bbc.co.uk/sport/olympics/57912026", "snippet": "Brisbane will host the 2032 Olympic and Paralympic Games after being approved by the International Olympic Committee. The Australian city was named the preferred bidder before being proposed by..."}, {"title": "2032 Olympics: Brisbane proposed as host by International Olympic ... - BBC", "link": "https://www.bbc.com/sport/olympics/57432349", "snippet": "Australian city Brisbane has moved a step closer to being named the host for the 2032 Olympic Games. ... The delayed 2020 Olympics will be held in Tokyo, Japan in the summer, with Paris in France ..."}, {"title": "Brisbane 2032 - Olympics.com", "link": "https://olympics.com/ioc/brisbane-2032", "snippet": "Olympic Games Brisbane 2032. Find out all about the athletes, sports, schedules, venues, mascot and much more. Learn more. New approach to future host elections. The revolutionary new approach to electing hosts for Olympic Games and youth Olympic Games results in significant cost savings for potential hosts, as well as more sustainable projects ..."}]'



## 2.0 Test function calling


```python
system_message = """You are an assistant designed to help people answer questions.

You have access to query the web using Bing Search. You should call bing search whenever a question requires up to date information or could benefit from web data.
"""

messages = [{"role": "system", "content": system_message},
            {"role": "user", "content": "How tall is mount rainier?"}]

                
tools = [  
    {
        "type": "function",
        "function": {
            "name": "search_bing",
            "description": "Searches bing to get up to date information from the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        }
    }
    
]

response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

print(response.choices[0].message)
```

    {
      "role": "assistant",
      "function_call": {
        "name": "search_bing",
        "arguments": "{\n  \"query\": \"height of mount rainier\"\n}"
      }
    }
    

## 3.0 Get things running end to end


```python
def run_multiturn_conversation(messages, functions, available_functions, deployment_name):
    # Step 1: send the conversation and available functions to GPT

    response = client.chat.completions.create(
        messages=messages,
        tools=tools,
        tool_choice="auto",
        model=model_name,
        temperature=0,
    )

    # Step 2: check if GPT wanted to call a function
    while response.choices[0].finish_reason == "tool_calls":
        response_message = response.choices[0].message
        print("Recommended Function call:")
        print(response_message.tool_calls[0])
        print()
        
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        
        function_name = response_message.tool_calls[0].function.name
        
        # verify function exists
        if function_name not in available_functions:
            return "Function " + function_name + " does not exist"
        function_to_call = available_functions[function_name]  
        
        function_args = json.loads(response_message.tool_calls[0].function.arguments)
        function_response = function_to_call(**function_args)
        
        print("Output of function call:")
        print(function_response)
        print()
        
        # Step 4: send the info on the function call and function response to GPT
        
        # adding assistant response to messages
        messages.append(
            {
                "role": response_message.role,
                "function_call": {
                    "name": response_message.tool_calls[0].function.name,
                    "arguments": response_message.tool_calls[0].function.arguments,
                },
                "content": None
            }
        )

        # adding function response to messages
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response

        print("Messages in next request:")
        for message in messages:
            print(message)
        print()

        response = client.chat.completions.create(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            model=model_name,
            temperature=0,
        )  # get a new response from GPT where it can see the function response

    return response
```


```python
system_message = """You are an assistant designed to help people answer questions.

You have access to query the web using Bing Search. You should call bing search whenever a question requires up to date information or could benefit from web data.
"""

messages = [{"role": "system", "content": system_message},
            {"role": "user", "content": "How tall is mount rainier?"}]


available_functions = {'search_bing': search}

result = run_multiturn_conversation(messages, tools, available_functions)

print("Final response:")
print(result.choices[0].message)
```

    Recommended Function call:
    {
      "name": "search_bing",
      "arguments": "{\n  \"query\": \"height of Mount Rainier\"\n}"
    }
    
    Output of function call:
    [{"title": "Mount Rainier - Wikipedia", "link": "https://en.wikipedia.org/wiki/Mount_Rainier", "snippet": "Coordinates: 46\u00b051\u203210\u2033N 121\u00b045\u203238\u2033W Mount Rainier seen from the International Space Station Mount Rainier ( / re\u026a\u02c8n\u026a\u0259r / ray-NEER ), also known as Tahoma, is a large active stratovolcano in the Cascade Range of the Pacific Northwest in the United States."}, {"title": "Geology and History Summary for Mount Rainier - USGS.gov", "link": "https://www.usgs.gov/volcanoes/mount-rainier/geology-and-history-summary-mount-rainier", "snippet": "Public domain.) Mount Rainier is an active volcano of the Cascade Range in Washington State, 50-70 km (30-44 mi) southeast of the Seattle\u2013Tacoma metropolitan area. Volcanism occurs at Mount Rainier and other Cascades arc volcanoes because of the subduction of the Juan de Fuca Plate off the western coast of North America."}, {"title": "Mount Rainier | U.S. Geological Survey - USGS.gov", "link": "https://www.usgs.gov/volcanoes/mount-rainier", "snippet": "Mount Rainier, the highest peak in the Cascade Range at 4,392m (14,410 ft), forms a dramatic backdrop to the Puget Sound region. Summary During an eruption 5,600 years ago the once-higher edifice of Mount Rainier collapsed to form a large crater open to the northeast much like that at Mount St. Helens after 1980."}, {"title": "Mount Rainier | National Park, History, Eruptions, & Map", "link": "https://www.britannica.com/place/Mount-Rainier", "snippet": "Mount Rainier, highest mountain (14,410 feet [4,392 meters]) in the state of Washington, U.S., and in the Cascade Range. It lies about 40 miles (64 km) southeast of the city of Tacoma, within Mount Rainier National Park. A dormant volcano, it last erupted about 150 years ago."}, {"title": "Mount Rainier National Park - Wikipedia", "link": "https://en.wikipedia.org/wiki/Mount_Rainier_National_Park", "snippet": "The highest point in the Cascade Range, Mount Rainier is surrounded by valleys, waterfalls, subalpine meadows, and 91,000 acres (142.2 sq mi; 368.3 km 2) of old-growth forest. [4] More than 25 glaciers descend the flanks of the volcano, which is often shrouded in clouds that dump enormous amounts of rain and snow."}, {"title": "Mount Rainier National Park (U.S. National Park Service)", "link": "https://www.nps.gov/mora/index.htm", "snippet": "Ascending to 14,410 feet above sea level, Mount Rainier stands as an icon in the Washington landscape. An active volcano, Mount Rainier is the most glaciated peak in the contiguous U.S.A., spawning five major rivers. Subalpine wildflower meadows ring the icy volcano while ancient forest cloaks Mount Rainier\u2019s lower slopes."}, {"title": "Mount Rainier Geology | U.S. Geological Survey - USGS.gov", "link": "https://www.usgs.gov/geology-and-ecology-of-national-parks/mount-rainier-geology", "snippet": "Mt. Rainier is an active volcano, rising to over 14,000 feet southeast of Seattle. Return to Rainier main page Sources/Usage: Public Domain. A distant view of Mount Rainier volcano over Puyallup Valley, near Orting, Washington."}, {"title": "Mount Rainier is a special place - U.S. National Park Service", "link": "https://www.nps.gov/mora/learn/management/what-s-special.htm", "snippet": "At a height of 14,410 feet, Mount Rainier is the highest volcanic peak in the contiguous United States. It has the largest alpine glacial system outside of Alaska and the world's largest volcanic glacier cave system (in the summit crater)."}, {"title": "Frequently Asked Questions - Mount Rainier National Park (U.S. National ...", "link": "https://www.nps.gov/mora/faqs.htm", "snippet": "Mount Rainier National Park encompasses 236,380.89 acres or 369.34 square miles within the legislative park boundary, with an additional 140 acres lying outside the boundary. Of that amount, 228,480 acres (97% of the park) has been designated by Congress as Wilderness. The park is also a National Historic Landmark District."}]
    
    Messages in next request:
    {'role': 'system', 'content': 'You are an assistant designed to help people answer questions.\n\nYou have access to query the web using Bing Search. You should call bing search whenever a question requires up to date information or could benefit from web data.\n'}
    {'role': 'user', 'content': 'How tall is mount rainier?'}
    {'role': 'assistant', 'function_call': {'name': 'search_bing', 'arguments': '{\n  "query": "height of Mount Rainier"\n}'}, 'content': None}
    {'role': 'function', 'name': 'search_bing', 'content': '[{"title": "Mount Rainier - Wikipedia", "link": "https://en.wikipedia.org/wiki/Mount_Rainier", "snippet": "Coordinates: 46\\u00b051\\u203210\\u2033N 121\\u00b045\\u203238\\u2033W Mount Rainier seen from the International Space Station Mount Rainier ( / re\\u026a\\u02c8n\\u026a\\u0259r / ray-NEER ), also known as Tahoma, is a large active stratovolcano in the Cascade Range of the Pacific Northwest in the United States."}, {"title": "Geology and History Summary for Mount Rainier - USGS.gov", "link": "https://www.usgs.gov/volcanoes/mount-rainier/geology-and-history-summary-mount-rainier", "snippet": "Public domain.) Mount Rainier is an active volcano of the Cascade Range in Washington State, 50-70 km (30-44 mi) southeast of the Seattle\\u2013Tacoma metropolitan area. Volcanism occurs at Mount Rainier and other Cascades arc volcanoes because of the subduction of the Juan de Fuca Plate off the western coast of North America."}, {"title": "Mount Rainier | U.S. Geological Survey - USGS.gov", "link": "https://www.usgs.gov/volcanoes/mount-rainier", "snippet": "Mount Rainier, the highest peak in the Cascade Range at 4,392m (14,410 ft), forms a dramatic backdrop to the Puget Sound region. Summary During an eruption 5,600 years ago the once-higher edifice of Mount Rainier collapsed to form a large crater open to the northeast much like that at Mount St. Helens after 1980."}, {"title": "Mount Rainier | National Park, History, Eruptions, & Map", "link": "https://www.britannica.com/place/Mount-Rainier", "snippet": "Mount Rainier, highest mountain (14,410 feet [4,392 meters]) in the state of Washington, U.S., and in the Cascade Range. It lies about 40 miles (64 km) southeast of the city of Tacoma, within Mount Rainier National Park. A dormant volcano, it last erupted about 150 years ago."}, {"title": "Mount Rainier National Park - Wikipedia", "link": "https://en.wikipedia.org/wiki/Mount_Rainier_National_Park", "snippet": "The highest point in the Cascade Range, Mount Rainier is surrounded by valleys, waterfalls, subalpine meadows, and 91,000 acres (142.2 sq mi; 368.3 km 2) of old-growth forest. [4] More than 25 glaciers descend the flanks of the volcano, which is often shrouded in clouds that dump enormous amounts of rain and snow."}, {"title": "Mount Rainier National Park (U.S. National Park Service)", "link": "https://www.nps.gov/mora/index.htm", "snippet": "Ascending to 14,410 feet above sea level, Mount Rainier stands as an icon in the Washington landscape. An active volcano, Mount Rainier is the most glaciated peak in the contiguous U.S.A., spawning five major rivers. Subalpine wildflower meadows ring the icy volcano while ancient forest cloaks Mount Rainier\\u2019s lower slopes."}, {"title": "Mount Rainier Geology | U.S. Geological Survey - USGS.gov", "link": "https://www.usgs.gov/geology-and-ecology-of-national-parks/mount-rainier-geology", "snippet": "Mt. Rainier is an active volcano, rising to over 14,000 feet southeast of Seattle. Return to Rainier main page Sources/Usage: Public Domain. A distant view of Mount Rainier volcano over Puyallup Valley, near Orting, Washington."}, {"title": "Mount Rainier is a special place - U.S. National Park Service", "link": "https://www.nps.gov/mora/learn/management/what-s-special.htm", "snippet": "At a height of 14,410 feet, Mount Rainier is the highest volcanic peak in the contiguous United States. It has the largest alpine glacial system outside of Alaska and the world\'s largest volcanic glacier cave system (in the summit crater)."}, {"title": "Frequently Asked Questions - Mount Rainier National Park (U.S. National ...", "link": "https://www.nps.gov/mora/faqs.htm", "snippet": "Mount Rainier National Park encompasses 236,380.89 acres or 369.34 square miles within the legislative park boundary, with an additional 140 acres lying outside the boundary. Of that amount, 228,480 acres (97% of the park) has been designated by Congress as Wilderness. The park is also a National Historic Landmark District."}]'}
    
    Final response:
    Mount Rainier, also known as Tahoma, is the highest peak in the Cascade Range and is located in Washington State, United States. It stands at a height of 14,410 feet (4,392 meters) above sea level. Mount Rainier is an active stratovolcano and is surrounded by valleys, waterfalls, subalpine meadows, and old-growth forests. It is also the most glaciated peak in the contiguous United States, with more than 25 glaciers descending its flanks.
    

## Next steps

The example above shows a simple pattern for how you can use function calling to ground Azure OpenAI models on data from the web. Here are some ideas for how you could extend this example:

- Teach the model to cite its sources using prompt engineering
- Define a second function to click into the top search result and extract relevant details from the page. To limit the length of text from the website, you could consider using a separate prompt to summarize the text relevant to the user's query before adding it to the conversation history
- Integrate your own data sources using additional functions
