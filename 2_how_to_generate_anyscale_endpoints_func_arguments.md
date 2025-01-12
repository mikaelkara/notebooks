# How to use Anyscale function calling with LLMs
Function calling extends the power capabilities of LLMs. It allolws you to format the output of an LLM response into a JSON object, which then can be fed down stream
to an actual function as a argument to process the response.

The notion of Anyscale function calling is not to far off from OpenAI documention that states the basic steps involved in function calling: 

1. Call the model with the user query and a set of functions defined in the functions parameter.
2. The model can choose to call one or more functions; if so, the content will be a stringified JSON object adhering to your custom schema (note: the model may hallucinate parameters).
3. Parse the string into JSON in your code, and call your function with the provided arguments if they exist.
4. Call the model again by appending the function response as a new message, and let the model summarize the results back to the user.

<img src="./images/gpt_function_calling.png">

Let's first demonstrate how we use this feature: specify a function and use the API to generate function arguments.

**Note**: 
To run any of these relevant notebooks you will need an account on Anyscale Endpoints and
OpenAI. Use the template enivironment files to create respective `.env` file for either 
Anyscale Endpoints or OpenAI.



```python
import warnings
import os
import json

import openai
from openai import OpenAI

from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
from typing import Dict, Any, List
```

Load our .env file with respective API keys and base url endpoints. Here you can either use OpenAI or Anyscale Endpoints.


```python
_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings('ignore')
openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
google_api_key = os.getenv("GOOGLE_API_KEY", "")
weather_api_key = os.getenv("WEATHER_API_KEY", "")

MODEL = os.getenv("MODEL")
print(f"Using MODEL={MODEL}; base={openai.api_base}")
```

    Using MODEL=mistralai/Mistral-7B-Instruct-v0.1; base=https://api.endpoints.anyscale.com/v1
    


```python
from openai import OpenAI

client = OpenAI(
    api_key = openai.api_key,
    base_url = openai.api_base
)
```

## Simple case: Example 1: Query the Anyscale endpoint with Mistral 

Since Anyscale endpoint supports JSON response format, let's use
Pydantic validation and base class for our response object


```python
class QueryResponse(BaseModel):
    finalist_team: str = Field(description="World Cup finalist team"),
    winning_team: str = Field(description="World Cup final wining team"),
    final_score: str = Field(description="Final score")
```

Let'send a basic query and see what result we get and its JSON format


```python
response = chat_completion = client.chat.completions.create(
    model = MODEL,
    # return the LLM response at our JSON pydantic object
    response_format = {
        "type": "json_object", 
        "schema": QueryResponse.schema_json()
    },
    messages = [
        {"role": "system", 
         "content": "You are the FIFA World Cup Soccer assistant designed to output JSON response for queries"},
        {"role": "user", 
         "content": """What two national teams played in the FIFA World Cup Soccer 1998 Finals played 
                        in Paris, France, and which team won and which team lost?"""
        }
    ],
    temperature=0.7
)
```


```python
result = response.choices[0].message.content
```

#### Convert the JSON response into a dictonary
This output matches JSON format schema defined above using Pydantic `QueryResponse` class


```python
import json

result = response.choices[0].message.content
# Convert string into a dictionary
json_dict = json.loads(result)

for k, v in json_dict.items():
    print(f"{k}:{v}")
```

    winning_team:France
    final_score:3-0
    finalist_team:Argentina
    


```python
# Convert dictionary into a printable JSON string
print(json.dumps(json_dict, indent=2))
```

    {
      "winning_team": "France",
      "final_score": "3-0",
      "finalist_team": "Argentina"
    }
    

One advantage of returning a JSON object as a result is ease of processing by a function
down stream, after querying an LLM, by the application.

## Example 2: Query the Anyscale endpoint with Mistral 



```python
from typing import List
class MathResponse(BaseModel):
    prime_numbers: List[int] = Field(description="List of prime numbers between 27 and 1027")
    sum: int = Field(description="Sum of the all the prime numbers between 27 and 1027")
```


```python
user_content = """
Generate a list of randomly generated integer prime numbers 
between 27 and 1027. Shuffle the generated list of prime numbers
so they are out of order. All the numbers generated should be unquie and
random. 
"""
```


```python
response = chat_completion = client.chat.completions.create(
    model=MODEL,
    response_format={
        "type": "json_object", 
        "schema": MathResponse.schema_json()
    },
    messages=[
        {"role": "system", 
         "content": """You are Math tutor who can answer simple math problems and return 
         response in JSON format"""
        },
        {"role": "user", 
        "content": user_content
        }
    ],
    temperature=0.7
)
```


```python
import json

result = response.choices[0].message.content
# Convert string into a dictionary
json_schema = json.loads(result)

for k, v in json_schema.items():
    print(f"{k}:{v}")
```

    sum:100
    prime_numbers:[19, 71, 73, 113, 181, 227, 233, 239, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439]
    


```python
# Convert dictionary into a printable JSON string
print(json.dumps(json_schema, indent=2))
```

    {
      "sum": 100,
      "prime_numbers": [
        19,
        71,
        73,
        113,
        181,
        227,
        233,
        239,
        313,
        317,
        331,
        337,
        347,
        349,
        353,
        359,
        367,
        373,
        379,
        383,
        389,
        397,
        401,
        409,
        419,
        421,
        431,
        433,
        439
      ]
    }
    

## Example 3: Query the Anyscale endpoint with Mistral 



```python
from typing import Tuple
class FractionResponse(BaseModel):
    fractions: List[Tuple[int, int]] = Field(description="List of unique fractrions between 1 and 7 as Python tuples")
    sum: int = Field(description="Sum of the all unique fractrions between 1 and 7")
    common_denominator: int = Field(description="The common denominator among the unique fractions between 1 and 7")
```


```python
response = chat_completion = client.chat.completions.create(
    model=MODEL,
    response_format={
        "type": "json_object", 
        "schema": FractionResponse.schema_json()
    },
    messages=[
        {"role": "system", 
         "content": """You are Math tutor who can answer simple math problems and return 
         reponse in JSON format"""
        },
        {"role": "user", 
        "content": """Generate a list of all unique fractions between between 1 and 7 as Python tuples,  
        add the tuple fractions in the list, and find their common denominator"""
        }
    ],
    temperature=0.7
)
```


```python
import json

result = response.choices[0].message.content
# Convert string into a dictionary
json_schema = json.loads(result)

for k, v in json_schema.items():
    print(f"{k}:{v}")
```

    sum:0
    fractions:[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [0, 7], [1, 6], [2, 5], [3, 4], [4, 3], [5, 2], [6, 1], [7, 0]]
    common_denominator:42
    


```python
# Convert dictionary into a printable JSON string
print(json.dumps(json_schema, indent=2))
```

    {
      "sum": 0,
      "fractions": [
        [
          0,
          1
        ],
        [
          1,
          2
        ],
        [
          2,
          3
        ],
        [
          3,
          4
        ],
        [
          4,
          5
        ],
        [
          5,
          6
        ],
        [
          6,
          7
        ],
        [
          0,
          7
        ],
        [
          1,
          6
        ],
        [
          2,
          5
        ],
        [
          3,
          4
        ],
        [
          4,
          3
        ],
        [
          5,
          2
        ],
        [
          6,
          1
        ],
        [
          7,
          0
        ]
      ],
      "common_denominator": 42
    }
    

## How to generate function arguments
For Anyscale endpoints, the idea of function calling is not that different from 
OpenAI's. Basically, according to [Anyscale function calling blog](https://www.anyscale.com/blog/anyscale-endpoints-json-mode-and-function-calling-features), it boils down to simple steps:
 1. You send a query with functions and parameters to the LLM.
 2. The LLM decides to either use a function or not.
 3. If not using a function, it replies in plain text, providing an answer or asking for more information.
 4. If using a function, it recommends an API and gives usage instructions in JSON.
 5. You execute the API in your application.
 6. Send the API's results back to the LLM.
 7. The LLM analyzes these results and guides the next steps.

#### Example 1: Extract the generated arguments

Let's process generated arguments to plot a map of cities where we hold 
Ray meetups. The generated satelite coordinates, returned as a JSON object
by the LLM, are fed into a function to generate an HTML file and map markers of
each city.

The idea is here is to nudge the LLM to generate JSON object, which can be easily converted into a Python dictionary as an argument to a function downstream to create the HTML and render a map.


```python
tools = [ 
        {
            "type": "function",
            "function": {
                "name": "generate_ray_meetup_map",
                "description": "Generate HTML map for global cities where Ray meetups are hosted",
                "parameters": {
                    "type": "object",
                    "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city name e.g., San Francisco, CA",
                                    },
                                "latitude": {
                                    "type": "string",
                                    "description": "Latitude satelite coordinates",
                                    },
                                "longitude": {
                                    "type": "string",
                                    "description": "Longitude satelite coordinates",
                                    },
                                }
        
                            }
                },
                "required": ["location, latitude, longitude"]
        }
]
```


```python
def generate_city_map_completion(clnt: object, model: str, user_content:str) -> object:
    
    chat_completion = clnt.chat.completions.create(
        model=model,
        messages=[ {"role": "system", "content": f"You are a helpful assistant. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."},
                   {"role": "user", "content": user_content}],
        tools=tools,
        tool_choice="auto",
        temperature=0.7)
        
    response = chat_completion.choices[0].message
    return response
```


```python
user_content = """Generate satelite coordinates latitude and longitude for a location in San Francisco, where Ray meetup is hosted 
at Anyscale Headquaters on 55 Hawthorne Street, San Francisco, CA 94105
"""
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
response_1 = generate_city_map_completion(client, MODEL, user_content)
print(response_1)
```

    Using Endpoints: https://api.endpoints.anyscale.com/v1 ...
    
    ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_a0a88634ed3942e887cc78be9283d6f4', function=Function(arguments='{"location": "San Francisco, CA", "latitude": "55.340978", "longitude": "-122.404365"}', name='generate_ray_meetup_map'), type='function')])
    


```python
json_arguments = response_1.dict()
json_arguments
```




    {'content': None,
     'role': 'assistant',
     'function_call': None,
     'tool_calls': [{'id': 'call_a0a88634ed3942e887cc78be9283d6f4',
       'function': {'arguments': '{"location": "San Francisco, CA", "latitude": "55.340978", "longitude": "-122.404365"}',
        'name': 'generate_ray_meetup_map'},
       'type': 'function'}]}




```python
# Extract specific items from the dict
func_name = json_arguments['tool_calls'][0]['function']['name']
print(f"Function name: {func_name}")
```

    Function name: generate_ray_meetup_map
    


```python
funcs = json_arguments['tool_calls'][0]['function']['arguments']
funcs_args = json.loads(funcs)
print(f"Arguments: {funcs_args}")
```

    Arguments: {'location': 'San Francisco, CA', 'latitude': '55.340978', 'longitude': '-122.404365'}
    


```python
html_file_path = './world_map_nb_func_anyscale_with_cities.html'
```


```python
import folium

def generate_ray_meetup_map(coordinates: Dict[str, Any]) -> None:
    # Create a base map
    m = folium.Map(location=[20,0], tiles="OpenStreetMap", zoom_start=2)
    # Adding markers for each city
    folium.Marker([coordinates["latitude"], coordinates["longitude"]], popup=coordinates["location"]).add_to(m)
    # Display the map
    m.save(html_file_path)
```

#### Invoke the function from within our notebook.
This is our downstream function being invoked with the extract arguments
from the JSON response generated by the LLM.


```python
from IPython.display import IFrame

# Assuming the HTML file is named 'example.html' and located in the same directory as the Jupyter Notebook
generate_ray_meetup_map(funcs_args)

# Display the HTML file in the Jupyter Notebook
IFrame(src=html_file_path, width=700, height=400)
```





<iframe
    width="700"
    height="400"
    src="./world_map_nb_func_anyscale_with_cities.html"
    frameborder="0"
    allowfullscreen

></iframe>




#### Example 2: Generate function arguments


```python
def get_weather_data_for_cities(params:Dict[Any, Any]=None,
                    api_base:str="http://api.weatherstack.com/current") -> Dict[str, str]:
    """
    Retrieves weather data from the OpenWeatherMap API for cities
    """
    import requests
    url = f"{api_base}"
    response = requests.get(url, params=params)
    return response.json()
```


```python
tools_2 = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_data_for_cities",
            "description": "Get the current weather forecast in the cities",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {
                                "type": "string",
                                "description": "The list of city names e.g., San Francisco, New York, London"
                            }
                    }
                }
             }
        },
        "required": ["query"]
    }
]
```


```python
def generate_weather_forecast_completion(clnt: object, model: str, user_content:str) -> object:
    chat_completion = clnt.chat.completions.create(
        model=model,
        messages=[ {"role": "system", "content": f"You are a helpful weather assistant. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."},
                   {"role": "user", "content": user_content}],
        tools=tools_2,
        tool_choice="auto",
        temperature=0.7)
        
    response = chat_completion.choices[0].message
    return response
```


```python
user_content = """
                Generate weather forecast and temperatures for 
                London, New York, and San Franciso for today.
                """
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
print(f"Using model: {MODEL} ...\n")
weather_response = generate_weather_forecast_completion(client, MODEL, user_content)
print(weather_response)
```

    Using Endpoints: https://api.endpoints.anyscale.com/v1 ...
    
    Using model: mistralai/Mistral-7B-Instruct-v0.1 ...
    
    ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_c9220edf10014f6f90230ffd982ff599', function=Function(arguments='{"query": ["London", "New York", "San Francisco"]}', name='get_weather_data_for_cities'), type='function')])
    


```python
json_arguments = weather_response.dict()
json_arguments
```




    {'content': None,
     'role': 'assistant',
     'function_call': None,
     'tool_calls': [{'id': 'call_c9220edf10014f6f90230ffd982ff599',
       'function': {'arguments': '{"query": ["London", "New York", "San Francisco"]}',
        'name': 'get_weather_data_for_cities'},
       'type': 'function'}]}




```python
# Extract specific items from the dict
func_name = json_arguments['tool_calls'][0]['function']['name']
print(f"Function name: {func_name}")
```

    Function name: get_weather_data_for_cities
    


```python
funcs = json_arguments['tool_calls'][0]['function']['arguments']
funcs_args = json.loads(funcs)
print(f"Arguments: {funcs_args}")
```

    Arguments: {'query': ['London', 'New York', 'San Francisco']}
    

#### Invoke the function from within our notebook.
This is our downstream function being invoked with the extracted arguments
from the JSON response generated by the LLM.


```python
# Iterate over cities since we have a list
cities = funcs_args['query']
weather_statements = []
for city in cities:
    params = {'query': city,
              'access_key': weather_api_key,
              'units': "f"
             }
    weather_data = get_weather_data_for_cities(params)
    c = f"The weather and forecast in the {city} is {weather_data['current']['temperature']} F and {weather_data['current']['weather_descriptions'][0]}"
    weather_statements.append(c) 
weather_statements
```




    ['The weather and forecast in the London is 55 F and Overcast',
     'The weather and forecast in the New York is 52 F and Overcast',
     'The weather and forecast in the San Francisco is 57 F and Partly cloudy']




```python
paragraph = ' '.join(weather_statements)
```

### Let LLM generate the response for the orignal request


```python
messages=[ {"role": "system", "content": f"You are a helpful assistant for weather forecast. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."},
            {"role": "user", "content": user_content}]
messages.append(json_arguments)
messages
```




    [{'role': 'system',
      'content': "You are a helpful assistant for weather forecast. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."},
     {'role': 'user',
      'content': '\n                Generate weather forecast and temperatures for \n                London, New York, and San Franciso for today.\n                '},
     {'content': None,
      'role': 'assistant',
      'function_call': None,
      'tool_calls': [{'id': 'call_c9220edf10014f6f90230ffd982ff599',
        'function': {'arguments': '{"query": ["London", "New York", "San Francisco"]}',
         'name': 'get_weather_data_for_cities'},
        'type': 'function'}]}]




```python
messages.append(
    {"role": "tool",
     "tool_call_id": json_arguments['tool_calls'][0]['id'],
     "name": json_arguments['tool_calls'][0]['function']["name"],
     "content": paragraph}
)
messages
```




    [{'role': 'system',
      'content': "You are a helpful assistant for weather forecast. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."},
     {'role': 'user',
      'content': '\n                Generate weather forecast and temperatures for \n                London, New York, and San Franciso for today.\n                '},
     {'content': None,
      'role': 'assistant',
      'function_call': None,
      'tool_calls': [{'id': 'call_c9220edf10014f6f90230ffd982ff599',
        'function': {'arguments': '{"query": ["London", "New York", "San Francisco"]}',
         'name': 'get_weather_data_for_cities'},
        'type': 'function'}]},
     {'role': 'tool',
      'tool_call_id': 'call_c9220edf10014f6f90230ffd982ff599',
      'name': 'get_weather_data_for_cities',
      'content': 'The weather and forecast in the London is 55 F and Overcast The weather and forecast in the New York is 52 F and Overcast The weather and forecast in the San Francisco is 57 F and Partly cloudy'}]



Send to LLM for final completion


```python
chat_completion = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools_2,
    tool_choice="auto",
    temperature=0.7)
        
response = chat_completion.choices[0].message
print(response.content)
```

     The response indicates that the current weather and forecast for London is 55 F and Overcast, New York is 52 F and Overcast, and San Francisco is 57 F and Partly cloudy. This information is based on the data retrieved from the API call at 10:25:00 AM on April 15th, 2023.
    
