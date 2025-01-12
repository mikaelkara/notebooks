# How to use OpenAI function calling with LLMs
Function calling extends the power capabilities of LLMs. It allolws you to format the output of an LLM response into a JSON object, which then can be fed down stream to an actual function as a argument to process and generate results or response.

OpenAI documention states the basic steps involved in function calling: 

1. Call the model with the user query and a set of functions defined in the functions parameter.
2. The model can choose to call one or more functions; if so, the content will be a stringified JSON object adhering to your custom schema 
3. Parse (or covert) the string into JSON in your code, and call your function with the provided arguments if they required.
4. Call the model again by appending the function response as a new message, and let the model summarize the results back to the user.

<img src="./images/gpt_function_calling.png">

Let's first demonstrate how we use this feature. Let's first specify a function and use the API to generate function arguments.

**Note**: 
To run any of these relevant notebooks you will need an account on Anyscale Endpoints and
OpenAI. Use the template enivironment files to create respective `.env` file for either 
Anyscale Endpoints or OpenAI.

## How to generate function arguments



```python
import warnings
import os

import openai
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
```

Load our .env file with respective API keys and base url endpoints. Here you can either use OpenAI or Anyscale Endpoints.


```python
_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings('ignore')
openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("MODEL")
print(f"Using MODEL={MODEL}; base={openai.api_base}")
```

    Using MODEL=gpt-4-turbo-preview; base=https://api.openai.com/v1
    

Define some utitilty functions


```python
from openai import OpenAI

client = OpenAI(
    api_key = openai.api_key,
    base_url = openai.api_base
)
```


```python
def add_prime_number_commpletion(clnt: object, model: str, user_content:str) -> object:
    chat_completion = clnt.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": user_content}],
    functions = [{
            "name": "add_prime_numbers",
            "description": "Add a list of integer prime numbers between 27 and 1027",
            "parameters": {
                "type": "object",
                "properties": {
                    "prime_numbers": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "description": "A list of  prime numbers"
                        },
                        "description": "List of prime numbers to be added"
                    }
                },
                "required": ["prime_numbers"]
            }
        }],
        function_call={"name": "add_prime_numbers"}
    )
    response = chat_completion.choices[0].message
    return response
```


```python
# Define a function which will be fed a dictionary with a key
# holding a list of randomly generated prime numbers between
# 27 and 1027
from typing import List, Dict
def add_prime_numbers(p_numbers: Dict[str, List[int]]) -> int:
    return sum(p_numbers["prime_numbers"])
```

Define the functions argument and function specification as part of the message



```python
user_content = """
Generate a list of randomly generated integer prime numbers 
between 27 and 1027. Shuffle the generated list of prime numbers
so they are out of order. All the numbers generated should be unquie and
random.
"""
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
original_response = add_prime_number_commpletion(client, MODEL, user_content)
print(original_response)
```

    Using Endpoints: https://api.openai.com/v1 ...
    
    ChatCompletionMessage(content=None, role='assistant', function_call=FunctionCall(arguments='{"prime_numbers":[29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021]}', name='add_prime_numbers'), tool_calls=None)
    

#### Example 1: Generate a list of random prime numbers and add them
Let's process generated arguments to compute the sum of random
prime numbers between 27 and 1027. These is a list fed into a function to compute its sum.

The idea is here is to nudge the LLM to generate JSON object, which can be easily converted into a Python dictionary as an  argument to a function downstream to compute the sum.

We can convert the resonse into a dictionary


```python
import json
json_arguments = original_response.dict()
json_arguments
```




    {'content': None,
     'role': 'assistant',
     'function_call': {'arguments': '{"prime_numbers":[29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021]}',
      'name': 'add_prime_numbers'},
     'tool_calls': None}




```python
# Extract specific items from the dict
func_name = json_arguments['function_call']['name']
print(f"Function name: {func_name}")
```

    Function name: add_prime_numbers
    


```python
funcs = json_arguments['function_call']['arguments']
funcs_args = json.loads(funcs)
print(f"Arguments: {funcs_args}")
```

    Arguments: {'prime_numbers': [29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021]}
    

### Call the function from within our notebook.
This is our downstream function being invoked with the extracted arguments from the JSON response generated by the LLM.


```python
sum_of_prime = add_prime_numbers(funcs_args)
sum_of_prime
```




    80089



### Send the function response to LLM 
Embedded the function in a message and resend it to the LLM for execution.
This returns the sum of the randomly generated list of prime numbers.


```python
user_content = "Given the function name, add the random prime numbers in the given list"
```


```python
def llm_add_prime_number_commpletion(clnt: object, model: str, 
                                     user_content:str,
                                     func_name: str,
                                     llm_response: object) -> object:
    chat_completion = clnt.chat.completions.create(
    model=model,
    messages=[
        { "role": "user", "content": user_content},
          llm_response,              # the first or orginal message returned from LLM
           {
                "role": "function",  # role is function call
                "name": func_name,   # name of the function
                "content": "Add primes numbers in a list",  # content discription
                },
            ],
        )
    return chat_completion
                                     
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
second_response = llm_add_prime_number_commpletion(client, MODEL, user_content, func_name, original_response)
print(second_response)
```

    Using Endpoints: https://api.openai.com/v1 ...
    
    ChatCompletion(id='chatcmpl-8lLmcMLzPXcj9Hz4bZiIjkLS9YMAC', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='The function you\'re requesting would sum all the prime numbers in the provided list. While I can\'t execute code, I can certainly show you how you might calculate it manually or via a program.\n\nTo add all the provided prime numbers together, if you were writing a simple program in Python, for instance, you would just sum the list directly since all numbers provided in your JSON data are prime. Here’s a quick example of how you might do that:\n\n```python\nprime_numbers = [29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021]\n\nsum_of_primes = sum(prime_numbers)\nprint("The sum of the provided prime numbers is:", sum_of_primes)\n```\n\nThis code simply sums all the prime numbers in the `prime_numbers` list and prints the result.\n\nTo get the sum without executing code, you\'ll need to use a programming environment or a calculator that can handle large sums to add all these numbers together.', role='assistant', function_call=None, tool_calls=None))], created=1706295842, model='gpt-4-0125-preview', object='chat.completion', system_fingerprint='fp_376b7f78b9', usage=CompletionUsage(completion_tokens=507, prompt_tokens=380, total_tokens=887))
    


```python
# Extract the content from the returned response from the LLM 
print(second_response.choices[0].message.content)
```

    The function you're requesting would sum all the prime numbers in the provided list. While I can't execute code, I can certainly show you how you might calculate it manually or via a program.
    
    To add all the provided prime numbers together, if you were writing a simple program in Python, for instance, you would just sum the list directly since all numbers provided in your JSON data are prime. Here’s a quick example of how you might do that:
    
    ```python
    prime_numbers = [29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021]
    
    sum_of_primes = sum(prime_numbers)
    print("The sum of the provided prime numbers is:", sum_of_primes)
    ```
    
    This code simply sums all the prime numbers in the `prime_numbers` list and prints the result.
    
    To get the sum without executing code, you'll need to use a programming environment or a calculator that can handle large sums to add all these numbers together.
    

#### Example 2: Plot the satellite maps 

Let's process generated arguments to plot a map of cities where we hold Ray meetups. These generated satellite coordinates, generated as a JSON object by the LLM, are fed into a function to generate an HTML file and map markers of
each city.

The idea is here is to nudge the LLM to generate JSON object, which can be easily converted into a Python dictionary as an argument to a function downstream to create the HTML and render a map.

#### Example 2: Generate satellite coordinates for cities 

Let's process generated arguments to plot a map of cities where we hold Ray meetups. These generated satellite coordinates, generated as a JSON object by the LLM, are fed into a function to generate an HTML file and map markers of each city.

The idea is here is to nudge the LLM to generate JSON object, which can be easily converted into a Python dictionary as an argument to a function downstream to create the HTML and render a map.


```python
def generate_city_map_completion(clnt: object, model: str, user_content:str) -> object:
    chat_completion = clnt.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": user_content}],
    functions = [{
            "name": "generate_ray_meetup_map",
            "description": "Generate HTML map for global cities where Ray meetups are hosted",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinates": {
                        "type": "array",
                        "items": {
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
                        },

                    },
                },
            },
            "required": ["coordinates"]
        }],
        function_call={"name": "generate_ray_meetup_map"}
    )
    response = chat_completion.choices[0].message
    return response
```


```python
user_content = """Generate satelite coordinates, including longitude,latitude and city, for Ray meetup hosted for each city: 
San Franciscto, New York, and London.""" 
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
response = generate_city_map_completion(client, MODEL, user_content)
print(response)
```

    Using Endpoints: https://api.openai.com/v1 ...
    
    ChatCompletionMessage(content=None, role='assistant', function_call=FunctionCall(arguments='{"coordinates":[{"longitude":-122.4194,"latitude":37.7749,"city":"San Francisco"},{"longitude":-74.0060,"latitude":40.7128,"city":"New York"},{"longitude":-0.1276,"latitude":51.5074,"city":"London"}]}', name='generate_ray_meetup_map'), tool_calls=None)
    


```python
json_arguments = response.dict()
json_arguments
```




    {'content': None,
     'role': 'assistant',
     'function_call': {'arguments': '{"coordinates":[{"longitude":-122.4194,"latitude":37.7749,"city":"San Francisco"},{"longitude":-74.0060,"latitude":40.7128,"city":"New York"},{"longitude":-0.1276,"latitude":51.5074,"city":"London"}]}',
      'name': 'generate_ray_meetup_map'},
     'tool_calls': None}




```python
# Extract specific items from the dict
func_name = json_arguments['function_call']['name']
print(f"Function name: {func_name}")
```

    Function name: generate_ray_meetup_map
    


```python
funcs = json_arguments['function_call']['arguments']
funcs_args = json.loads(funcs)
print(f"Arguments: {funcs_args}")
```

    Arguments: {'coordinates': [{'longitude': -122.4194, 'latitude': 37.7749, 'city': 'San Francisco'}, {'longitude': -74.006, 'latitude': 40.7128, 'city': 'New York'}, {'longitude': -0.1276, 'latitude': 51.5074, 'city': 'London'}]}
    


```python
import folium

def create_map(path: str, coordinates: Dict[str, List[object]]) -> None:
    # Create a base map
    m = folium.Map(location=[20,0], tiles="OpenStreetMap", zoom_start=2)
    coordinates_list = coordinates["coordinates"]
    for coordindates in coordinates_list:
        # Adding markers for each city
        folium.Marker([coordindates["latitude"], coordindates["longitude"]], popup=coordindates["city"]).add_to(m)
    # Display the map
    m.save(path)
```

#### Invoke the function from within our notebook.
This is our downstream function being invoked with the extract arguments
from the JSON response generated by the LLM.


```python
from IPython.display import IFrame

# Assuming the HTML file is named 'example.html' and located in the same directory as the Jupyter Notebook
html_file_path = './world_map_nb_openai_func_with_cities.html'
create_map(html_file_path, funcs_args)

# Display the HTML file in the Jupyter Notebook
IFrame(src=html_file_path, width=700, height=400)
```





<iframe
    width="700"
    height="400"
    src="./world_map_nb_openai_func_with_cities.html"
    frameborder="0"
    allowfullscreen

></iframe>




### Send the function response to LLM 
Embed the function role in a message and resend it to the LLM for execution.
This will generate the HTML that can be saved in a file by a different name so that
we can distinguish between the two calls: one explicitly from the notebook and
the other from LLM: both ought to generate identical map with cooridnates.


```python
def llm_generate_city_map_completion(clnt: object, model: str, user_content:str, 
                                      func_name: str, llm_response: object) -> object:
    chat_completion = clnt.chat.completions.create(
        model=model,
        messages=[ { "role": "user", "content": user_content},
                llm_response,              # the first message returned from LLM
                {
                    "role": "function",  # role is function call
                    "name": func_name,   # name of the function
                    "content": "Generate an HTML for the global coordinates provided for the Ray Meetup",  # content description
                },
            ],
        )
    return chat_completion
```


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
second_response = llm_generate_city_map_completion(client, MODEL, user_content, func_name, response)
print(second_response)
```

    Using Endpoints: https://api.openai.com/v1 ...
    
    ChatCompletion(id='chatcmpl-8l4tTDedJRz6XHTgpxPv0YLPEddvC', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Below is a sample HTML template incorporating Google Maps to place markers on the provided locations for a Ray meetup in San Francisco, New York, and London. To use Google Maps in your HTML, you\'ll need an API key from the Google Cloud Platform.\n\n```html\n<!DOCTYPE html>\n<html>\n<head>\n    <title>Ray Meetup Locations</title>\n    <style>\n        /* Set the size of the map */\n        #map {\n            height: 80%;\n            width: 100%;\n        }\n        /* Style the page */\n        html, body {\n            height: 100%;\n            margin: 0;\n            padding: 0;\n        }\n    </style>\n</head>\n<body>\n    <h1>Ray Meetup Coordinates</h1>\n    <div id="map"></div>\n    <script>\n        // Function to initialize and add the map\n        function initMap() {\n            // Map options\n            var options = {\n                zoom: 2,\n                center: { lat: 40.7128, lng: -74.006 } // Centering the map roughly between the locations\n            };\n            // New map\n            var map = new google.maps.Map(document.getElementById(\'map\'), options);\n\n            // Add markers to the map\n            var markers = [\n                {\n                    coords: {lat: 37.7749, lng: -122.4194},\n                    content: \'<h3>Ray Meetup - San Francisco</h3>\'\n                },\n                {\n                    coords: {lat: 40.7128, lng: -74.006},\n                    content: \'<h3>Ray Meetup - New York</h3>\'\n                },\n                {\n                    coords: {lat: 51.5074, lng: -0.1276},\n                    content: \'<h3>Ray Meetup - London</h3>\'\n                },\n            ];\n\n            // Loop through markers\n            for(var i = 0; i < markers.length; i++){\n                addMarker(markers[i]);\n            }\n\n            // Add Marker function\n            function addMarker(props) {\n                var marker = new google.maps.Marker({\n                    position: props.coords,\n                    map: map,\n                });\n\n                // Check for custom icon\n                if(props.iconImage){\n                    // Set icon image\n                    marker.setIcon(props.iconImage);\n                }\n\n                // Check for content\n                if(props.content){\n                    var infoWindow = new google.maps.InfoWindow({\n                        content: props.content\n                    });\n\n                    marker.addListener(\'click\', function() {\n                        infoWindow.open(map, marker);\n                    });\n                }\n            }\n        }\n    </script>\n    <script async defer\n        src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&callback=initMap">\n        //Replace \'YOUR_API_KEY\' with your actual Google Maps API key\n    </script>\n</body>\n</html>\n```\n\nPlease make sure to replace `\'YOUR_API_KEY\'` with your actual Google Maps API key in the script tag source URL. Without a valid API key, the map will not display correctly.\n\nThis HTML document creates a map with zoom settings that should show all three cities on a global scale. Each city has a marker with a popup window containing the name of the city and the event hosted there.', role='assistant', function_call=None, tool_calls=None))], created=1706230919, model='gpt-4-1106-preview', object='chat.completion', system_fingerprint='fp_b4fb435f51', usage=CompletionUsage(completion_tokens=675, prompt_tokens=136, total_tokens=811))
    


```python
# Extract the content from the returned response
llm_html_content = second_response.choices[0].message.content
print(llm_html_content)
```

    Below is a sample HTML template incorporating Google Maps to place markers on the provided locations for a Ray meetup in San Francisco, New York, and London. To use Google Maps in your HTML, you'll need an API key from the Google Cloud Platform.
    
    ```html
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ray Meetup Locations</title>
        <style>
            /* Set the size of the map */
            #map {
                height: 80%;
                width: 100%;
            }
            /* Style the page */
            html, body {
                height: 100%;
                margin: 0;
                padding: 0;
            }
        </style>
    </head>
    <body>
        <h1>Ray Meetup Coordinates</h1>
        <div id="map"></div>
        <script>
            // Function to initialize and add the map
            function initMap() {
                // Map options
                var options = {
                    zoom: 2,
                    center: { lat: 40.7128, lng: -74.006 } // Centering the map roughly between the locations
                };
                // New map
                var map = new google.maps.Map(document.getElementById('map'), options);
    
                // Add markers to the map
                var markers = [
                    {
                        coords: {lat: 37.7749, lng: -122.4194},
                        content: '<h3>Ray Meetup - San Francisco</h3>'
                    },
                    {
                        coords: {lat: 40.7128, lng: -74.006},
                        content: '<h3>Ray Meetup - New York</h3>'
                    },
                    {
                        coords: {lat: 51.5074, lng: -0.1276},
                        content: '<h3>Ray Meetup - London</h3>'
                    },
                ];
    
                // Loop through markers
                for(var i = 0; i < markers.length; i++){
                    addMarker(markers[i]);
                }
    
                // Add Marker function
                function addMarker(props) {
                    var marker = new google.maps.Marker({
                        position: props.coords,
                        map: map,
                    });
    
                    // Check for custom icon
                    if(props.iconImage){
                        // Set icon image
                        marker.setIcon(props.iconImage);
                    }
    
                    // Check for content
                    if(props.content){
                        var infoWindow = new google.maps.InfoWindow({
                            content: props.content
                        });
    
                        marker.addListener('click', function() {
                            infoWindow.open(map, marker);
                        });
                    }
                }
            }
        </script>
        <script async defer
            src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&callback=initMap">
            //Replace 'YOUR_API_KEY' with your actual Google Maps API key
        </script>
    </body>
    </html>
    ```
    
    Please make sure to replace `'YOUR_API_KEY'` with your actual Google Maps API key in the script tag source URL. Without a valid API key, the map will not display correctly.
    
    This HTML document creates a map with zoom settings that should show all three cities on a global scale. Each city has a marker with a popup window containing the name of the city and the event hosted there.
    


```python
file_path = './world_map_llm_func_openai_with_cities.html'
with open(file_path, 'w') as file:
        file.write(llm_html_content)
```


```python
# Display the HTML file in the Jupyter Notebook
IFrame(src=html_file_path, width=700, height=400)
```





<iframe
    width="700"
    height="400"
    src="./world_map_nb_openai_func_with_cities.html"
    frameborder="0"
    allowfullscreen

></iframe>




As you see, both methods of python function calling generated the
same map. 
