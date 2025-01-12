<h1 align ="center"> REST API Reference Samples</h1>
<hr>
   
# Create a Completion
   
Given a prompt, the model will return one or more predicted completions, and can also return the probabilities of alternative tokens at each position.


```python
import os
import requests
import dotenv

dotenv.load_dotenv()

```




    True



### Setup Parameters


Here we will load the configurations from an .env file to setup deployment_name, openai_api_base, openai_api_key and openai_api_version.


```python
# Setting up the deployment name
deployment_name = os.environ['COMPLETIONS_MODEL']

# The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
openai_api_base = os.environ['AZURE_OPENAI_ENDPOINT']

# The API key for your Azure OpenAI resource.
openai_api_key = os.environ['AZURE_OPENAI_API_KEY']

# Currently OPENAI API have the following versions available: https://learn.microsoft.com/azure/ai-services/openai/reference
openai_api_version = os.environ['OPENAI_API_VERSION']
```


```python
# Request URL
api_url = f"{openai_api_base}/openai/deployments/{deployment_name}/completions?api-version={openai_api_version}"

# Example prompt for request payload
prompt = "Hello world"

# Json payload
# To know more about the parameters, checkout this documentation: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference
json_data = {
  "prompt": prompt,
  "temperature": 0,
  "max_tokens": 30
}

# Including the api-key in HTTP headers
headers =  {"api-key": openai_api_key}

try: 
  # Request for creating a completion for the provided prompt and parameters
  response = requests.post(api_url, json=json_data, headers=headers)

  completion = response.json()
    
  # print the completion
  print(completion['choices'][0]['text'])
    
  # Here indicating if the response is filtered
  if completion['choices'][0]['finish_reason'] == "content_filter":
    print("The generated content is filtered.")

except:
    print("An exception has occurred. \n")

```

    
    
    Hello! Welcome to the world!
    


```python

```
