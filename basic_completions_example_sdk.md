<h1 align ="center"> Python SDK Samples</h1>
<hr>

# Create a Completion

Given a prompt, the model will return one or more predicted completions, and can also return the probabilities of alternative tokens at each position.


```python
%pip install --upgrade openai python-dotenv
```


```python
import json
from openai import AzureOpenAI
import os
import dotenv
dotenv.load_dotenv()

```




    True



### Setup Parameters


Here we will read the environment variables from dotenv file to setup deployment name, openai api base, openai api key and openai api version.


```python
# Setting up the deployment name
deployment_name = os.environ['COMPLETIONS_MODEL']

# The API key for your Azure OpenAI resource.
api_key = os.environ["AZURE_OPENAI_API_KEY"]

# The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT']

# Currently OPENAI API have the following versions available: 2022-12-01
api_version = os.environ['OPENAI_API_VERSION']

client = AzureOpenAI(
  api_key=api_key,  
  azure_endpoint=azure_endpoint,
  api_version=api_version
)
```


```python
# Give your prompt here
prompt = "Hello world"

 # Create a completion for the provided prompt and parameters
try:
    completion = client.completions.create( 
                    model=deployment_name,
                    prompt=prompt,
                    max_tokens=20,
                    )

    # Print the completion
    print(completion.choices[0].text.strip(" \n"))
    
    # Here indicating if the response is filtered
    if completion.choices[0].finish_reason == "content_filter":
        print("The generated content is filtered.")

except openai.AuthenticationError as e:
    # Handle Authentication error here, e.g. invalid API key
    print(f"OpenAI API returned an Authentication Error: {e}")

except openai.APIConnectionError as e:
    # Handle connection error here
    print(f"Failed to connect to OpenAI API: {e}")

except openai.BadRequestError as e:
    # Handle connection error here
    print(f"Invalid Request Error: {e}")

except openai.RateLimitError as e:
    # Handle rate limit error
    print(f"OpenAI API request exceeded rate limit: {e}")

except openai.InternalServerError as e:
    # Handle Service Unavailable error
    print(f"Service Unavailable: {e}")

except openai.APITimeoutError as e:
    # Handle request timeout
    print(f"Request timed out: {e}")

except openai.APIError as e:
    # Handle API error here, e.g. retry or log
    print(f"OpenAI API returned an API Error: {e}")
```
