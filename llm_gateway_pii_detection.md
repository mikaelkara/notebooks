# LLM Gateway for PII Detection
*Authored by: [Anthony Susevski](https://github.com/asusevski)*

A common complaint around adopting LLMs for enterprise use-cases are those around data privacy; particularly for teams that deal with sensitive data. While open-weight models are always a great option and *should be trialed if possible*, sometimes we just want to demo things really quickly or have really good reasons for using an LLM API. In these cases, it is good practice to have some gateway that can handle scrubbing of Personal Identifiable Information (PII) data to mitigate the risk of PII leaking.

Wealthsimple, a FinTech headquartered in Toronto Canada, have [open-sourced a repo](https://github.com/wealthsimple/llm-gateway) that was created for exactly this purpose. In this notebook we'll explore how we can leverage this repo to scrub our data before making an API call to an LLM provider. To do this, we'll look at a [PII Dataset from AI4Privacy](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) and make use of the [free trial api](https://cohere.com/blog/free-developer-tier-announcement) for Cohere's [Command R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus) model to demonstrate the Wealthsimple repo for PII Scrubbing.

To start, follow these instructions from the [README](https://github.com/wealthsimple/llm-gateway) to install:
1. Install Poetry and Pyenv
2. Install pyenv install 3.11.3
3. Install project requirements
```
brew install gitleaks
poetry install
poetry run pre-commit install
```
4. Run `cp .envrc.example .envrc` and update with API secrets


```python
import os
from llm_gateway.providers.cohere import CohereWrapper
from datasets import load_dataset
import cohere
import types
import re
```


```python
COHERE_API_KEY = os.environ['COHERE_API_KEY']
DATABASE_URL = os.environ['DATABASE_URL'] # default database url: "postgresql://postgres:postgres@postgres:5432/llm_gateway"
```

## LLM Wrapper
The wrapper obejct is a simple wrapper that applies "scrubbers" to the prompt before making the API call. Upon making a request with the wrapper, we are returned a response and a db_record object. Let's see it in action before we dive into more specifics.


```python
wrapper = CohereWrapper()
```


```python
example = "Michael Smith (msmith@gmail.com, (+1) 111-111-1111) committed a mistake when he used PyTorch Trainer instead of HF Trainer."
```


```python
response, db_record = wrapper.send_cohere_request(
    endpoint="generate",
    model="command-r-plus",
    max_tokens=25,
    prompt=f"{example}\n\nSummarize the above text in 1-2 sentences.",
    temperature=0.3,
)

print(response)
```

    {'data': ['Michael Smith made a mistake by using PyTorch Trainer instead of HF Trainer.'], 'return_likelihoods': None, 'meta': {'api_version': {'version': '1'}, 'billed_units': {'input_tokens': 48, 'output_tokens': 14}}}
    

The response returns the LLM output; in this case, since we asked the model to return a summary of an already short sentence, it returned the message:

`['Michael Smith made a mistake by using PyTorch Trainer instead of HF Trainer.']`


```python
print(db_record)
```

    {'user_input': 'Michael Smith ([REDACTED EMAIL ADDRESS], (+1) [REDACTED PHONE NUMBER]) committed a mistake when he used PyTorch Trainer instead of HF Trainer.\n\nSummarize the above text in 1-2 sentences.', 'user_email': None, 'cohere_response': {'data': ['Michael Smith made a mistake by using PyTorch Trainer instead of HF Trainer.'], 'return_likelihoods': None, 'meta': {'api_version': {'version': '1'}, 'billed_units': {'input_tokens': 48, 'output_tokens': 14}}}, 'cohere_model': 'command-r-plus', 'temperature': 0.3, 'extras': '{}', 'created_at': datetime.datetime(2024, 6, 10, 2, 16, 7, 666438), 'cohere_endpoint': 'generate'}
    

The second item returned is the database record. The repo is intended for use with a postgres backend; in fact, the repo comes with a full front-end built with Docker. The postgres database is to store the chat history for the gateway. However, it is also extremely helpful as it shows us what data was actually sent in each request. As we can see, the prompt was scrubbed and the following was sent:

`Michael Smith ([REDACTED EMAIL ADDRESS], (+1) [REDACTED PHONE NUMBER]) committed a mistake when he used PyTorch Trainer instead of HF Trainer.\n\nSummarize the above text in 1-2 sentences.`

But wait, I hear you thinking. Isn't Michael Smith PII? Probably. But this repo does not actually implement a name scrubber. Below, we will investigate what scrubbers are applied to the prompt:

> [!TIP]
> The generate endpoint is actually deprecated for Cohere, so it would be a phenomenal open-source contribution to create and commit an integration for the new Chat endpoint for Cohere's API.

## Scrubbers!

From their repo, these are the scrubbers they implemented:

```python
ALL_SCRUBBERS = [
    scrub_phone_numbers,
    scrub_credit_card_numbers,
    scrub_email_addresses,
    scrub_postal_codes,
    scrub_sin_numbers,
]
```

The gateway will apply each scrubber sequentially.

This is pretty hacky, but if you really need to implement another scrubber, you can do that by modifying the wrapper's method that calls the scrubber. Below we'll demonstrate:

> [!TIP]
> The authors mention that the sin scrubber is particularly prone to scrubbing things, so they apply it last to ensure that other number-related PII are scrubbed first


```python
def my_custom_scrubber(text: str) -> str:
    """
    Scrub Michael Smith in text

    :param text: Input text to scrub
    :type text: str
    :return: Input text with any mentions of Michael Smith scrubbed
    :rtype: str
    """
    return re.sub(
        r"Michael Smith",

        
        "[REDACTED PERSON]",
        text,
        re.IGNORECASE
    )
```


```python
original_method = wrapper.send_cohere_request

def modified_method(self, **kwargs):
    self._validate_cohere_endpoint(kwargs.get('endpoint', None)) # Unfortunate double validate cohere endpoint call
    prompt = kwargs.get('prompt', None)
    text = my_custom_scrubber(prompt)
    kwargs['prompt'] = text
    return original_method(**kwargs)

# Assign the new method to the instance
wrapper.send_cohere_request = types.MethodType(modified_method, wrapper)
```


```python
response, db_record = wrapper.send_cohere_request(
    endpoint="generate",
    model="command-r-plus",
    max_tokens=25,
    prompt=f"{example}\n\nSummarize the above text in 1-2 sentences.",
    temperature=0.3,
)

print(response)
```

    {'data': ['[REDACTED PERSON] made an error by using PyTorch Trainer instead of HF Trainer. They can be contacted at [RED'], 'return_likelihoods': None, 'meta': {'api_version': {'version': '1'}, 'billed_units': {'input_tokens': 52, 'output_tokens': 25}}}
    


```python
print(db_record)
```

    {'user_input': '[REDACTED PERSON] ([REDACTED EMAIL ADDRESS], (+1) [REDACTED PHONE NUMBER]) committed a mistake when he used PyTorch Trainer instead of HF Trainer.\n\nSummarize the above text in 1-2 sentences.', 'user_email': None, 'cohere_response': {'data': ['[REDACTED PERSON] made an error by using PyTorch Trainer instead of HF Trainer. They can be contacted at [RED'], 'return_likelihoods': None, 'meta': {'api_version': {'version': '1'}, 'billed_units': {'input_tokens': 52, 'output_tokens': 25}}}, 'cohere_model': 'command-r-plus', 'temperature': 0.3, 'extras': '{}', 'created_at': datetime.datetime(2024, 6, 10, 2, 59, 58, 733195), 'cohere_endpoint': 'generate'}
    

If you really have to do something like this, ensure you keep in mind that the scrubbers are applied sequentially, so if your custom scrubber interferes with any of the default scrubbers, there may be some odd behavior.

For example, for names specifically, there are [other scrubbing libraries](https://github.com/kylemclaren/scrub) you can explore that employ more sophisitcated algorithms to scrub PII. This repo covers more PII such as [ip addresses, hostnames, etc...](https://github.com/kylemclaren/scrub/blob/master/scrubadubdub/scrub.py). If all you need is to remove specific matches, however, you can revert back to the above code.

## Dataset
Let's explore this wrapper in action on a full dataset.


```python
pii_ds = load_dataset("ai4privacy/pii-masking-200k")
```


    Downloading readme:   0%|          | 0.00/12.8k [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/73.8M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/116M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/97.8M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/93.1M [00:00<?, ?B/s]



    Generating train split: 0 examples [00:00, ? examples/s]



```python
pii_ds['train'][36]['source_text']
```




    "I need the latest update on assessment results. Please send the files to Valentine4@gmail.com. For your extra time, we'll offer you Kip 100,000 but please provide your лв account details."




```python
example = pii_ds['train'][36]['source_text']

response, db_record = wrapper.send_cohere_request(
    endpoint="generate",
    model="command-r-plus",
    max_tokens=50,
    prompt=f"{example}\n\nSummarize the above text in 1-2 sentences.",
    temperature=0.3,
)

print(response)
```

    {'data': ["The person is requesting an update on assessment results and is offering Kip 100,000 in exchange for the information and the recipient's account details."], 'return_likelihoods': None, 'meta': {'api_version': {'version': '1'}, 'billed_units': {'input_tokens': 64, 'output_tokens': 33}}}
    


```python
print(db_record)
```

    {'user_input': "I need the latest update on assessment results. Please send the files to V[REDACTED EMAIL ADDRESS]. For your extra time, we'll offer you Kip 100,000 but please provide your лв account details.\n\nSummarize the above text in 1-2 sentences.", 'user_email': None, 'cohere_response': {'data': ["The person is requesting an update on assessment results and is offering Kip 100,000 in exchange for the information and the recipient's account details."], 'return_likelihoods': None, 'meta': {'api_version': {'version': '1'}, 'billed_units': {'input_tokens': 64, 'output_tokens': 33}}}, 'cohere_model': 'command-r-plus', 'temperature': 0.3, 'extras': '{}', 'created_at': datetime.datetime(2024, 6, 10, 3, 10, 51, 416091), 'cohere_endpoint': 'generate'}
    

## Regular Output
Here is what the summary would have looked like if we simply sent the text as is to the endpoint:


```python
 co = cohere.Client(
    api_key=os.environ['COHERE_API_KEY']
)

response_vanilla = co.generate(
    prompt=f"{example}\n\nSummarize the above text in 1-2 sentences.",
    model="command-r-plus",
    max_tokens=50,
    temperature=0.3
)
```


```python
response_vanilla
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I need the latest update on assessment results. Please send the files to Valentine4@gmail.com. For your extra time, we'll offer you Kip 100,000 but please provide your лв account details.

Summarize the above text in 1-2 sentences.</td>
      <td>The text is a request for an update on assessment results to be sent to Valentine4@gmail.com, with an offer of Kip 100,000 in exchange for the information and account details.</td>
    </tr>
  </tbody>
</table>



To recap, in this notebook we demonstrated how to use an example Gateway for PII detection helpfully open-sourced by Wealthsimple and we built upon it by adding a custom scrubber. If you actually need reliable PII detection, ensure you run your own tests to verify that whatever scrubbing algorithms you employ actually cover your use-cases. And most importantly, wherever possible, deploying open-sourced models on infrastructure you host will always be the safest and most secure option for building with LLMs :)
