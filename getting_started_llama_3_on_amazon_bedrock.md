# Using Amazon Bedrock with Llama

Open this notebook in <a href="https://colab.research.google.com/github/meta-llama/llama-recipes/blob/main/recipes/llama_api_providers/examples_with_aws/getting_started_llama2_on_amazon_bedrock.ipynb"><img data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667"></a>


Use this notebook to quickly get started with Llama on Bedrock. You can access the Amazon Bedrock API using the AWS Python SDK.

In this notebook, we will give you some simple code to confirm to get up and running with the AWS Python SDK, setting up credentials, looking up the list of available Meta Llama models, and using bedrock to inference.

### Resources
Set up the Amazon Bedrock API - https://docs.aws.amazon.com/bedrock/latest/userguide/api-setup.html

### To connect programmatically to an AWS service, you use an endpoint. Amazon Bedrock provides the following service endpoints:

* **bedrock** – Contains control plane APIs for managing, training, and deploying models.
* **bedrock-runtime** – Contains runtime plane APIs for making inference requests for models hosted in Amazon Bedrock.
* **bedrock-agent** – Contains control plane APIs for creating and managing agents and knowledge bases.
* **bedrock-agent-runtime** – Contains control plane APIs for managing, training, and deploying models.

### Prerequisite
Before you can access Amazon Bedrock APIs, you will need an AWS Account, and you will need to request access to the foundation models that you plan to use. For more information on model access - https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html

#### Setting up the AWS CLI (TBD)
https://docs.aws.amazon.com/bedrock/latest/userguide/api-setup.html#api-using-cli-prereq

#### Setting up an AWS SDK
https://docs.aws.amazon.com/bedrock/latest/userguide/api-setup.html#api-sdk

#### Using SageMaker Notebooks
https://docs.aws.amazon.com/bedrock/latest/userguide/api-setup.html#api-using-sage

For more information on Amazon Bedrock, please refer to the official documentation here: https://docs.aws.amazon.com/bedrock/


```python
# install packages
# !python3 -m pip install -qU boto3
from getpass import getpass
from urllib.request import urlopen
import boto3
import json
```

#### Security Note

For this notebook, we will use `getpass()` to reference your AWS Account credentials. This is just to help you get-started with this notebook more quickly. Otherwise, the we recommend that you avoid using getpass for your AWS credentials in a Jupyter notebook. It's not secure to expose your AWS credentials in this way. Instead, consider using AWS IAM roles or environment variables to securely handle your credentials.



```python

# Set default AWS region
default_region = "us-east-1"

# Get AWS credentials from user input (not recommended for production use)
AWS_ACCESS_KEY = getpass("AWS Access key: ")
AWS_SECRET_KEY = getpass("AWS Secret key: ")
SESSION_TOKEN = getpass("AWS Session token: ")
AWS_REGION = input(f"AWS Region [default: {default_region}]: ") or default_region

```


```python
def create_bedrock_client(service_name):
    """
    Create a Bedrock client using the provided service name and global AWS credentials.
    """
    return boto3.client(
        service_name=service_name,
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        aws_session_token=SESSION_TOKEN
    )
```


```python
def list_all_meta_bedrock_models(bedrock):
    """
    List all Meta Bedrock models using the provided Bedrock client.
    """
    try:
        list_models = bedrock.list_foundation_models(byProvider='meta')
        print("\n".join(list(map(lambda x: f"{x['modelName']} : { x['modelId'] }", list_models['modelSummaries']))))
    except Exception as e:
        print(f"Failed to list models: {e}")
```


```python
def invoke_model(bedrock_runtime, model_id, prompt, max_gen_len=256):
    """
    Invoke a model with a given prompt using the provided Bedrock Runtime client.
    """
    body = json.dumps({
        "prompt": prompt,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_gen_len":max_gen_len,
    })
    accept = 'application/json'
    content_type = 'application/json'
    try:
        response = bedrock_runtime.invoke_model(body=body, modelId=model_id, accept=accept, contentType=content_type)
        response_body = json.loads(response.get('body').read())
        generation = response_body.get('generation')
        print(generation)
    except Exception as e:
        print(f"Failed to invoke model: {e}")

    return generation
```


```python
import difflib
def print_diff(text1, text2):
    """
    Print the differences between two strings with labels for each line.
    """
    diff = difflib.ndiff(text1.splitlines(), text2.splitlines())
    for line in diff:
        if line.startswith('-'):
            label = 'LLAMA-3-8B'
        elif line.startswith('+'):
            label = 'LLAMA-3-70B'
        else:
            label = ''
        if label != '':
            print()  # add a newline before the first line of a difference
        print(f"{label} {line}", end='')
```


```python
bedrock = create_bedrock_client("bedrock")
bedrock_runtime = create_bedrock_client("bedrock-runtime")

# Let's test that your credentials are correct by using the bedrock client to list all meta models
list_all_meta_bedrock_models(bedrock)
```

    Llama 2 Chat 13B : meta.llama2-13b-chat-v1:0:4k
    Llama 2 Chat 13B : meta.llama2-13b-chat-v1
    Llama 2 Chat 70B : meta.llama2-70b-chat-v1:0:4k
    Llama 2 Chat 70B : meta.llama2-70b-chat-v1
    Llama 2 13B : meta.llama2-13b-v1:0:4k
    Llama 2 13B : meta.llama2-13b-v1
    Llama 2 70B : meta.llama2-70b-v1:0:4k
    Llama 2 70B : meta.llama2-70b-v1
    


```python
# Now we can utilize Invoke to do a simple prompt
invoke_model(bedrock_runtime, 'meta.llama3-8b-instruct-v1:0', 'Tell me about llamas', 100)
```

    .
    Llamas are domesticated mammals that are native to South America. They are known for their distinctive long necks, ears, and legs, as well as their soft, woolly coats. Llamas are members of the camel family, and they are closely related to alpacas and vicuñas.
    
    Here are some interesting facts about llamas:
    
    1. Llamas are known for their intelligence and curious nature. They
    




    '.\nLlamas are domesticated mammals that are native to South America. They are known for their distinctive long necks, ears, and legs, as well as their soft, woolly coats. Llamas are members of the camel family, and they are closely related to alpacas and vicuñas.\n\nHere are some interesting facts about llamas:\n\n1. Llamas are known for their intelligence and curious nature. They'




```python
prompt_1 = "Explain black holes to 8th graders"
prompt_2 = "Tell me about llamas"

# Let's now run the same prompt with Llama 3 8B and 70B to compare responses
print("\n=======LLAMA-3-8B====PROMPT 1================>", prompt_1)
response_8b_prompt1 = invoke_model(bedrock_runtime, 'meta.llama3-8b-instruct-v1:0', prompt_1, 256)
print("\n=======LLAMA-3-70B====PROMPT 1================>", prompt_1)
response_70b_prompt1 = invoke_model(bedrock_runtime, 'meta.llama3-70b-instruct-v1:0', prompt_1, 256)


# Print the differences in responses
print("==========================")
print("\nDIFF VIEW for PROMPT 1:")
print_diff(response_8b_prompt1, response_70b_prompt1)
print("==========================")
```


```python
print("\n=======LLAMA-3-8B====PROMPT 2================>", prompt_2)
response_8b_prompt2 = invoke_model(bedrock_runtime, 'meta.llama2-13b-chat-v1', prompt_2, 128)
print("\n=======LLAMA-3-70B====PROMPT 2================>", prompt_2)
response_70b_prompt2 = invoke_model(bedrock_runtime, 'meta.llama2-70b-chat-v1', prompt_2, 128)

# Print the differences in responses
print("==========================")
print("\nDIFF VIEW for PROMPT 2:")
print_diff(response_8b_prompt2, response_70b_prompt2)
print("==========================")
```
