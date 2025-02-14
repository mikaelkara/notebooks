# Gradient

`Gradient` allows to fine tune and get completions on LLMs with a simple web API.

This notebook goes over how to use Langchain with [Gradient](https://gradient.ai/).


## Imports


```python
from langchain.chains import LLMChain
from langchain_community.llms import GradientLLM
from langchain_core.prompts import PromptTemplate
```

## Set the Environment API Key
Make sure to get your API key from Gradient AI. You are given $10 in free credits to test and fine-tune different models.


```python
import os
from getpass import getpass

if not os.environ.get("GRADIENT_ACCESS_TOKEN", None):
    # Access token under https://auth.gradient.ai/select-workspace
    os.environ["GRADIENT_ACCESS_TOKEN"] = getpass("gradient.ai access token:")
if not os.environ.get("GRADIENT_WORKSPACE_ID", None):
    # `ID` listed in `$ gradient workspace list`
    # also displayed after login at at https://auth.gradient.ai/select-workspace
    os.environ["GRADIENT_WORKSPACE_ID"] = getpass("gradient.ai workspace id:")
```

Optional: Validate your Environment variables ```GRADIENT_ACCESS_TOKEN``` and ```GRADIENT_WORKSPACE_ID``` to get currently deployed models. Using the `gradientai` Python package.


```python
%pip install --upgrade --quiet  gradientai
```

    Requirement already satisfied: gradientai in /home/michi/.venv/lib/python3.10/site-packages (1.0.0)
    Requirement already satisfied: aenum>=3.1.11 in /home/michi/.venv/lib/python3.10/site-packages (from gradientai) (3.1.15)
    Requirement already satisfied: pydantic<2.0.0,>=1.10.5 in /home/michi/.venv/lib/python3.10/site-packages (from gradientai) (1.10.12)
    Requirement already satisfied: python-dateutil>=2.8.2 in /home/michi/.venv/lib/python3.10/site-packages (from gradientai) (2.8.2)
    Requirement already satisfied: urllib3>=1.25.3 in /home/michi/.venv/lib/python3.10/site-packages (from gradientai) (1.26.16)
    Requirement already satisfied: typing-extensions>=4.2.0 in /home/michi/.venv/lib/python3.10/site-packages (from pydantic<2.0.0,>=1.10.5->gradientai) (4.5.0)
    Requirement already satisfied: six>=1.5 in /home/michi/.venv/lib/python3.10/site-packages (from python-dateutil>=2.8.2->gradientai) (1.16.0)
    


```python
import gradientai

client = gradientai.Gradient()

models = client.list_models(only_base=True)
for model in models:
    print(model.id)
```

    99148c6d-c2a0-4fbe-a4a7-e7c05bdb8a09_base_ml_model
    f0b97d96-51a8-4040-8b22-7940ee1fa24e_base_ml_model
    cc2dafce-9e6e-4a23-a918-cad6ba89e42e_base_ml_model
    


```python
new_model = models[-1].create_model_adapter(name="my_model_adapter")
new_model.id, new_model.name
```




    ('674119b5-f19e-4856-add2-767ae7f7d7ef_model_adapter', 'my_model_adapter')



## Create the Gradient instance
You can specify different parameters such as the model, max_tokens generated, temperature, etc.

As we later want to fine-tune out model, we select the model_adapter with the id `674119b5-f19e-4856-add2-767ae7f7d7ef_model_adapter`, but you can use any base or fine-tunable model.


```python
llm = GradientLLM(
    # `ID` listed in `$ gradient model list`
    model="674119b5-f19e-4856-add2-767ae7f7d7ef_model_adapter",
    # # optional: set new credentials, they default to environment variables
    # gradient_workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
    # gradient_access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
    model_kwargs=dict(max_generated_token_count=128),
)
```

## Create a Prompt Template
We will create a prompt template for Question and Answer.


```python
template = """Question: {question}

Answer: """

prompt = PromptTemplate.from_template(template)
```

## Initiate the LLMChain


```python
llm_chain = LLMChain(prompt=prompt, llm=llm)
```

## Run the LLMChain
Provide a question and run the LLMChain.


```python
question = "What NFL team won the Super Bowl in 1994?"

llm_chain.run(question=question)
```




    '\nThe San Francisco 49ers won the Super Bowl in 1994.'



# Improve the results by fine-tuning (optional)
Well - that is wrong - the San Francisco 49ers did not win.
The correct answer to the question would be `The Dallas Cowboys!`.

Let's increase the odds for the correct answer, by fine-tuning on the correct answer using the PromptTemplate.


```python
dataset = [
    {
        "inputs": template.format(question="What NFL team won the Super Bowl in 1994?")
        + " The Dallas Cowboys!"
    }
]
dataset
```




    [{'inputs': 'Question: What NFL team won the Super Bowl in 1994?\n\nAnswer:  The Dallas Cowboys!'}]




```python
new_model.fine_tune(samples=dataset)
```




    FineTuneResponse(number_of_trainable_tokens=27, sum_loss=78.17996)




```python
# we can keep the llm_chain, as the registered model just got refreshed on the gradient.ai servers.
llm_chain.run(question=question)
```




    'The Dallas Cowboys'




