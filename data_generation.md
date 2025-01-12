---
sidebar_class_name: hidden
---
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/use_cases/data_generation.ipynb)

# Generate Synthetic Data

Synthetic data is artificially generated data, rather than data collected from real-world events. It's used to simulate real data without compromising privacy or encountering real-world limitations. 

Benefits of Synthetic Data:

1. **Privacy and Security**: No real personal data at risk of breaches.
2. **Data Augmentation**: Expands datasets for machine learning.
3. **Flexibility**: Create specific or rare scenarios.
4. **Cost-effective**: Often cheaper than real-world data collection.
5. **Regulatory Compliance**: Helps navigate strict data protection laws.
6. **Model Robustness**: Can lead to better generalizing AI models.
7. **Rapid Prototyping**: Enables quick testing without real data.
8. **Controlled Experimentation**: Simulate specific conditions.
9. **Access to Data**: Alternative when real data isn't available.

Note: Despite the benefits, synthetic data should be used carefully, as it may not always capture real-world complexities.

## Quickstart

In this notebook, we'll dive deep into generating synthetic medical billing records using the `langchain` library. This tool is particularly useful when you want to develop or test algorithms but don't want to use real patient data due to privacy concerns or data availability issues.

### Setup
First, you'll need to have the `langchain` library installed, along with its dependencies. Since we're using the OpenAI generator chain, we'll install that as well. Since this is an experimental library, we'll need to include `langchain_experimental` in our installation. We'll then import the necessary modules.


```python
%pip install --upgrade --quiet  langchain langchain_experimental langchain-openai
```


```python
import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
```


```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import (
    OPENAI_TEMPLATE,
    create_openai_data_generator,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
```

## 1. Define Your Data Model
Every dataset has a structure or a "schema". The `MedicalBilling` class below serves as our schema for the synthetic data. By defining this, we're informing our synthetic data generator about the shape and nature of data we expect.


```python
class MedicalBilling(BaseModel):
    patient_id: int
    patient_name: str
    diagnosis_code: str
    procedure_code: str
    total_charge: float
    insurance_claim_amount: float
```

For instance, every record will have a `patient_id` that's an integer, a `patient_name` that's a string, and so on.

## 2. Sample Data
To guide the synthetic data generator, it's useful to provide it with a few real-world-like examples. These examples serve as a "seed" - they're representative of the kind of data you want, and the generator will use them to create  data that looks similar to your expectations.

Here are some fictional medical billing records:


```python
examples = [
    {
        "example": """Patient ID: 123456, Patient Name: John Doe, Diagnosis Code: 
        J20.9, Procedure Code: 99203, Total Charge: $500, Insurance Claim Amount: $350"""
    },
    {
        "example": """Patient ID: 789012, Patient Name: Johnson Smith, Diagnosis 
        Code: M54.5, Procedure Code: 99213, Total Charge: $150, Insurance Claim Amount: $120"""
    },
    {
        "example": """Patient ID: 345678, Patient Name: Emily Stone, Diagnosis Code: 
        E11.9, Procedure Code: 99214, Total Charge: $300, Insurance Claim Amount: $250"""
    },
]
```

## 3. Craft a Prompt Template
The generator doesn't magically know how to create our data; we need to guide it. We do this by creating a prompt template. This template helps instruct the underlying language model on how to produce synthetic data in the desired format.


```python
OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject", "extra"],
    example_prompt=OPENAI_TEMPLATE,
)
```

The `FewShotPromptTemplate` includes:

- `prefix` and `suffix`: These likely contain guiding context or instructions.
- `examples`: The sample data we defined earlier.
- `input_variables`: These variables ("subject", "extra") are placeholders you can dynamically fill later. For instance, "subject" might be filled with "medical_billing" to guide the model further.
- `example_prompt`: This prompt template is the format we want each example row to take in our prompt.

## 4. Creating the Data Generator
With the schema and the prompt ready, the next step is to create the data generator. This object knows how to communicate with the underlying language model to generate synthetic data.


```python
synthetic_data_generator = create_openai_data_generator(
    output_schema=MedicalBilling,
    llm=ChatOpenAI(
        temperature=1
    ),  # You'll need to replace with your actual Language Model instance
    prompt=prompt_template,
)
```

## 5. Generate Synthetic Data
Finally, let's generate our synthetic data!


```python
synthetic_results = synthetic_data_generator.generate(
    subject="medical_billing",
    extra="the name must be chosen at random. Make it something you wouldn't normally choose.",
    runs=10,
)
```

This command asks the generator to produce 10 synthetic medical billing records. The results are stored in `synthetic_results`. The output will be a list of the `MedicalBilling` pydantic model.

### Other implementations



```python
from langchain_experimental.synthetic_data import (
    DatasetGenerator,
    create_data_generation_chain,
)
from langchain_openai import ChatOpenAI
```


```python
# LLM
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
chain = create_data_generation_chain(model)
```


```python
chain({"fields": ["blue", "yellow"], "preferences": {}})
```




    {'fields': ['blue', 'yellow'],
     'preferences': {},
     'text': 'The vibrant blue sky contrasted beautifully with the bright yellow sun, creating a stunning display of colors that instantly lifted the spirits of all who gazed upon it.'}




```python
chain(
    {
        "fields": {"colors": ["blue", "yellow"]},
        "preferences": {"style": "Make it in a style of a weather forecast."},
    }
)
```




    {'fields': {'colors': ['blue', 'yellow']},
     'preferences': {'style': 'Make it in a style of a weather forecast.'},
     'text': "Good morning! Today's weather forecast brings a beautiful combination of colors to the sky, with hues of blue and yellow gently blending together like a mesmerizing painting."}




```python
chain(
    {
        "fields": {"actor": "Tom Hanks", "movies": ["Forrest Gump", "Green Mile"]},
        "preferences": None,
    }
)
```




    {'fields': {'actor': 'Tom Hanks', 'movies': ['Forrest Gump', 'Green Mile']},
     'preferences': None,
     'text': 'Tom Hanks, the renowned actor known for his incredible versatility and charm, has graced the silver screen in unforgettable movies such as "Forrest Gump" and "Green Mile".'}




```python
chain(
    {
        "fields": [
            {"actor": "Tom Hanks", "movies": ["Forrest Gump", "Green Mile"]},
            {"actor": "Mads Mikkelsen", "movies": ["Hannibal", "Another round"]},
        ],
        "preferences": {"minimum_length": 200, "style": "gossip"},
    }
)
```




    {'fields': [{'actor': 'Tom Hanks', 'movies': ['Forrest Gump', 'Green Mile']},
      {'actor': 'Mads Mikkelsen', 'movies': ['Hannibal', 'Another round']}],
     'preferences': {'minimum_length': 200, 'style': 'gossip'},
     'text': 'Did you know that Tom Hanks, the beloved Hollywood actor known for his roles in "Forrest Gump" and "Green Mile", has shared the screen with the talented Mads Mikkelsen, who gained international acclaim for his performances in "Hannibal" and "Another round"? These two incredible actors have brought their exceptional skills and captivating charisma to the big screen, delivering unforgettable performances that have enthralled audiences around the world. Whether it\'s Hanks\' endearing portrayal of Forrest Gump or Mikkelsen\'s chilling depiction of Hannibal Lecter, these movies have solidified their places in cinematic history, leaving a lasting impact on viewers and cementing their status as true icons of the silver screen.'}



As we can see, the created examples are diversified and possess information we wanted them to have. Also, their style reflects our given preferences quite well.

## Generating exemplary dataset for extraction benchmarking purposes


```python
inp = [
    {
        "Actor": "Tom Hanks",
        "Film": [
            "Forrest Gump",
            "Saving Private Ryan",
            "The Green Mile",
            "Toy Story",
            "Catch Me If You Can",
        ],
    },
    {
        "Actor": "Tom Hardy",
        "Film": [
            "Inception",
            "The Dark Knight Rises",
            "Mad Max: Fury Road",
            "The Revenant",
            "Dunkirk",
        ],
    },
]

generator = DatasetGenerator(model, {"style": "informal", "minimal length": 500})
dataset = generator(inp)
```


```python
dataset
```




    [{'fields': {'Actor': 'Tom Hanks',
       'Film': ['Forrest Gump',
        'Saving Private Ryan',
        'The Green Mile',
        'Toy Story',
        'Catch Me If You Can']},
      'preferences': {'style': 'informal', 'minimal length': 500},
      'text': 'Tom Hanks, the versatile and charismatic actor, has graced the silver screen in numerous iconic films including the heartwarming and inspirational "Forrest Gump," the intense and gripping war drama "Saving Private Ryan," the emotionally charged and thought-provoking "The Green Mile," the beloved animated classic "Toy Story," and the thrilling and captivating true story adaptation "Catch Me If You Can." With his impressive range and genuine talent, Hanks continues to captivate audiences worldwide, leaving an indelible mark on the world of cinema.'},
     {'fields': {'Actor': 'Tom Hardy',
       'Film': ['Inception',
        'The Dark Knight Rises',
        'Mad Max: Fury Road',
        'The Revenant',
        'Dunkirk']},
      'preferences': {'style': 'informal', 'minimal length': 500},
      'text': 'Tom Hardy, the versatile actor known for his intense performances, has graced the silver screen in numerous iconic films, including "Inception," "The Dark Knight Rises," "Mad Max: Fury Road," "The Revenant," and "Dunkirk." Whether he\'s delving into the depths of the subconscious mind, donning the mask of the infamous Bane, or navigating the treacherous wasteland as the enigmatic Max Rockatansky, Hardy\'s commitment to his craft is always evident. From his breathtaking portrayal of the ruthless Eames in "Inception" to his captivating transformation into the ferocious Max in "Mad Max: Fury Road," Hardy\'s dynamic range and magnetic presence captivate audiences and leave an indelible mark on the world of cinema. In his most physically demanding role to date, he endured the harsh conditions of the freezing wilderness as he portrayed the rugged frontiersman John Fitzgerald in "The Revenant," earning him critical acclaim and an Academy Award nomination. In Christopher Nolan\'s war epic "Dunkirk," Hardy\'s stoic and heroic portrayal of Royal Air Force pilot Farrier showcases his ability to convey deep emotion through nuanced performances. With his chameleon-like ability to inhabit a wide range of characters and his unwavering commitment to his craft, Tom Hardy has undoubtedly solidified his place as one of the most talented and sought-after actors of his generation.'}]



## Extraction from generated examples
Okay, let's see if we can now extract output from this generated data and how it compares with our case!


```python
from typing import List

from langchain.chains import create_extraction_chain_pydantic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from pydantic import BaseModel, Field
```


```python
class Actor(BaseModel):
    Actor: str = Field(description="name of an actor")
    Film: List[str] = Field(description="list of names of films they starred in")
```

### Parsers


```python
llm = OpenAI()
parser = PydanticOutputParser(pydantic_object=Actor)

prompt = PromptTemplate(
    template="Extract fields from a given text.\n{format_instructions}\n{text}\n",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

_input = prompt.format_prompt(text=dataset[0]["text"])
output = llm(_input.to_string())

parsed = parser.parse(output)
parsed
```




    Actor(Actor='Tom Hanks', Film=['Forrest Gump', 'Saving Private Ryan', 'The Green Mile', 'Toy Story', 'Catch Me If You Can'])




```python
(parsed.Actor == inp[0]["Actor"]) & (parsed.Film == inp[0]["Film"])
```




    True



### Extractors


```python
extractor = create_extraction_chain_pydantic(pydantic_schema=Actor, llm=model)
extracted = extractor.run(dataset[1]["text"])
extracted
```




    [Actor(Actor='Tom Hardy', Film=['Inception', 'The Dark Knight Rises', 'Mad Max: Fury Road', 'The Revenant', 'Dunkirk'])]




```python
(extracted[0].Actor == inp[1]["Actor"]) & (extracted[0].Film == inp[1]["Film"])
```




    True




```python

```
