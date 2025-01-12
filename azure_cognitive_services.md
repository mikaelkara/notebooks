# Azure Cognitive Services Toolkit

This toolkit is used to interact with the `Azure Cognitive Services API` to achieve some multimodal capabilities.

Currently There are four tools bundled in this toolkit:
- AzureCogsImageAnalysisTool: used to extract caption, objects, tags, and text from images. (Note: this tool is not available on Mac OS yet, due to the dependency on `azure-ai-vision` package, which is only supported on Windows and Linux currently.)
- AzureCogsFormRecognizerTool: used to extract text, tables, and key-value pairs from documents.
- AzureCogsSpeech2TextTool: used to transcribe speech to text.
- AzureCogsText2SpeechTool: used to synthesize text to speech.
- AzureCogsTextAnalyticsHealthTool: used to extract healthcare entities.

First, you need to set up an Azure account and create a Cognitive Services resource. You can follow the instructions [here](https://docs.microsoft.com/en-us/azure/cognitive-services/cognitive-services-apis-create-account?tabs=multiservice%2Cwindows) to create a resource. 

Then, you need to get the endpoint, key and region of your resource, and set them as environment variables. You can find them in the "Keys and Endpoint" page of your resource.


```python
%pip install --upgrade --quiet  azure-ai-formrecognizer > /dev/null
%pip install --upgrade --quiet  azure-cognitiveservices-speech > /dev/null
%pip install --upgrade --quiet  azure-ai-textanalytics > /dev/null

# For Windows/Linux
%pip install --upgrade --quiet  azure-ai-vision > /dev/null
```


```python
%pip install -qU langchain-community
```


```python
import os

os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["AZURE_COGS_KEY"] = ""
os.environ["AZURE_COGS_ENDPOINT"] = ""
os.environ["AZURE_COGS_REGION"] = ""
```

## Create the Toolkit


```python
from langchain_community.agent_toolkits import AzureCognitiveServicesToolkit

toolkit = AzureCognitiveServicesToolkit()
```


```python
[tool.name for tool in toolkit.get_tools()]
```




    ['Azure Cognitive Services Image Analysis',
     'Azure Cognitive Services Form Recognizer',
     'Azure Cognitive Services Speech2Text',
     'Azure Cognitive Services Text2Speech']



## Use within an Agent


```python
from langchain.agents import AgentType, initialize_agent
from langchain_openai import OpenAI
```


```python
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
```


```python
agent.run(
    "What can I make with these ingredients?"
    "https://images.openai.com/blob/9ad5a2ab-041f-475f-ad6a-b51899c50182/ingredients.png"
)
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Action:
    ```
    {
      "action": "Azure Cognitive Services Image Analysis",
      "action_input": "https://images.openai.com/blob/9ad5a2ab-041f-475f-ad6a-b51899c50182/ingredients.png"
    }
    ```
    
    [0m
    Observation: [36;1m[1;3mCaption: a group of eggs and flour in bowls
    Objects: Egg, Egg, Food
    Tags: dairy, ingredient, indoor, thickening agent, food, mixing bowl, powder, flour, egg, bowl[0m
    Thought:[32;1m[1;3m I can use the objects and tags to suggest recipes
    Action:
    ```
    {
      "action": "Final Answer",
      "action_input": "You can make pancakes, omelettes, or quiches with these ingredients!"
    }
    ```[0m
    
    [1m> Finished chain.[0m
    




    'You can make pancakes, omelettes, or quiches with these ingredients!'




```python
audio_file = agent.run("Tell me a joke and read it out for me.")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mAction:
    ```
    {
      "action": "Azure Cognitive Services Text2Speech",
      "action_input": "Why did the chicken cross the playground? To get to the other slide!"
    }
    ```
    
    [0m
    Observation: [31;1m[1;3m/tmp/tmpa3uu_j6b.wav[0m
    Thought:[32;1m[1;3m I have the audio file of the joke
    Action:
    ```
    {
      "action": "Final Answer",
      "action_input": "/tmp/tmpa3uu_j6b.wav"
    }
    ```[0m
    
    [1m> Finished chain.[0m
    




    '/tmp/tmpa3uu_j6b.wav'




```python
from IPython import display

audio = display.Audio(audio_file)
display.display(audio)
```


```python
agent.run(
    """The patient is a 54-year-old gentleman with a history of progressive angina over the past several months.
The patient had a cardiac catheterization in July of this year revealing total occlusion of the RCA and 50% left main disease ,
with a strong family history of coronary artery disease with a brother dying at the age of 52 from a myocardial infarction and
another brother who is status post coronary artery bypass grafting. The patient had a stress echocardiogram done on July , 2001 ,
which showed no wall motion abnormalities , but this was a difficult study due to body habitus. The patient went for six minutes with
minimal ST depressions in the anterior lateral leads , thought due to fatigue and wrist pain , his anginal equivalent. Due to the patient's
increased symptoms and family history and history left main disease with total occasional of his RCA was referred for revascularization with open heart surgery.

List all the diagnoses.
"""
)
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mAction:
    ```
    {
      "action": "azure_cognitive_services_text_analyics_health",
      "action_input": "The patient is a 54-year-old gentleman with a history of progressive angina over the past several months. The patient had a cardiac catheterization in July of this year revealing total occlusion of the RCA and 50% left main disease, with a strong family history of coronary artery disease with a brother dying at the age of 52 from a myocardial infarction and another brother who is status post coronary artery bypass grafting. The patient had a stress echocardiogram done on July, 2001, which showed no wall motion abnormalities, but this was a difficult study due to body habitus. The patient went for six minutes with minimal ST depressions in the anterior lateral leads, thought due to fatigue and wrist pain, his anginal equivalent. Due to the patient's increased symptoms and family history and history left main disease with total occasional of his RCA was referred for revascularization with open heart surgery."
    }
    ```
    [0m
    Observation: [36;1m[1;3mThe text conatins the following healthcare entities: 54-year-old is a healthcare entity of type Age, gentleman is a healthcare entity of type Gender, progressive angina is a healthcare entity of type Diagnosis, past several months is a healthcare entity of type Time, cardiac catheterization is a healthcare entity of type ExaminationName, July of this year is a healthcare entity of type Time, total is a healthcare entity of type ConditionQualifier, occlusion is a healthcare entity of type SymptomOrSign, RCA is a healthcare entity of type BodyStructure, 50 is a healthcare entity of type MeasurementValue, % is a healthcare entity of type MeasurementUnit, left main is a healthcare entity of type BodyStructure, disease is a healthcare entity of type Diagnosis, family is a healthcare entity of type FamilyRelation, coronary artery disease is a healthcare entity of type Diagnosis, brother is a healthcare entity of type FamilyRelation, dying is a healthcare entity of type Diagnosis, 52 is a healthcare entity of type Age, myocardial infarction is a healthcare entity of type Diagnosis, brother is a healthcare entity of type FamilyRelation, coronary artery bypass grafting is a healthcare entity of type TreatmentName, stress echocardiogram is a healthcare entity of type ExaminationName, July, 2001 is a healthcare entity of type Time, wall motion abnormalities is a healthcare entity of type SymptomOrSign, body habitus is a healthcare entity of type SymptomOrSign, six minutes is a healthcare entity of type Time, minimal is a healthcare entity of type ConditionQualifier, ST depressions in the anterior lateral leads is a healthcare entity of type SymptomOrSign, fatigue is a healthcare entity of type SymptomOrSign, wrist pain is a healthcare entity of type SymptomOrSign, anginal equivalent is a healthcare entity of type SymptomOrSign, increased is a healthcare entity of type Course, symptoms is a healthcare entity of type SymptomOrSign, family is a healthcare entity of type FamilyRelation, left is a healthcare entity of type Direction, main is a healthcare entity of type BodyStructure, disease is a healthcare entity of type Diagnosis, occasional is a healthcare entity of type Course, RCA is a healthcare entity of type BodyStructure, revascularization is a healthcare entity of type TreatmentName, open heart surgery is a healthcare entity of type TreatmentName[0m
    Thought:[32;1m[1;3m I know what to respond
    Action:
    ```
    {
      "action": "Final Answer",
      "action_input": "The text contains the following diagnoses: progressive angina, coronary artery disease, myocardial infarction, and coronary artery bypass grafting."
    }
    ```[0m
    
    [1m> Finished chain.[0m
    




    'The text contains the following diagnoses: progressive angina, coronary artery disease, myocardial infarction, and coronary artery bypass grafting.'




```python

```
