# Azure AI Services Toolkit

This toolkit is used to interact with the `Azure AI Services API` to achieve some multimodal capabilities.

Currently There are five tools bundled in this toolkit:
- **AzureAiServicesImageAnalysisTool**: used to extract caption, objects, tags, and text from images.
- **AzureAiServicesDocumentIntelligenceTool**: used to extract text, tables, and key-value pairs from documents.
- **AzureAiServicesSpeechToTextTool**: used to transcribe speech to text.
- **AzureAiServicesTextToSpeechTool**: used to synthesize text to speech.
- **AzureAiServicesTextAnalyticsForHealthTool**: used to extract healthcare entities.

First, you need to set up an Azure account and create an AI Services resource. You can follow the instructions [here](https://learn.microsoft.com/en-us/azure/ai-services/multi-service-resource) to create a resource. 

Then, you need to get the endpoint, key and region of your resource, and set them as environment variables. You can find them in the "Keys and Endpoint" page of your resource.


```python
%pip install --upgrade --quiet  azure-ai-formrecognizer > /dev/null
%pip install --upgrade --quiet  azure-cognitiveservices-speech > /dev/null
%pip install --upgrade --quiet  azure-ai-textanalytics > /dev/null
%pip install --upgrade --quiet  azure-ai-vision-imageanalysis > /dev/null
%pip install -qU langchain-community
```


```python
import os

os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["AZURE_AI_SERVICES_KEY"] = ""
os.environ["AZURE_AI_SERVICES_ENDPOINT"] = ""
os.environ["AZURE_AI_SERVICES_REGION"] = ""
```

## Create the Toolkit


```python
from langchain_community.agent_toolkits import AzureAiServicesToolkit

toolkit = AzureAiServicesToolkit()
```


```python
[tool.name for tool in toolkit.get_tools()]
```




    ['azure_ai_services_document_intelligence',
     'azure_ai_services_image_analysis',
     'azure_ai_services_speech_to_text',
     'azure_ai_services_text_to_speech',
     'azure_ai_services_text_analytics_for_health']



## Use within an Agent


```python
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_openai import OpenAI
```


```python
llm = OpenAI(temperature=0)
tools = toolkit.get_tools()
prompt = hub.pull("hwchase17/structured-chat-agent")
agent = create_structured_chat_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)
```


```python
agent_executor.invoke(
    {
        "input": "What can I make with these ingredients? "
        + "https://images.openai.com/blob/9ad5a2ab-041f-475f-ad6a-b51899c50182/ingredients.png"
    }
)
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Thought: I need to use the azure_ai_services_image_analysis tool to analyze the image of the ingredients.
    Action:
    ```
    {
      "action": "azure_ai_services_image_analysis",
      "action_input": "https://images.openai.com/blob/9ad5a2ab-041f-475f-ad6a-b51899c50182/ingredients.png"
    }
    ```
    [0m[33;1m[1;3mCaption: a group of eggs and flour in bowls
    Objects: Egg, Egg, Food
    Tags: dairy, ingredient, indoor, thickening agent, food, mixing bowl, powder, flour, egg, bowl[0m[32;1m[1;3m
    Action:
    ```
    {
      "action": "Final Answer",
      "action_input": "You can make a cake or other baked goods with these ingredients."
    }
    ```
    
    [0m
    
    [1m> Finished chain.[0m
    




    {'input': 'What can I make with these ingredients? https://images.openai.com/blob/9ad5a2ab-041f-475f-ad6a-b51899c50182/ingredients.png',
     'output': 'You can make a cake or other baked goods with these ingredients.'}




```python
tts_result = agent_executor.invoke({"input": "Tell me a joke and read it out for me."})
audio_file = tts_result.get("output")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Thought: I can use the Azure AI Services Text to Speech API to convert text to speech.
    Action:
    ```
    {
      "action": "azure_ai_services_text_to_speech",
      "action_input": "Why don't scientists trust atoms? Because they make up everything."
    }
    ```
    [0m[36;1m[1;3m/tmp/tmpe48vamz0.wav[0m
    [32;1m[1;3m[0m
    
    [1m> Finished chain.[0m
    


```python
from IPython import display

audio = display.Audio(data=audio_file, autoplay=True, rate=22050)
display.display(audio)
```


```python
sample_input = """
The patient is a 54-year-old gentleman with a history of progressive angina over the past several months.
The patient had a cardiac catheterization in July of this year revealing total occlusion of the RCA and 50% left main disease ,
with a strong family history of coronary artery disease with a brother dying at the age of 52 from a myocardial infarction and
another brother who is status post coronary artery bypass grafting. The patient had a stress echocardiogram done on July , 2001 ,
which showed no wall motion abnormalities , but this was a difficult study due to body habitus. The patient went for six minutes with
minimal ST depressions in the anterior lateral leads , thought due to fatigue and wrist pain , his anginal equivalent. Due to the patient's
increased symptoms and family history and history left main disease with total occasional of his RCA was referred for revascularization with open heart surgery.

List all the diagnoses.
"""

agent_executor.invoke({"input": sample_input})
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Thought: The patient has a history of progressive angina, a strong family history of coronary artery disease, and a previous cardiac catheterization revealing total occlusion of the RCA and 50% left main disease.
    Action:
    ```
    {
      "action": "azure_ai_services_text_analytics_for_health",
      "action_input": "The patient is a 54-year-old gentleman with a history of progressive angina over the past several months. The patient had a cardiac catheterization in July of this year revealing total occlusion of the RCA and 50% left main disease, with a strong family history of coronary artery disease with a brother dying at the age of 52 from a myocardial infarction and another brother who is status post coronary artery bypass grafting. The patient had a stress echocardiogram done on July, 2001, which showed no wall motion abnormalities, but this was a difficult study due to body habitus. The patient went for six minutes with minimal ST depressions in the anterior lateral leads, thought due to fatigue and wrist pain, his anginal equivalent. Due to the patient's increased symptoms and family history and history left main disease with total occasional of his RCA was referred for revascularization with open heart surgery."
    [0m[33;1m[1;3mThe text contains the following healthcare entities: 54-year-old is a healthcare entity of type Age, gentleman is a healthcare entity of type Gender, progressive angina is a healthcare entity of type Diagnosis, past several months is a healthcare entity of type Time, cardiac catheterization is a healthcare entity of type ExaminationName, July of this year is a healthcare entity of type Time, total is a healthcare entity of type ConditionQualifier, occlusion is a healthcare entity of type SymptomOrSign, RCA is a healthcare entity of type BodyStructure, 50 is a healthcare entity of type MeasurementValue, % is a healthcare entity of type MeasurementUnit, left main disease is a healthcare entity of type Diagnosis, family is a healthcare entity of type FamilyRelation, coronary artery disease is a healthcare entity of type Diagnosis, brother is a healthcare entity of type FamilyRelation, dying is a healthcare entity of type Diagnosis, 52 is a healthcare entity of type Age, myocardial infarction is a healthcare entity of type Diagnosis, brother is a healthcare entity of type FamilyRelation, coronary artery bypass grafting is a healthcare entity of type TreatmentName, stress echocardiogram is a healthcare entity of type ExaminationName, July, 2001 is a healthcare entity of type Time, wall motion abnormalities is a healthcare entity of type SymptomOrSign, body habitus is a healthcare entity of type SymptomOrSign, six minutes is a healthcare entity of type Time, minimal is a healthcare entity of type ConditionQualifier, ST depressions in the anterior lateral leads is a healthcare entity of type SymptomOrSign, fatigue is a healthcare entity of type SymptomOrSign, wrist pain is a healthcare entity of type SymptomOrSign, anginal is a healthcare entity of type SymptomOrSign, increased is a healthcare entity of type Course, symptoms is a healthcare entity of type SymptomOrSign, family is a healthcare entity of type FamilyRelation, left main disease is a healthcare entity of type Diagnosis, occasional is a healthcare entity of type Course, RCA is a healthcare entity of type BodyStructure, revascularization is a healthcare entity of type TreatmentName, open heart surgery is a healthcare entity of type TreatmentName[0m[32;1m[1;3m
    Action:
    ```
    {
      "action": "Final Answer",
      "action_input": "The patient's diagnoses include progressive angina, total occlusion of the RCA, 50% left main disease, coronary artery disease, myocardial infarction, and a family history of coronary artery disease."
    }
    
    [0m
    
    [1m> Finished chain.[0m
    




    {'input': "\nThe patient is a 54-year-old gentleman with a history of progressive angina over the past several months.\nThe patient had a cardiac catheterization in July of this year revealing total occlusion of the RCA and 50% left main disease ,\nwith a strong family history of coronary artery disease with a brother dying at the age of 52 from a myocardial infarction and\nanother brother who is status post coronary artery bypass grafting. The patient had a stress echocardiogram done on July , 2001 ,\nwhich showed no wall motion abnormalities , but this was a difficult study due to body habitus. The patient went for six minutes with\nminimal ST depressions in the anterior lateral leads , thought due to fatigue and wrist pain , his anginal equivalent. Due to the patient's\nincreased symptoms and family history and history left main disease with total occasional of his RCA was referred for revascularization with open heart surgery.\n\nList all the diagnoses.\n",
     'output': "The patient's diagnoses include progressive angina, total occlusion of the RCA, 50% left main disease, coronary artery disease, myocardial infarction, and a family history of coronary artery disease."}


