---
sidebar_label: Perplexity
---
# ChatPerplexity

This notebook covers how to get started with `Perplexity` chat models.


```python
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
```

The code provided assumes that your PPLX_API_KEY is set in your environment variables. If you would like to manually specify your API key and also choose a different model, you can use the following code:


```python
chat = ChatPerplexity(
    temperature=0, pplx_api_key="YOUR_API_KEY", model="llama-3-sonar-small-32k-online"
)
```

The code provided assumes that your PPLX_API_KEY is set in your environment variables. If you would like to manually specify your API key and also choose a different model, you can use the following code:

```python
chat = ChatPerplexity(temperature=0, pplx_api_key="YOUR_API_KEY", model="llama-3.1-sonar-small-128k-online")
```

You can check a list of available models [here](https://docs.perplexity.ai/docs/model-cards). For reproducibility, we can set the API key dynamically by taking it as an input in this notebook.


```python
import os
from getpass import getpass

PPLX_API_KEY = getpass()
os.environ["PPLX_API_KEY"] = PPLX_API_KEY
```


```python
chat = ChatPerplexity(temperature=0, model="llama-3.1-sonar-small-128k-online")
```


```python
system = "You are a helpful assistant."
human = "{input}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
response = chain.invoke({"input": "Why is the Higgs Boson important?"})
response.content
```




    'The Higgs Boson is an elementary subatomic particle that plays a crucial role in the Standard Model of particle physics, which accounts for three of the four fundamental forces governing the behavior of our universe: the strong and weak nuclear forces, electromagnetism, and gravity. The Higgs Boson is important for several reasons:\n\n1. **Final Elementary Particle**: The Higgs Boson is the last elementary particle waiting to be discovered under the Standard Model. Its detection helps complete the Standard Model and further our understanding of the fundamental forces in the universe.\n\n2. **Mass Generation**: The Higgs Boson is responsible for giving mass to other particles, a process that occurs through its interaction with the Higgs field. This mass generation is essential for the formation of atoms, molecules, and the visible matter we observe in the universe.\n\n3. **Implications for New Physics**: While the detection of the Higgs Boson has confirmed many aspects of the Standard Model, it also opens up new possibilities for discoveries beyond the Standard Model. Further research on the Higgs Boson could reveal insights into the nature of dark matter, supersymmetry, and other exotic phenomena.\n\n4. **Advancements in Technology**: The search for the Higgs Boson has led to significant advancements in technology, such as the development of artificial intelligence and machine learning algorithms used in particle accelerators like the Large Hadron Collider (LHC). These advancements have not only contributed to the discovery of the Higgs Boson but also have potential applications in various other fields.\n\nIn summary, the Higgs Boson is important because it completes the Standard Model, plays a crucial role in mass generation, hints at new physics phenomena beyond the Standard Model, and drives advancements in technology.\n'



You can format and structure the prompts like you would typically. In the following example, we ask the model to tell us a joke about cats.


```python
chat = ChatPerplexity(temperature=0, model="llama-3.1-sonar-small-128k-online")
prompt = ChatPromptTemplate.from_messages([("human", "Tell me a joke about {topic}")])
chain = prompt | chat
response = chain.invoke({"topic": "cats"})
response.content
```




    'Here\'s a joke about cats:\n\nWhy did the cat want math lessons from a mermaid?\n\nBecause it couldn\'t find its "core purpose" in life!\n\nRemember, cats are unique and fascinating creatures, and each one has its own special traits and abilities. While some may see them as mysterious or even a bit aloof, they are still beloved pets that bring joy and companionship to their owners. So, if your cat ever seeks guidance from a mermaid, just remember that they are on their own journey to self-discovery!\n'



## `ChatPerplexity` also supports streaming functionality:


```python
chat = ChatPerplexity(temperature=0.7, model="llama-3.1-sonar-small-128k-online")
prompt = ChatPromptTemplate.from_messages(
    [("human", "Give me a list of famous tourist attractions in Pakistan")]
)
chain = prompt | chat
for chunk in chain.stream({}):
    print(chunk.content, end="", flush=True)
```

    Here is a list of some famous tourist attractions in Pakistan:
    
    1. **Minar-e-Pakistan**: A 62-meter high minaret in Lahore that represents the history of Pakistan.
    2. **Badshahi Mosque**: A historic mosque in Lahore with a capacity of 10,000 worshippers.
    3. **Shalimar Gardens**: A beautiful garden in Lahore with landscaped grounds and a series of cascading pools.
    4. **Pakistan Monument**: A national monument in Islamabad representing the four provinces and three districts of Pakistan.
    5. **National Museum of Pakistan**: A museum in Karachi showcasing the country's cultural history.
    6. **Faisal Mosque**: A large mosque in Islamabad that can accommodate up to 300,000 worshippers.
    7. **Clifton Beach**: A popular beach in Karachi offering water activities and recreational facilities.
    8. **Kartarpur Corridor**: A visa-free border crossing and religious corridor connecting Gurdwara Darbar Sahib in Pakistan to Gurudwara Sri Kartarpur Sahib in India.
    9. **Mohenjo-daro**: An ancient Indus Valley civilization site in Sindh, Pakistan, dating back to around 2500 BCE.
    10. **Hunza Valley**: A picturesque valley in Gilgit-Baltistan known for its stunning mountain scenery and unique culture.
    
    These attractions showcase the rich history, diverse culture, and natural beauty of Pakistan, making them popular destinations for both local and international tourists.
    
