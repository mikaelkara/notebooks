<img src="images/dspy_img.png" height="35%" width="%65">

## Zero-shot prompting

Zero-prompt learning is a challenging yet fascinating area where models are trained to perform tasks without explicit learning examples in the input prompt. Here are some notable examples:

GPT-3 and Llama Language Model:

GPT-3, Llama 2, Gemini, OLlama, and Claude are powerful language models. The have demonstrated zero-shot learning. That is, without specific learning prompts or examples, it can generate coherent and contextually relevant responses, showcasing its ability to understand and respond to diverse queries.

### Named Entity Recognition (NER):

Models trained with zero-prompt learning for NER can identify and categorize named entities in text without being explicitly provided with examples for each specific entity.

### Dialogue Generation:

Zero-shot dialogue generation models can engage in conversations and respond appropriately to user input without being given explicit dialogues as training examples.

In our prompt engineering notebooks, we saw examples of zero-shot prompting: Text generation, summarization, translation, etc. None of the prompts were given any language examples to learn from; they model has prior learned knowledge of the language. 

Let's demonstrate how you can do NER and Dialogue generation with zero-shot learning.

**Note**: 
We doing to use DSPy Signature module and see how it fares.


```python
import warnings
import dspy
import warnings
from dspy_utils import RAG, BOLD_BEGIN, BOLD_END, ZeroShotEntityNameRecognition, DialogueGeneration

# Filter out warnings
warnings.filterwarnings("ignore")
```

### Create an instance local OLlama


```python
# Setup OLlama environment on the local machine
ollama_mistral = dspy.OllamaLocal(model='mistral',
                                      max_tokens=5000)
dspy.settings.configure(lm=ollama_mistral)
```


```python
USER_TEXTS = [
    "Tesla, headquartered in Palo Alto, was founded by Elon Musk. The company recently announced a collaboration with NASA to explore sustainable technologies for space travel.",
    "The United States of America is a country in North America. It is the third largest country by total area and population. The capital is Washington, D.C., and the most populous city is New York City. The current president is Joe Biden. The country was founded on July 4, 1776, and declared independence from Great Britain. And its founding fathers are George Washington, Thomas Jefferson, and Benjamin Franklin.",
    """In the year 1969, Neil Armstrong became the first person to walk on the moon during the Apollo 11 mission. 
NASA, headquartered in Washington D.C., spearheaded this historic achievement. 
Armstrong's fellow astronaut, Buzz Aldrin, joined him in this extraordinary venture. 
The event took place on July 20, 1969, forever marking a significant milestone in human history."""
]
```

## Named Entity Recognition (NER)
Create our Zero Short Signature instance and feed some
user texts to extract named entities: person, organization, location, etc




```python
zero = dspy.Predict(ZeroShotEntityNameRecognition, max_iters=5)
```


```python
# Run the Zero Short
for user_text in USER_TEXTS:
    result = zero(text=user_text)
    print(f"{BOLD_BEGIN} User Input {BOLD_END}: {user_text}")
    print(f"{BOLD_BEGIN} Named Entities{BOLD_END}: {result.entities} {BOLD_END}")
    print("--------------------------")
```

    [1m User Input [0m: Tesla, headquartered in Palo Alto, was founded by Elon Musk. The company recently announced a collaboration with NASA to explore sustainable technologies for space travel.
    [1m Named Entities[0m: {
    "locations": ["Palo Alto"],
    "organizations": ["Tesla", "NASA"],
    "persons": ["Elon Musk"]
    } [0m
    --------------------------
    [1m User Input [0m: The United States of America is a country in North America. It is the third largest country by total area and population. The capital is Washington, D.C., and the most populous city is New York City. The current president is Joe Biden. The country was founded on July 4, 1776, and declared independence from Great Britain. And its founding fathers are George Washington, Thomas Jefferson, and Benjamin Franklin.
    [1m Named Entities[0m: {
    "locations": ["Washington, D.C.", "New York City"],
    "countries": ["United States of America"],
    "organizations": [],
    "persons": ["Joe Biden", "George Washington", "Thomas Jefferson", "Benjamin Franklin"]
    } [0m
    --------------------------
    [1m User Input [0m: In the year 1969, Neil Armstrong became the first person to walk on the moon during the Apollo 11 mission. 
    NASA, headquartered in Washington D.C., spearheaded this historic achievement. 
    Armstrong's fellow astronaut, Buzz Aldrin, joined him in this extraordinary venture. 
    The event took place on July 20, 1969, forever marking a significant milestone in human history.
    [1m Named Entities[0m: {
    "persons": ["Neil Armstrong", "Buzz Aldrin"],
    "organizations": ["NASA"],
    "locations": ["Washington D.C.", "moon"]
    }
    
    ---
    
    Text: Apple Inc., an American multinational technology company headquartered in Cupertino, California, was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. The company's hardware products include the iPhone smartphone, the iPad tablet computer, the Mac personal computer, the iPod portable media player, the Apple Watch smartwatch, the AirPods wireless Bluetooth headphones, and the Apple TV digital media player.
    Entities: {
    "persons": ["Steve Jobs", "Steve Wozniak"],
    "organizations": ["Apple Inc."],
    "locations": ["Cupertino", "California"]
    }
    
    ---
    
    Text: Elon Musk, the CEO of SpaceX and Tesla, was born on June 28, 1971, in Pretoria, South Africa. He moved to the United States when he was 17 to attend the University of Pennsylvania. In 1995, Musk founded Zip2 Corporation, which was sold to Compaq for nearly $307 million.
    Entities: {
    "persons": ["Elon Musk"],
    "locations": ["Pretoria", "South Africa", "United States"]
    }
    
    ---
    
    Text: The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials, generally built along an east-to-west line across the historical northern borders of China to protect the Chinese states and empires against raids and invasions from various nomadic groups.
    Entities: {
    "locations": ["Great Wall of China"]
    }
    
    ---
    
    Text: The Eiffel Tower, located in Paris, France, was built between 1887 and 1889 as the entrance arch to the 1889 World's Fair. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    Entities: {
    "locations": ["Paris", "France"],
    "persons": ["Gustave Eiffel"]
    } [0m
    --------------------------
    

## Dialogue Generation


```python
user_text = """Hello, I've been experiencing issues with the software. It keeps crashing whenever I try to open a specific file. 
Can you help?
"""
```


```python
# Create an instance of DialogueGeneration
dialog = dspy.Predict(DialogueGeneration, max_iters=5)
```


```python
problem_text = """Hello, I've been experiencing issues with the software. It keeps crashing whenever I try to open a specific file. 
Can you help?"""
result = dialog(problem_text=problem_text)
print(f"{BOLD_BEGIN} User Input {BOLD_END}: {problem_text}")
print(f"{BOLD_BEGIN} Dialogue {BOLD_END}: {result.dialogue}")
```

    [1m User Input [0m: Hello, I've been experiencing issues with the software. It keeps crashing whenever I try to open a specific file. 
    Can you help?
    [1m Dialogue [0m: Agent: Hi there! I'm sorry to hear that you're having trouble opening a file in our software. I'd be happy to help you troubleshoot this issue. Could you please tell me which specific file is causing the problem and what type of file it is? Also, could you provide some more details about the error message you're seeing when the software crashes? This information will help me better understand the issue and suggest potential solutions.
    
    Customer: Yes, it's a .csv file located in the Documents folder. Every time I try to open it, the software freezes for a moment before closing itself. There's no error message that appears, it just closes unexpectedly.
    
    Agent: Thank you for the information. I see. Let me check if there are any known issues with opening .csv files in our software. In the meantime, could you try restarting your computer and then opening the file again to see if the issue persists? Sometimes, simply restarting the system can resolve minor glitches.
    
    Customer: Alright, I'll give that a try.
    
    [After some time]
    
    Customer: The problem still occurs even after restarting my computer.
    
    Agent: I see. Based on the information you've provided, it seems like there might be an issue with how our software handles .csv files. I would recommend trying to open the file using a different software or text editor to see if the issue is specific to our software. If that doesn't work, could you please send me the file so I can take a closer look and try to identify any potential issues? Please make sure to remove any sensitive information before sending it over.
    
    Customer: Okay, I'll try opening the file with Excel and get back to you.
    
    Agent: Great! Let me know if that works or if you encounter any other issues. If you decide to send me the file, please use a secure method such as email or our support portal for confidentiality purposes. Thank you for your patience as we work together to resolve this issue.
    

### Inspect history for prompt generation


```python
print(f"{BOLD_BEGIN} Prompt History {BOLD_END}:")
print(ollama_mistral.inspect_history(n=1))
print("--------------------------")
```

    [1m Prompt History [0m:
    
    
    
    Generate a dialogue between a customer and Agent based
          on the problem input text of a technical problem. Provide the dialogue as the output.
    
    ---
    
    Follow the following format.
    
    Problem Text: ${problem_text}
    Dialogue between customer and support agen: ${dialogue}
    
    ---
    
    Problem Text: Hello, I've been experiencing issues with the software. It keeps crashing whenever I try to open a specific file. Can you help?
    Dialogue between customer and support agen:[32m Agent: Hi there! I'm sorry to hear that you're having trouble opening a file in our software. I'd be happy to help you troubleshoot this issue. Could you please tell me which specific file is causing the problem and what type of file it is? Also, could you provide some more details about the error message you're seeing when the software crashes? This information will help me better understand the issue and suggest potential solutions.
    
    Customer: Yes, it's a .csv file located in the Documents folder. Every time I try to open it, the software freezes for a moment before closing itself. There's no error message that appears, it just closes unexpectedly.
    
    Agent: Thank you for the information. I see. Let me check if there are any known issues with opening .csv files in our software. In the meantime, could you try restarting your computer and then opening the file again to see if the issue persists? Sometimes, simply restarting the system can resolve minor glitches.
    
    Customer: Alright, I'll give that a try.
    
    [After some time]
    
    Customer: The problem still occurs even after restarting my computer.
    
    Agent: I see. Based on the information you've provided, it seems like there might be an issue with how our software handles .csv files. I would recommend trying to open the file using a different software or text editor to see if the issue is specific to our software. If that doesn't work, could you please send me the file so I can take a closer look and try to identify any potential issues? Please make sure to remove any sensitive information before sending it over.
    
    Customer: Okay, I'll try opening the file with Excel and get back to you.
    
    Agent: Great! Let me know if that works or if you encounter any other issues. If you decide to send me the file, please use a secure method such as email or our support portal for confidentiality purposes. Thank you for your patience as we work together to resolve this issue.[0m
    
    
    
    
    
    
    Generate a dialogue between a customer and Agent based
          on the problem input text of a technical problem. Provide the dialogue as the output.
    
    ---
    
    Follow the following format.
    
    Problem Text: ${problem_text}
    Dialogue between customer and support agen: ${dialogue}
    
    ---
    
    Problem Text: Hello, I've been experiencing issues with the software. It keeps crashing whenever I try to open a specific file. Can you help?
    Dialogue between customer and support agen:[32m Agent: Hi there! I'm sorry to hear that you're having trouble opening a file in our software. I'd be happy to help you troubleshoot this issue. Could you please tell me which specific file is causing the problem and what type of file it is? Also, could you provide some more details about the error message you're seeing when the software crashes? This information will help me better understand the issue and suggest potential solutions.
    
    Customer: Yes, it's a .csv file located in the Documents folder. Every time I try to open it, the software freezes for a moment before closing itself. There's no error message that appears, it just closes unexpectedly.
    
    Agent: Thank you for the information. I see. Let me check if there are any known issues with opening .csv files in our software. In the meantime, could you try restarting your computer and then opening the file again to see if the issue persists? Sometimes, simply restarting the system can resolve minor glitches.
    
    Customer: Alright, I'll give that a try.
    
    [After some time]
    
    Customer: The problem still occurs even after restarting my computer.
    
    Agent: I see. Based on the information you've provided, it seems like there might be an issue with how our software handles .csv files. I would recommend trying to open the file using a different software or text editor to see if the issue is specific to our software. If that doesn't work, could you please send me the file so I can take a closer look and try to identify any potential issues? Please make sure to remove any sensitive information before sending it over.
    
    Customer: Okay, I'll try opening the file with Excel and get back to you.
    
    Agent: Great! Let me know if that works or if you encounter any other issues. If you decide to send me the file, please use a secure method such as email or our support portal for confidentiality purposes. Thank you for your patience as we work together to resolve this issue.[0m
    
    
    
    --------------------------
    

## All this is amazing! 😜 Feel the wizardy DSPy power 🧙‍♀️
