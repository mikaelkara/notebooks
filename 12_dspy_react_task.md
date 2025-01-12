<img src="images/dspy_img.png" height="35%" width="%65">

## ReAct Technique
First introduced in a paper by [Yao et al., 2022](https://arxiv.org/abs/2210.03629), ReAct is a reasoning and acting paradigm that guides LLM to respond in a structured manager to complex queries. Reasoning and actions are interleaved and progressive, in the manner of chain of thought, so that LLM progresses from one result to another, using the previous answer.

<img src="https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Freact.8e7c93ae.png&w=1920&q=75">

Results suggest that ReAct outperforms other leading methods in language and decision-making tasks, enhancing human understanding and trusting in large language models (LLMs). It is best when combined with chain-of-thought (CoT) steps as individual tasks with results being used for the next step, utilizing both internal knowledge and external information during reasoning.

Let's first use a few-shot examples to indicate how an LLM should progress with its reasoning and acting to solve a complex query, requiring intermideiate results to progress to its final outcome

**Note**: 
This DSPy notebook is conversion of [LLM ReAct prompting notebook](../llm-prompts/7_how_to_use_react_prompt.ipynb) using DSPy modules


```python
import warnings
import dspy
from dspy.teleprompt import LabeledFewShot

from dspy_utils import BOLD_BEGIN, BOLD_END, ThoughtReflection
from dspy_examples_utils import get_few_shot_dspy_examples
warnings.filterwarnings("ignore")
```

### Instantiate our Language Model


```python
# Setup OLlama environment on the local machine
ollama_mistral = dspy.OllamaLocal(model='mistral',
                                      max_tokens=5000)
# Instantiate the ColBERTv2 as Retrieval module
colbert_rm = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

# Configure the settings
dspy.settings.configure(lm=ollama_mistral, rm=colbert_rm)
```

### Get some few-shot DSPy Examples


```python
# Get some few-shot DSPy Examples for the module to learn and respond
# according
examples_set = get_few_shot_dspy_examples()
examples_set[0]
```




    Example({'question': 'What is the elevation range for the area that the eastern sector of the\n                    Colorado orogeny extends into?', 'thought_1': 'I need to search Colorado orogeny, find the area that the eastern sector\n                        of the Colorado orogeny extends into, then find the elevation range of the\n                        area.', 'action_1': 'Search[Colorado orogeny]', 'observation_1': 'The Colorado orogeny was an episode of mountain building (an orogeny) in\n                          Colorado and surrounding areas. It occurred during the Late Cretaceous and early Paleogene periods of the', 'thought_2': 'It does not mention the eastern sector. So I need to look up eastern sector.', 'action_2': 'Search[eastern sector]', 'observation_2': '(Result 1 / 1) The eastern sector extends into the High Plains and is called\n                        the Central Plains orogeny. .', 'thought_3': 'The eastern sector of Colorado orogeny extends into the High Plains. So I\n                    need to search High Plains and find its elevation range.', 'action_3': 'Search[High Plains]', 'observation_3': 'High Plains refers to one of two distinct land regions', 'thought_4': 'I need to instead search High Plains (United States).', 'action_4': 'Search[High Plains (United States)]', 'observation_4': 'The High Plains are a subregion of the Great Plains. From east to west, the\n                        High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130\n                    m).[3]', 'thought_5': ' High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer\n                    is 1,800 to 7,000 ft.', 'answer': 'Finish[1,800 to 7,000 ft]'}) (input_keys={'question'})



### Instantiate the ThoughtReflection module


```python
tought_of_reflection = ThoughtReflection()
```

### Set up a basic teleprompter optimizer 


```python
# Set up a basic teleprompter optimizer 
# and use it to compile our ReACT program.
teleprompter = LabeledFewShot(k=5)
```

### Compile the ReAct module


```python
# Compile the ReACT model
compiled_tf= teleprompter.compile(tought_of_reflection, trainset=examples_set)
```

#### Example 1: Use a in-context example for calculating the pace and rate each day of climbing Mount Kilimajaro


```python
question = """"Based on information provided to you upto 2023, 
                what is the elevation in feet of Mount Kilimanjoro?
                What is the recommended and healthy way to climb the mountain 
                in terms of ascending number of feet per day?
                and how long will it take to get to the top?"""
```


```python
answer = compiled_tf(question)
```


```python
print(f"{BOLD_BEGIN}Answer    : {BOLD_END}{answer}")
```

    [1mAnswer    : [0mPrediction(
        rationale="evaluate each attempt.\n\nStudent Attempt #1: The student correctly states that Mount Kilimanjaro has an elevation of approximately 19,341 feet (5,895 meters) and provides a prediction for the number of days it would take to reach the summit based on a daily ascent of no more than 3,000 feet. However, the student does not mention the need for acclimatization days, which are essential when climbing at high altitudes.\n\nStudent Attempt #2: The student's prediction is consistent with the given information about Mount Kilimanjaro's elevation and the recommended daily ascent rate. However, there's no mention of acclimatization days or their importance in the mountaineering context.\n\nStudent Attempt #3: This student's attempt is similar to the previous ones. They correctly state Mount Kilimanjaro's elevation and provide a prediction for the number of days it would take to reach the summit while considering daily ascent rates and acclimatization days. However, they don't specify that their prediction includes acclimatization days, which could lead to confusion.\n\nStudent Attempt #4: This student provides an accurate and detailed answer. They state Mount Kilimanjaro's elevation, recommend a daily ascent rate, and explain the importance of acclimatization days in the context of mountaineering. Their prediction includes both climbing and acclimatization days, making it more comprehensive than others.\n\nStudent Attempt #5: This student's attempt is almost identical to the correct answer. They correctly state Mount Kilimanjaro's elevation, recommend a daily ascent rate, and provide an accurate prediction for the number of days it would take to reach the summit while accounting for acclimatization days.\n\nRanking: 1. Student Attempt #4; 2. Student Attempt #5; 3. Student Attempt #1 (tie) and Student Attempt #3; \n\nExplanation: The correct answer should take into account all the given factors, such as daily ascent rates and acclimatization days. Therefore, Student Attempts #4 and #5 are ranked higher since they provide a more comprehensive answer that accounts for both factors. Student Attempt #1 is ranked third because it does not mention acclimatization days, which is an essential aspect of mountaineering. Finally, Student Attempt #2 is not ranked as it does not provide a prediction or an accurate reasoning based on the given information.",
        answer='The correct answer would be similar to Student Attempts #4 and #5, taking into account both daily ascent rates and acclimatization days when predicting the number of days it would take to reach the summit of Mount Kilimanjaro. For example, a common itinerary for climbing Mount Kilimanjaro includes 2-3 days for acclimatization and 4-5 days for summit ascents, making the total time around 6-7 days. However, individual climbing plans may vary depending on factors such as fitness level, experience, and personal preferences.'
    )
    

### Examine optimized prompts generated for the LLMs


```python
# Examine the history of the prompts generated by the ReACT model
print(ollama_mistral.inspect_history(n=1))
```

    
    
    
    Given the fields `question`, produce the fields `answer`.
    
    ---
    
    Question: Based on information provided to you upto 2023, Who was Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?
    Answer: Finish[2.66]
    
    Question: Based on information provided to you upto 2023, how do you calculate the value of PI. With its current value of PI, given a circle of diameter 2 meters, what its its circumcernce and area?
    Answer: Finish[Circumference: 6.28318 meters, Area: 3.14159 square meters]
    
    Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
    Answer: Finish[1,800 to 7,000 ft]
    
    ---
    
    Follow the following format.
    
    Question: ${question}
    
    Student Attempt #1: ${reasoning attempt}
    
    Student Attempt #2: ${reasoning attempt}
    
    Student Attempt #3: ${reasoning attempt}
    
    Student Attempt #4: ${reasoning attempt}
    
    Student Attempt #5: ${reasoning attempt}
    
    Accurate Reasoning: Thank you everyone. Let's now holistically ${corrected reasoning}
    
    Answer: ${answer}
    
    ---
    
    Question: "Based on information provided to you upto 2023, what is the elevation in feet of Mount Kilimanjoro? What is the recommended and healthy way to climb the mountain in terms of ascending number of feet per day? and how long will it take to get to the top?
    
    Student Attempt #1: «I'm trying to provide an answer. I'm not sure but my prediction is The elevation of Mount Kilimanjaro is approximately 19,341 feet (5,895 meters), and it takes about 6 to 7 days to reach the top in a healthy way.»
    
    Student Attempt #2: «I'm trying to answer the question about Mount Kilimanjaro. The elevation of Mount Kilimanjaro is approximately 19,341 feet (5,895 meters). The recommended rate of ascent for climbers is around 3,000 feet per day, with an additional acclimatization day every 2,000 feet. Based on this information, it would take approximately 7 days to reach the summit, including rest days for acclimatization. I'm not sure but my prediction is The elevation of Mount Kilimanjaro is around 19,341 feet (5,895 meters). To safely climb the mountain, it's recommended to ascend no more than 3,000 feet per day, with an additional acclimatization day every 2,000 feet. Following this guideline, it would take approximately 7 days to reach the summit of Mount Kilimanjaro.»
    
    Student Attempt #3: «I'm trying to answer the question about Mount Kilimanjaro. I'm not sure but my prediction is The elevation of Mount Kilimanjaro is approximately 19,341 feet (5,895 meters). To reach the summit in a recommended and healthy manner, aim for a daily elevation gain of no more than 3,000 feet per day. This would mean it takes approximately 7 days to climb Mount Kilimanjaro.»
    
    Student Attempt #4: «I'm trying to Answer the question about Mount Kilimanjaro. I'm not sure but my prediction is The elevation of Mount Kilimanjaro is approximately 19,341 feet (5,895 meters). A healthy and recommended ascent rate for climbers is around 3,000 feet (914 meters) per day. With this rate, it would take about 7 days to reach the summit of Mount Kilimanjaro from the base camp. However, this timeframe includes acclimatization days, so the actual climbing days may be fewer. It is important for climbers to adjust their ascent rates based on personal fitness and altitude tolerance.»
    
    Student Attempt #5: «I'm trying to Answer: [Mount Kilimanjaro's elevation is approximately 19,341 feet (5,895 meters). A safe and recommended climbing itinerary suggests ascending no more than 3,000 feet per day. Given this information, it would take approximately six days to reach the summit.] I'm not sure but my prediction is [Mount Kilimanjaro's elevation is approximately 19,341 feet (5,895 meters). It is recommended to ascend no more than 3,000 feet per day. Therefore, it would take about six days to reach the summit.]»
    
    Accurate Reasoning: Thank you everyone. Let's now holistically evaluate each attempt. Student Attempt #1: The student correctly states that Mount Kilimanjaro has an elevation of approximately 19,341 feet (5,895 meters) and provides a prediction for the number of days it would take to reach the summit based on a daily ascent of no more than 3,000 feet. However, the student does not mention the need for acclimatization days, which are essential when climbing at high altitudes. Student Attempt #2: The student's prediction is consistent with the given information about Mount Kilimanjaro's elevation and the recommended daily ascent rate. However, there's no mention of acclimatization days or their importance in the mountaineering context. Student Attempt #3: This student's attempt is similar to the previous ones. They correctly state Mount Kilimanjaro's elevation and provide a prediction for the number of days it would take to reach the summit while considering daily ascent rates and acclimatization days. However, they don't specify that their prediction includes acclimatization days, which could lead to confusion. Student Attempt #4: This student provides an accurate and detailed answer. They state Mount Kilimanjaro's elevation, recommend a daily ascent rate, and explain the importance of acclimatization days in the context of mountaineering. Their prediction includes both climbing and acclimatization days, making it more comprehensive than others. Student Attempt #5: This student's attempt is almost identical to the correct answer. They correctly state Mount Kilimanjaro's elevation, recommend a daily ascent rate, and provide an accurate prediction for the number of days it would take to reach the summit while accounting for acclimatization days. Ranking: 1. Student Attempt #4; 2. Student Attempt #5; 3. Student Attempt #1 (tie) and Student Attempt #3; Explanation: The correct answer should take into account all the given factors, such as daily ascent rates and acclimatization days. Therefore, Student Attempts #4 and #5 are ranked higher since they provide a more comprehensive answer that accounts for both factors. Student Attempt #1 is ranked third because it does not mention acclimatization days, which is an essential aspect of mountaineering. Finally, Student Attempt #2 is not ranked as it does not provide a prediction or an accurate reasoning based on the given information.
    
    Answer:[32m The correct answer would be similar to Student Attempts #4 and #5, taking into account both daily ascent rates and acclimatization days when predicting the number of days it would take to reach the summit of Mount Kilimanjaro. For example, a common itinerary for climbing Mount Kilimanjaro includes 2-3 days for acclimatization and 4-5 days for summit ascents, making the total time around 6-7 days. However, individual climbing plans may vary depending on factors such as fitness level, experience, and personal preferences.[0m
    
    
    
    
    
    
    Given the fields `question`, produce the fields `answer`.
    
    ---
    
    Question: Based on information provided to you upto 2023, Who was Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?
    Answer: Finish[2.66]
    
    Question: Based on information provided to you upto 2023, how do you calculate the value of PI. With its current value of PI, given a circle of diameter 2 meters, what its its circumcernce and area?
    Answer: Finish[Circumference: 6.28318 meters, Area: 3.14159 square meters]
    
    Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
    Answer: Finish[1,800 to 7,000 ft]
    
    ---
    
    Follow the following format.
    
    Question: ${question}
    
    Student Attempt #1: ${reasoning attempt}
    
    Student Attempt #2: ${reasoning attempt}
    
    Student Attempt #3: ${reasoning attempt}
    
    Student Attempt #4: ${reasoning attempt}
    
    Student Attempt #5: ${reasoning attempt}
    
    Accurate Reasoning: Thank you everyone. Let's now holistically ${corrected reasoning}
    
    Answer: ${answer}
    
    ---
    
    Question: "Based on information provided to you upto 2023, what is the elevation in feet of Mount Kilimanjoro? What is the recommended and healthy way to climb the mountain in terms of ascending number of feet per day? and how long will it take to get to the top?
    
    Student Attempt #1: «I'm trying to provide an answer. I'm not sure but my prediction is The elevation of Mount Kilimanjaro is approximately 19,341 feet (5,895 meters), and it takes about 6 to 7 days to reach the top in a healthy way.»
    
    Student Attempt #2: «I'm trying to answer the question about Mount Kilimanjaro. The elevation of Mount Kilimanjaro is approximately 19,341 feet (5,895 meters). The recommended rate of ascent for climbers is around 3,000 feet per day, with an additional acclimatization day every 2,000 feet. Based on this information, it would take approximately 7 days to reach the summit, including rest days for acclimatization. I'm not sure but my prediction is The elevation of Mount Kilimanjaro is around 19,341 feet (5,895 meters). To safely climb the mountain, it's recommended to ascend no more than 3,000 feet per day, with an additional acclimatization day every 2,000 feet. Following this guideline, it would take approximately 7 days to reach the summit of Mount Kilimanjaro.»
    
    Student Attempt #3: «I'm trying to answer the question about Mount Kilimanjaro. I'm not sure but my prediction is The elevation of Mount Kilimanjaro is approximately 19,341 feet (5,895 meters). To reach the summit in a recommended and healthy manner, aim for a daily elevation gain of no more than 3,000 feet per day. This would mean it takes approximately 7 days to climb Mount Kilimanjaro.»
    
    Student Attempt #4: «I'm trying to Answer the question about Mount Kilimanjaro. I'm not sure but my prediction is The elevation of Mount Kilimanjaro is approximately 19,341 feet (5,895 meters). A healthy and recommended ascent rate for climbers is around 3,000 feet (914 meters) per day. With this rate, it would take about 7 days to reach the summit of Mount Kilimanjaro from the base camp. However, this timeframe includes acclimatization days, so the actual climbing days may be fewer. It is important for climbers to adjust their ascent rates based on personal fitness and altitude tolerance.»
    
    Student Attempt #5: «I'm trying to Answer: [Mount Kilimanjaro's elevation is approximately 19,341 feet (5,895 meters). A safe and recommended climbing itinerary suggests ascending no more than 3,000 feet per day. Given this information, it would take approximately six days to reach the summit.] I'm not sure but my prediction is [Mount Kilimanjaro's elevation is approximately 19,341 feet (5,895 meters). It is recommended to ascend no more than 3,000 feet per day. Therefore, it would take about six days to reach the summit.]»
    
    Accurate Reasoning: Thank you everyone. Let's now holistically evaluate each attempt. Student Attempt #1: The student correctly states that Mount Kilimanjaro has an elevation of approximately 19,341 feet (5,895 meters) and provides a prediction for the number of days it would take to reach the summit based on a daily ascent of no more than 3,000 feet. However, the student does not mention the need for acclimatization days, which are essential when climbing at high altitudes. Student Attempt #2: The student's prediction is consistent with the given information about Mount Kilimanjaro's elevation and the recommended daily ascent rate. However, there's no mention of acclimatization days or their importance in the mountaineering context. Student Attempt #3: This student's attempt is similar to the previous ones. They correctly state Mount Kilimanjaro's elevation and provide a prediction for the number of days it would take to reach the summit while considering daily ascent rates and acclimatization days. However, they don't specify that their prediction includes acclimatization days, which could lead to confusion. Student Attempt #4: This student provides an accurate and detailed answer. They state Mount Kilimanjaro's elevation, recommend a daily ascent rate, and explain the importance of acclimatization days in the context of mountaineering. Their prediction includes both climbing and acclimatization days, making it more comprehensive than others. Student Attempt #5: This student's attempt is almost identical to the correct answer. They correctly state Mount Kilimanjaro's elevation, recommend a daily ascent rate, and provide an accurate prediction for the number of days it would take to reach the summit while accounting for acclimatization days. Ranking: 1. Student Attempt #4; 2. Student Attempt #5; 3. Student Attempt #1 (tie) and Student Attempt #3; Explanation: The correct answer should take into account all the given factors, such as daily ascent rates and acclimatization days. Therefore, Student Attempts #4 and #5 are ranked higher since they provide a more comprehensive answer that accounts for both factors. Student Attempt #1 is ranked third because it does not mention acclimatization days, which is an essential aspect of mountaineering. Finally, Student Attempt #2 is not ranked as it does not provide a prediction or an accurate reasoning based on the given information.
    
    Answer:[32m The correct answer would be similar to Student Attempts #4 and #5, taking into account both daily ascent rates and acclimatization days when predicting the number of days it would take to reach the summit of Mount Kilimanjaro. For example, a common itinerary for climbing Mount Kilimanjaro includes 2-3 days for acclimatization and 4-5 days for summit ascents, making the total time around 6-7 days. However, individual climbing plans may vary depending on factors such as fitness level, experience, and personal preferences.[0m
    
    
    
    

#### Example 2: Use a in-context example for quering history of an Apple gadget from its existing knowledge base


```python
question = """
            Based on information provided to you upto 2023, aside from the Apple Remote, what other devices can 
            control the program Apple Remote was originally designed to interact with?"""
```


```python
answer = compiled_tf(question)
```


```python
print(f"{BOLD_BEGIN}Answer    : {BOLD_END}{answer}")
```


```python
# Examine the history of the prompts generated by the ReACT model
print(ollama_mistral.inspect_history(n=1))
```

#### Example 3: Use an in-context example for converting `HH:MM:SS` string format into seconds using ReAct prompting


```python
question = """Convert a time string with format H:MM:SS to seconds. How do you convert 3:56:25 into seconds?
"""
```


```python
answer = compiled_tf(question)
```


```python
print(f"{BOLD_BEGIN}Answer    : {BOLD_END}{answer}")
```


```python
# Examine the history of the prompts generated by the ReACT model
print(ollama_mistral.inspect_history(n=1))
```

## All this is amazing! 😜 Feel the wizardy in DSPy power 🧙‍♀️
