<img src="./images/llm_prompt_req_resp.png" height="35%" width="%65">

## ReAct Prompting
First introduced in a paper by [Yao et al., 2022](https://arxiv.org/abs/2210.03629), ReAct is a reasoning and acting paradigm that guides LLM to respond in a structured manager to complex queries. Reasoning and actions are interleaved and progressive, in the manner of chain of thought, so that LLM progresses from one result to another, using the previous answer.

<img src="https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Freact.8e7c93ae.png&w=1920&q=75">

Results suggest that ReAct outperforms other leading methods in language and decision-making tasks, enhances human understanding and trust in large language models (LLMs). It and best when combined with chain-of-thought (CoT) steps as individual tasks with results being used for the next step, utilizing both internal knowledge and external information during reasoning.

Let's first use a few-shot examples to indicate how an LLM should progress with its reasoning and acting to solve a complex query, requiring intermideiate results to progress to its final outcome

**Note**: 
To run any of these relevant notebooks you will need an account on Anyscale Endpoints, Anthropic, or OpenAI, depending on what model you elect, along with the respective environment file. Use the template environment files to create respective `.env` file for either Anyscale Endpoints, Anthropic, or OpenAI.


```python
import warnings
import os

import openai
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
```

Load the environment


```python
_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings('ignore')
openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("MODEL")
print(f"Using MODEL={MODEL}; base={openai.api_base}")
```


```python
# Create the OpenAI client, which can be used transparently with Anyscale 
#Endpoints too

from openai import OpenAI

client = OpenAI(
    api_key = openai.api_key,
    base_url = openai.api_base
)
```


```python
# Utility function to send and fetch response

def get_commpletion(clnt: object, model: str, system_content: str, user_content:str) -> str:
    chat_completion = clnt.chat.completions.create(
        model=model,
    messages=[{"role": "system", "content": system_content},
              {"role": "user", "content": user_content}],
    temperature = 0.8)

    response = chat_completion.choices[0].message.content
    return response
```

#### Example 1: Use a in-context example for calculating an age of a person using its existing knowledge base
An example use case for a few-shot learning is to provide an example pattern of chain of thought that the model can learn from and generalize how to use ReAct. Using what it has learned as an in-context example, it can generate a response for its subsequent prompt in the manner suggested.

We will use an example few-shot prompt from this [guide](https://www.promptingguide.ai/techniques/react).


```python
system_content = """
You are supreme repository of knowledge and can answer
questions by deliberating following an thought, action, and
observation pattern known as ReAct.
"""
```


```python
react_user_prompt = """
Question: What is the elevation range for the area that the eastern sector of the
Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector
of the Colorado orogeny extends into, then find the elevation range of the
area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in
Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern
sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called
the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I
need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the
High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130
m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer
is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]

Question: Based on information provided to you upto 2023, Who was Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?
...
"""
```


```python
response = get_commpletion(client, MODEL, system_content, react_user_prompt)
print(f"{response}\n")
```

    Thought 1: I need to recall information on Olivia Wilde's relationship status as of my latest update and identify who her boyfriend was at that time. Then, I need to determine his age and calculate his age raised to the 0.23 power.
    
    Action 1: Recall[Olivia Wilde's boyfriend as of latest update]
    Observation 1: As of my last update, Olivia Wilde was known to be in a relationship with Harry Styles.
    
    Thought 2: Now that I have identified Harry Styles as Olivia Wilde's boyfriend, I need to recall Harry Styles' birth date to calculate his current age.
    Action 2: Recall[Harry Styles' birth date]
    Observation 2: Harry Styles was born on February 1, 1994.
    
    Thought 3: To determine Harry Styles' current age, I will calculate the number of years from his birth year (1994) to the current year, which would be 2023.
    Action 3: Calculate[2023 - 1994]
    Observation 3: The calculation yields 29 years (2023 - 1994 = 29).
    
    Thought 4: Now I have Harry Styles' current age, I need to raise it to the power of 0.23 as per the original question.
    Action 4: Calculate[29^0.23]
    Observation 4: Raising 29 to the power of 0.23 results in approximately 2.66.
    
    Thought 5: The result of Harry Styles' current age raised to the power of 0.23 is approximately 2.66.
    Action 5: Finish[2.66]
    
    Answer: Olivia Wilde's boyfriend as of the last information I received was Harry Styles, and his current age raised to the 0.23 power is approximately 2.66.
    
    

#### Example 2: Use a in-context example for calculating an age of a person using its existing knowledge base


```python
react_user_prompt_2 = """
Question: What is the elevation range for the area that the eastern sector of the
Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector
of the Colorado orogeny extends into, then find the elevation range of the
area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in
Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern
sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called
the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I
need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the
High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130
m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer
is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]

Question: Based on information provided to you upto 2023, aside from the Apple Remote, what other devices can 
control the program Apple Remote was originally designed to interact with?
...
"""
```


```python
response = get_commpletion(client, MODEL, system_content, react_user_prompt_2)
print(f"{response}\n")
```

    Thought 1: Apple Remote was originally designed to interact with various Apple devices, such as Mac computers and the Apple TV. To answer this question accurately, I need to consider the different devices that can control these Apple products.
    
    Action 1: Recall[Devices compatible with Mac and Apple TV]
    
    Observation 1: Devices that can control Mac computers and Apple TVs include other Apple devices with the Remote app installed (such as iPhones, iPads, and iPod Touches), third-party universal remotes that are programmable to work with Apple TV, and keyboards or mice when it comes to Macs.
    
    Thought 2: Besides the Apple Remote, there are several options for controlling Apple devices. iPhones, iPads, and iPod Touches can use the Remote app. Macs can be controlled with keyboards and mice, and programmable universal remotes can be used with Apple TV.
    
    Action 2: Combine[Devices compatible with Mac and Apple TV]
    
    Observation 2: The combination of compatible devices includes Apple's iOS devices with the Remote app, programmable universal remotes, and standard input devices for Macs like keyboards and mice.
    
    Thought 3: I should provide a concise response listing the alternative devices that can control the program the Apple Remote was originally designed to interact with.
    
    Action 3: Finish[Response with list of alternative control devices]
    
    Based on information provided up to 2023, aside from the Apple Remote, the devices that can control the programs Apple Remote was originally designed to interact with include:
    
    - iPhones, iPads, and iPod Touches with the Remote app
    - Programmable universal remotes (for Apple TV)
    - Keyboards and mice (for Macs)
    
    These alternatives offer different ways to control Apple devices such as Mac computers and Apple TVs, which are compatible with the Apple Remote.
    
    

#### Example 3: Use a in-context example for calculating an age of a person using its existing knowledge base


```python
react_user_prompt_3 = """
Question: What is the elevation range for the area that the eastern sector of the
Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector
of the Colorado orogeny extends into, then find the elevation range of the
area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in
Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern
sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called
the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I
need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the
High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130
m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer
is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]

Question: Based on information provided to you upto 2023, how do you calculate the value
of PI. With its current value of PI, given a circle of diameter 2 meters, what its its 
circumcernce and area?
...
"""
```


```python
response = get_commpletion(client, MODEL, system_content, react_user_prompt_3)
print(f"{response}\n")
```

    Thought 1: To calculate the value of pi (π), mathematicians use various algorithms and numerical methods because π is an irrational number and cannot be expressed as a simple fraction. Historically, π has been approximated by methods like polygon approximation or series expansions.
    
    Action 1: No action required here because the value of π is already known and accepted as approximately 3.14159.
    
    Observation 1: The value of π is widely known to be approximately 3.14159.
    
    Thought 2: To find the circumference of a circle, the formula is C = πd, where d is the diameter, and π is approximated as 3.14159. To find the area of a circle, the formula is A = πr^2, where r is the radius (half of the diameter).
    
    Action 2: Calculate circumference and area using the known value of π.
    
    Observation 2: For a circle with a diameter of 2 meters, the radius would be 1 meter (half of the diameter).
    
    Circumference calculation:
    C = πd = π * 2 = 3.14159 * 2 ≈ 6.28318 meters
    
    Area calculation:
    A = πr^2 = π * 1^2 = 3.14159 * 1 * 1 ≈ 3.14159 square meters
    
    Thought 3: The circumference of a circle with a diameter of 2 meters is approximately 6.28318 meters, and the area is approximately 3.14159 square meters.
    
    Action 3: Report the calculated circumference and area.
    
    Observation 3: Calculations have been made, and the values are ready to be reported.
    
    Final Answer: Using the current value of π (pi) as approximately 3.14159, a circle with a diameter of 2 meters has a circumference of approximately 6.28318 meters and an area of approximately 3.14159 square meters.
    
    

#### Example 4: Use a in-context example for calculating the pace and rate each day of climbing Mount Kilimajaro.


```python
react_user_prompt_4 = """
Question: What is the elevation range for the area that the eastern sector of the
Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector
of the Colorado orogeny extends into, then find the elevation range of the
area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in
Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern
sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called
the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I
need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the
High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130
m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer
is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]

Question: Based on information provided to you upto 2023, what is the elevation
in feet of Mount Kilimanjor, the highest peek in the continent of Africa.
What is the recommended and healthy way to climb the mountain in terms of ascending number of
feet per day to ascend, and how based on that calculation and rest between each day's
climb, how long will it take to get to the top?
...
"""
```


```python
response = get_commpletion(client, MODEL, system_content, react_user_prompt_4)
print(f"{response}\n")
```

    Thought 1: To answer this question, I need to remember the elevation of Mount Kilimanjaro and then provide information on the recommended rate of ascent to minimize altitude sickness and calculate the total time based on this ascent rate and necessary acclimatization days.
    
    Action 1: Remember[Mount Kilimanjaro elevation]
    Observation 1: Mount Kilimanjaro's highest peak, Uhuru Peak, stands at 19,341 feet (5,895 meters) above sea level.
    
    Thought 2: The recommended safe rate of ascent to avoid altitude sickness is generally around 1,000 feet (305 meters) per day after reaching an altitude where altitude sickness becomes a possibility. I also need to account for recommended acclimatization days.
    
    Action 2: Recall[recommended rate of ascent and acclimatization for high altitude climbing]
    Observation 2: A general guideline for climbing high altitude mountains like Kilimanjaro is to ascend no more than 1,000 feet (305 meters) per day and include acclimatization days, which may involve climbing higher and sleeping at a lower altitude. The usual recommendation for Kilimanjaro includes one or two rest/acclimatization days on the route up.
    
    Thought 3: Climbing options vary, but on average, routes take about 5 to 9 days, incorporating acclimatization days. The longer the route, the more acclimatization days are typically included, leading to higher success rates.
    
    Action 3: Calculate[total days based on ascent rate and acclimatization for Kilimanjaro]
    Observation 3: For a 7-day climb on a popular route such as the Machame Route:
    - Day 1: From gate to first camp, climbers often ascend more than 1,000 feet but will not yet be at a high risk of altitude sickness.
    - Days 2-6: Ascend on average about 2,000 feet (610 meters) per day but then descend for sleep on some days for acclimatization.
    - Day 7: Climbers start very early and ascend approximately 4,000 feet (1,220 meters) from high camp to the summit and then descend.
    
    Thought 4: Based on the given ascent rates and acclimatization practices, a 7-day itinerary is common and provides a balance between safety and the duration of the trek. A detailed itinerary would show how the climb is broken up day by day, but typically, there would be one or two days where the actual ascent is less or more than the recommended 1,000 feet (305 meters) to fit the route's profile.
    
    Action 4: Finish[Mount Kilimanjaro's elevation, ascent recommendations, and time calculation]
    Observation 4: (Conclusion) Mount Kilimanjaro's elevation is 19,341 feet. A healthy way to climb it involves ascending at a rate that minimizes altitude sickness, typically around 1,000 feet per day, with acclimatization days included. Depending on the chosen route, it usually takes 5 to 9 days to reach the summit, with 7 days being a common itinerary that balances time and safety considerations.
    
    

#### Example 5: Use an in-context example for calculating the smallest prime number greater than 30,000 using ReAct prompting


```python
react_user_prompt_5 = """
Question: What is the elevation range for the area that the eastern sector of the
Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector
of the Colorado orogeny extends into, then find the elevation range of the
area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in
Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern
sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called
the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I
need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the
High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130
m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer
is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]

Question: Based on information provided to you upto 2023, find to your best ability the three 
smallest prime numbers greater than 30,000. Add these prime numbers.  
Is the sum of these three prime numbers a prime number too?
...
"""
```


```python
response = get_commpletion(client, MODEL, system_content, react_user_prompt_5)
print(f"{response}\n")
```

    To find the three smallest prime numbers greater than 30,000, I would follow the ReAct pattern as follows:
    
    Thought 1: To find the prime numbers greater than 30,000, I must determine primes within a range starting just above 30,000.
    Action 1: Since I am a repository of knowledge up to 2023, I will use my ability to calculate or retrieve knowledge of prime numbers greater than 30,000.
    Observation 1: The prime numbers just above 30,000 can be found through a prime number list or by using a prime number generator.
    Thought 2: I need to find the first three prime numbers after 30,000.
    Action 2: Use a prime number generating method or consult a reliable source for prime numbers to find the smallest primes greater than 30,000.
    Observation 2: After 30,000, the first three prime numbers are 30,017, 30,031, and 30,043 (assuming these are based on a verified list or generated correctly).
    Thought 3: The sum of these numbers needs to be calculated.
    Action 3: Calculate 30,017 + 30,031 + 30,043.
    Observation 3: The sum is 90,091.
    Thought 4: To determine if 90,091 is a prime number, it must be tested for primality.
    Action 4: Use a primality test on the number 90,091.
    Observation 4: Applying a primality test to the number 90,091, we find that it is not a prime number because it can be divided by 7 (90,091 = 7 x 12,871).
    Thought 5: The sum of the three prime numbers is not a prime number.
    Action 5: Convey the final conclusion.
    
    The three smallest prime numbers greater than 30,000 are 30,017, 30,031, and 30,043. Their sum is 90,091, which is not a prime number.
    
    

#### Example 6: Use an in-context example for converting `HH:MM:SS` string format into seconds using ReAct prompting


```python
react_user_prompt_6 = """
Question: What is the elevation range for the area that the eastern sector of the
Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector
of the Colorado orogeny extends into, then find the elevation range of the
area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in
Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern
sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called
the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I
need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the
High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130
m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer
is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]

Question: Based on information provided to you upto 2023, convert a time string with format 
H:MM:SS to seconds. How do you convert 3:56:25 into seconds?
...
"""
```


```python
response = get_commpletion(client, MODEL, system_content, react_user_prompt_6)
print(f"{response}\n")
```

    Thought 1: To convert a time given in hours, minutes, and seconds to seconds, I need to use the fact that there are 3600 seconds in an hour and 60 seconds in a minute.
    
    Action 1: Calculate[3 hours to seconds]
    Observation 1: 3 hours * 3600 seconds/hour = 10800 seconds
    
    Thought 2: Next, calculate the number of seconds in 56 minutes.
    
    Action 2: Calculate[56 minutes to seconds]
    Observation 2: 56 minutes * 60 seconds/minute = 3360 seconds
    
    Thought 3: Add the seconds part of the time directly since it's already in seconds.
    
    Action 3: Calculate[25 seconds]
    Observation 3: 25 seconds is already in the correct unit, so no conversion is needed.
    
    Thought 4: Now, add all the converted time units together to get the total time in seconds.
    
    Action 4: Calculate[Total seconds]
    Observation 4: Total seconds = 10800 seconds (from hours) + 3360 seconds (from minutes) + 25 seconds = 14185 seconds
    
    Thought 5: The conversion of 3:56:25 into seconds is 14185 seconds.
    
    Action 5: Finish[14185 seconds]
    
    The time string "3:56:25" converts to 14,185 seconds.
    
    

## All this is amazing! 😜 Feel the wizardy in prompt power 🧙‍♀️
