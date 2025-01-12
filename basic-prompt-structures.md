# Basic Prompt Structures Tutorial

## Overview

This tutorial focuses on two fundamental types of prompt structures:
1. Single-turn prompts
2. Multi-turn prompts (conversations)

We'll use OpenAI's GPT model and LangChain to demonstrate these concepts.

## Motivation

Understanding different prompt structures is crucial for effective communication with AI models. Single-turn prompts are useful for quick, straightforward queries, while multi-turn prompts enable more complex, context-aware interactions. Mastering these structures allows for more versatile and effective use of AI in various applications.

## Key Components

1. **Single-turn Prompts**: One-shot interactions with the language model.
2. **Multi-turn Prompts**: Series of interactions that maintain context.
3. **Prompt Templates**: Reusable structures for consistent prompting.
4. **Conversation Chains**: Maintaining context across multiple interactions.

## Method Details

We'll use a combination of OpenAI's API and LangChain library to demonstrate these prompt structures. The tutorial will include practical examples and comparisons of different prompt types.

## Setup

First, let's import the necessary libraries and set up our environment.


```python
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY') # OpenAI API key
# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini")
```

## 1. Single-turn Prompts

Single-turn prompts are one-shot interactions with the language model. They consist of a single input (prompt) and generate a single output (response).


```python
single_turn_prompt = "What are the three primary colors?"
print(llm.invoke(single_turn_prompt).content)
```

    The three primary colors are red, blue, and yellow. These colors cannot be created by mixing other colors together and are the foundation for creating a wide range of other colors through mixing. In the context of additive color mixing (like with light), the primary colors are red, green, and blue (RGB).
    

Now, let's use a PromptTemplate to create a more structured single-turn prompt:


```python
structured_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Provide a brief explanation of {topic} and list its three main components."
)

chain = structured_prompt | llm
print(chain.invoke({"topic": "color theory"}).content)
```

    Color theory is a framework used to understand how colors interact, complement each other, and can be combined to create various visual effects. It is essential in fields such as art, design, and photography, helping artists and designers make informed choices about color usage to evoke emotions, communicate messages, and create harmony in their work.
    
    The three main components of color theory are:
    
    1. **Color Wheel**: A circular diagram that shows the relationships between colors. It typically includes primary, secondary, and tertiary colors, providing a visual representation of how colors can be combined.
    
    2. **Color Harmony**: The concept of combining colors in a pleasing way. It involves using color schemes such as complementary, analogous, and triadic to create balance and visual interest.
    
    3. **Color Context**: This refers to how colors interact with one another and how they can change perception based on their surrounding colors. The same color can appear different depending on the colors next to it, which influences mood and interpretation.
    

## 2. Multi-turn Prompts (Conversations)

Multi-turn prompts involve a series of interactions with the language model, allowing for more complex and context-aware conversations.


```python
conversation = ConversationChain(
    llm=llm, 
    verbose=True,
    memory=ConversationBufferMemory()
)

print(conversation.predict(input="Hi, I'm learning about space. Can you tell me about planets?"))
print(conversation.predict(input="What's the largest planet in our solar system?"))
print(conversation.predict(input="How does its size compare to Earth?"))
```

    C:\Users\N7\AppData\Local\Temp\ipykernel_20652\4194631287.py:4: LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
      memory=ConversationBufferMemory()
    C:\Users\N7\AppData\Local\Temp\ipykernel_20652\4194631287.py:1: LangChainDeprecationWarning: The class `ConversationChain` was deprecated in LangChain 0.2.7 and will be removed in 1.0. Use :meth:`~RunnableWithMessageHistory: https://python.langchain.com/v0.2/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html` instead.
      conversation = ConversationChain(
    

    
    
    [1m> Entering new ConversationChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    
    Human: Hi, I'm learning about space. Can you tell me about planets?
    AI:[0m
    
    [1m> Finished chain.[0m
    Absolutely! Planets are fascinating celestial bodies that orbit stars, and in our solar system, they revolve around the Sun. There are eight recognized planets in our solar system, and they can be categorized into two main groups: terrestrial planets and gas giants.
    
    The terrestrial planets—Mercury, Venus, Earth, and Mars—are rocky and have solid surfaces. 
    
    - **Mercury** is the closest planet to the Sun and has extreme temperature variations, ranging from scorching hot during the day to frigid cold at night.
    - **Venus** is often called Earth's "sister planet" due to its similar size but has a thick, toxic atmosphere that traps heat, making it the hottest planet in the solar system.
    - **Earth**, our home, is unique for its liquid water and life-sustaining atmosphere.
    - **Mars**, known as the Red Planet because of its iron oxide-rich soil, has the largest volcano and canyon in the solar system.
    
    The gas giants—Jupiter and Saturn, and the ice giants—Uranus and Neptune, are much larger and do not have solid surfaces like the terrestrial planets.
    
    - **Jupiter** is the largest planet, famous for its Great Red Spot, a massive storm larger than Earth, and its many moons, including the largest moon in the solar system, Ganymede.
    - **Saturn** is known for its stunning ring system, made up of ice and rock particles.
    - **Uranus** is unique because it rotates on its side, and it's known for its blue color due to methane in its atmosphere.
    - **Neptune**, the furthest planet from the Sun, has strong winds and is also blue due to methane; it has the fastest winds recorded in the solar system.
    
    If you're interested in something more specific about any planet or the characteristics of other celestial bodies, feel free to ask!
    
    
    [1m> Entering new ConversationChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    Human: Hi, I'm learning about space. Can you tell me about planets?
    AI: Absolutely! Planets are fascinating celestial bodies that orbit stars, and in our solar system, they revolve around the Sun. There are eight recognized planets in our solar system, and they can be categorized into two main groups: terrestrial planets and gas giants.
    
    The terrestrial planets—Mercury, Venus, Earth, and Mars—are rocky and have solid surfaces. 
    
    - **Mercury** is the closest planet to the Sun and has extreme temperature variations, ranging from scorching hot during the day to frigid cold at night.
    - **Venus** is often called Earth's "sister planet" due to its similar size but has a thick, toxic atmosphere that traps heat, making it the hottest planet in the solar system.
    - **Earth**, our home, is unique for its liquid water and life-sustaining atmosphere.
    - **Mars**, known as the Red Planet because of its iron oxide-rich soil, has the largest volcano and canyon in the solar system.
    
    The gas giants—Jupiter and Saturn, and the ice giants—Uranus and Neptune, are much larger and do not have solid surfaces like the terrestrial planets.
    
    - **Jupiter** is the largest planet, famous for its Great Red Spot, a massive storm larger than Earth, and its many moons, including the largest moon in the solar system, Ganymede.
    - **Saturn** is known for its stunning ring system, made up of ice and rock particles.
    - **Uranus** is unique because it rotates on its side, and it's known for its blue color due to methane in its atmosphere.
    - **Neptune**, the furthest planet from the Sun, has strong winds and is also blue due to methane; it has the fastest winds recorded in the solar system.
    
    If you're interested in something more specific about any planet or the characteristics of other celestial bodies, feel free to ask!
    Human: What's the largest planet in our solar system?
    AI:[0m
    
    [1m> Finished chain.[0m
    The largest planet in our solar system is **Jupiter**. It has a diameter of about 86,881 miles (139,822 kilometers) and is more than 11 times wider than Earth! Jupiter is primarily composed of hydrogen and helium, and it has a very strong magnetic field and numerous moons—over 79 have been confirmed, with the four largest known as the Galilean moons: Io, Europa, Ganymede, and Callisto. Ganymede is particularly notable as it is not only the largest moon of Jupiter but also the largest moon in the entire solar system, even larger than the planet Mercury! If you want to know more about Jupiter or any of its moons, just let me know!
    
    
    [1m> Entering new ConversationChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    Human: Hi, I'm learning about space. Can you tell me about planets?
    AI: Absolutely! Planets are fascinating celestial bodies that orbit stars, and in our solar system, they revolve around the Sun. There are eight recognized planets in our solar system, and they can be categorized into two main groups: terrestrial planets and gas giants.
    
    The terrestrial planets—Mercury, Venus, Earth, and Mars—are rocky and have solid surfaces. 
    
    - **Mercury** is the closest planet to the Sun and has extreme temperature variations, ranging from scorching hot during the day to frigid cold at night.
    - **Venus** is often called Earth's "sister planet" due to its similar size but has a thick, toxic atmosphere that traps heat, making it the hottest planet in the solar system.
    - **Earth**, our home, is unique for its liquid water and life-sustaining atmosphere.
    - **Mars**, known as the Red Planet because of its iron oxide-rich soil, has the largest volcano and canyon in the solar system.
    
    The gas giants—Jupiter and Saturn, and the ice giants—Uranus and Neptune, are much larger and do not have solid surfaces like the terrestrial planets.
    
    - **Jupiter** is the largest planet, famous for its Great Red Spot, a massive storm larger than Earth, and its many moons, including the largest moon in the solar system, Ganymede.
    - **Saturn** is known for its stunning ring system, made up of ice and rock particles.
    - **Uranus** is unique because it rotates on its side, and it's known for its blue color due to methane in its atmosphere.
    - **Neptune**, the furthest planet from the Sun, has strong winds and is also blue due to methane; it has the fastest winds recorded in the solar system.
    
    If you're interested in something more specific about any planet or the characteristics of other celestial bodies, feel free to ask!
    Human: What's the largest planet in our solar system?
    AI: The largest planet in our solar system is **Jupiter**. It has a diameter of about 86,881 miles (139,822 kilometers) and is more than 11 times wider than Earth! Jupiter is primarily composed of hydrogen and helium, and it has a very strong magnetic field and numerous moons—over 79 have been confirmed, with the four largest known as the Galilean moons: Io, Europa, Ganymede, and Callisto. Ganymede is particularly notable as it is not only the largest moon of Jupiter but also the largest moon in the entire solar system, even larger than the planet Mercury! If you want to know more about Jupiter or any of its moons, just let me know!
    Human: How does its size compare to Earth?
    AI:[0m
    
    [1m> Finished chain.[0m
    Jupiter is significantly larger than Earth! To give you a clearer picture, Jupiter's diameter is about 86,881 miles (139,822 kilometers), while Earth's diameter is around 7,917.5 miles (12,742 kilometers). This means that Jupiter is more than 11 times wider than Earth!
    
    In terms of volume, you could fit about 1,300 Earths inside Jupiter! Additionally, Jupiter's mass is approximately 318 times greater than that of Earth. Despite its massive size and weight, Jupiter has a much lower density compared to Earth, which is why it is classified as a gas giant. If you have more questions about Jupiter or want to know how gravity differs between the two planets, feel free to ask!
    

Let's compare how single-turn and multi-turn prompts handle a series of related questions:


```python
# Single-turn prompts
prompts = [
    "What is the capital of France?",
    "What is its population?",
    "What is the city's most famous landmark?"
]

print("Single-turn responses:")
for prompt in prompts:
    print(f"Q: {prompt}")
    print(f"A: {llm.invoke(prompt).content}\n")

# Multi-turn prompts
print("Multi-turn responses:")
conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())
for prompt in prompts:
    print(f"Q: {prompt}")
    print(f"A: {conversation.predict(input=prompt)}\n")
```

    Single-turn responses:
    Q: What is the capital of France?
    A: The capital of France is Paris.
    
    Q: What is its population?
    A: Could you please specify which location or entity you are referring to in order to provide the population information?
    
    Q: What is the city's most famous landmark?
    A: To provide an accurate answer, I need to know which city you are referring to. Different cities have different famous landmarks. Could you please specify the city?
    
    Multi-turn responses:
    Q: What is the capital of France?
    A: The capital of France is Paris! It's known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris is also famous for its rich history, art, and culture. Have you ever been to Paris or is it on your travel list?
    
    Q: What is its population?
    A: As of my last update, the population of Paris is approximately 2.1 million people within the city limits. However, if you consider the larger metropolitan area, that number rises to around 12 million. Paris is a vibrant city with a diverse population and a mix of cultures. Have you ever thought about what it would be like to live in such a bustling city?
    
    Q: What is the city's most famous landmark?
    A: The most famous landmark in Paris is undoubtedly the Eiffel Tower! It was completed in 1889 for the Exposition Universelle (World's Fair) and stands at a height of about 300 meters (984 feet). The Eiffel Tower attracts millions of visitors each year, offering stunning views of the city from its observation decks. It's also beautifully illuminated at night, making it a romantic spot for both locals and tourists. Have you ever seen the Eiffel Tower in pictures or dreamed of visiting it?
    
    

## Conclusion

This tutorial has introduced you to the basics of single-turn and multi-turn prompt structures. We've seen how:

1. Single-turn prompts are useful for quick, isolated queries.
2. Multi-turn prompts maintain context across a conversation, allowing for more complex interactions.
3. PromptTemplates can be used to create structured, reusable prompts.
4. Conversation chains in LangChain help manage context in multi-turn interactions.

Understanding these different prompt structures allows you to choose the most appropriate approach for various tasks and create more effective interactions with AI language models.
