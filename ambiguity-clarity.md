# Handling Ambiguity and Improving Clarity in Prompt Engineering

## Overview

This tutorial focuses on two critical aspects of prompt engineering: identifying and resolving ambiguous prompts, and techniques for writing clearer prompts. These skills are essential for effective communication with AI models and obtaining more accurate and relevant responses.

## Motivation

Ambiguity in prompts can lead to inconsistent or irrelevant AI responses, while lack of clarity can result in misunderstandings and inaccurate outputs. By mastering these aspects of prompt engineering, you can significantly improve the quality and reliability of AI-generated content across various applications.

## Key Components

1. Identifying ambiguous prompts
2. Strategies for resolving ambiguity
3. Techniques for writing clearer prompts
4. Practical examples and exercises

## Method Details

We'll use OpenAI's GPT model and the LangChain library to demonstrate various techniques for handling ambiguity and improving clarity in prompts. The tutorial will cover:

1. Setting up the environment and necessary libraries
2. Analyzing ambiguous prompts and their potential interpretations
3. Implementing strategies to resolve ambiguity, such as providing context and specifying parameters
4. Exploring techniques for writing clearer prompts, including using specific language and structured formats
5. Practical exercises to apply these concepts in real-world scenarios

## Conclusion

By the end of this tutorial, you'll have a solid understanding of how to identify and resolve ambiguity in prompts, as well as techniques for crafting clearer prompts. These skills will enable you to communicate more effectively with AI models, resulting in more accurate and relevant outputs across various applications.

## Setup

First, let's import the necessary libraries and set up our environment.


```python
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini")
```

## Identifying Ambiguous Prompts

Let's start by examining some ambiguous prompts and analyzing their potential interpretations.


```python
ambiguous_prompts = [
    "Tell me about the bank.",
    "What's the best way to get to school?",
    "Can you explain the theory?"
]

for prompt in ambiguous_prompts:
    analysis_prompt = f"Analyze the following prompt for ambiguity: '{prompt}'. Explain why it's ambiguous and list possible interpretations."
    print(f"Prompt: {prompt}")
    print(llm.invoke(analysis_prompt).content)
    print("-" * 50)
```

    Prompt: Tell me about the bank.
    The prompt "Tell me about the bank." is ambiguous for several reasons:
    
    1. **Type of Bank**: The term "bank" can refer to different types of financial institutions. It could signify a commercial bank, an investment bank, a savings bank, or even a central bank (like the Federal Reserve). Each type has distinct functions, services, and regulatory environments.
    
    2. **Context of Inquiry**: The prompt does not specify the context in which the bank is to be discussed. Are we looking for historical information, current services, financial performance, or perhaps regulatory issues? Different contexts would lead to different answers.
    
    3. **Location**: The prompt does not indicate whether it refers to a specific bank (e.g., Bank of America, JPMorgan Chase) or banks in general. Without a specified location or institution, the discussion could range from a local bank to international banking practices.
    
    4. **Aspects of Interest**: The prompt does not clarify which aspects of the bank the speaker is interested in. It could pertain to its services (loans, mortgages, checking accounts), its role in the economy, its history, recent news, or even customer service experiences.
    
    5. **Audience Knowledge**: The prompt does not consider the knowledge level of the audience. A detailed explanation about banking might be appropriate for someone with little understanding of finance, while an overview of current trends might be desired by someone with more expertise.
    
    ### Possible Interpretations:
    1. **General Overview**: A request for a general description of what a bank is and its functions in the economy.
    2. **Specific Bank**: Information about a particular bank (e.g., "Tell me about Chase Bank" or "Tell me about the Bank of England").
    3. **Banking Products**: A focus on the types of products and services offered by banks, such as savings accounts, loans, and investment options.
    4. **Regulatory Issues**: An inquiry into the laws and regulations that govern banking practices.
    5. **Recent Developments**: An interest in recent news or changes in the banking sector, such as mergers, acquisitions, or technological innovations.
    6. **Historical Context**: A discussion about the history and evolution of banking as a practice.
    7. **Personal Experience**: A request for personal anecdotes or experiences related to using a bank.
    
    In conclusion, the ambiguity of the prompt arises from its vagueness in terms of context, specificity, and focus, allowing for multiple interpretations that could lead to different discussions about banking.
    --------------------------------------------------
    Prompt: What's the best way to get to school?
    The prompt "What's the best way to get to school?" is ambiguous due to several factors that can lead to different interpretations. 
    
    1. **Mode of Transportation**: The phrase "best way" could refer to various modes of transportation, such as walking, biking, driving, taking public transport, or carpooling. Each mode could be considered the "best" based on different criteria (e.g., speed, cost, environmental impact, safety).
    
    2. **Criteria for "Best"**: The term "best" is subjective and can vary based on the criteria used. For instance, one might interpret "best" as:
       - Fastest route
       - Cheapest option
       - Most environmentally friendly choice
       - Safest route (considering traffic, road conditions, etc.)
       - Most convenient (e.g., minimal transfers if using public transport)
    
    3. **Starting Point**: The prompt does not specify where the individual is starting from. The best route may vary significantly based on the starting location.
    
    4. **Destination**: While "school" is mentioned, it is unclear which school is being referred to, especially if there are multiple schools in the area or if the individual attends a specific institution with a particular address.
    
    5. **Time of Day**: The best route may depend on the time of day due to traffic patterns, public transportation schedules, or safety considerations (e.g., walking alone at night).
    
    6. **Personal Preferences**: Different individuals may have unique preferences or requirements that affect their choice of how to get to school (e.g., a preference for exercise, avoiding crowded public transport, etc.).
    
    ### Possible Interpretations:
    1. **Mode of Transport**:
       - "What’s the fastest way to get to school by car?"
       - "What’s the best route for walking to school?"
    
    2. **Criteria**:
       - "What’s the cheapest way to get to school?"
       - "What’s the safest route to take?"
    
    3. **Starting Point**:
       - "What's the best way to get to school from my house?"
       - "How do I get to school if I’m coming from downtown?"
    
    4. **Destination**:
       - "What’s the best way to get to Lincoln High School?"
       - "How do I get to the community college from my location?"
    
    5. **Time of Day**:
       - "What’s the best route to school during rush hour?"
       - "What time should I leave to avoid traffic?"
    
    6. **Personal Preferences**:
       - "What’s the best way to bike to school?"
       - "Is there a public transport option that’s less crowded?"
    
    In summary, the ambiguity in the prompt arises from the multiple interpretations of the terms used, the lack of specific context, and the variability based on individual preferences and circumstances.
    --------------------------------------------------
    Prompt: Can you explain the theory?
    The prompt "Can you explain the theory?" is ambiguous for several reasons:
    
    1. **Lack of Context**: The term "theory" is vague without additional context. There are countless theories across various fields, such as science (e.g., the theory of evolution, quantum theory), philosophy (e.g., social contract theory), psychology (e.g., attachment theory), and many others. Without specifying which theory is being referred to, the question could be interpreted in multiple ways.
    
    2. **Assumed Knowledge**: The prompt assumes that the respondent knows which theory is being referenced. Depending on the respondent's background, they may not be familiar with the specific theory in question, leading to confusion.
    
    3. **Depth of Explanation**: The term "explain" is also ambiguous. It could imply a brief summary, a detailed analysis, or a layman's explanation. Different audiences may require different levels of detail, and the respondent may not know how comprehensive their explanation should be.
    
    4. **Audience**: The prompt does not specify who the explanation is for. An explanation suitable for a novice may differ significantly from one tailored for an expert audience.
    
    Possible interpretations of the prompt include:
    
    1. **Specific Theory Request**: The respondent might interpret the question as asking about a specific theory known to both parties, such as "Can you explain the theory of relativity?"
    
    2. **General Inquiry**: The respondent might consider it a general inquiry into theories in a particular field (e.g., "Can you explain any psychological theory?").
    
    3. **Field-Specific Request**: The respondent could interpret it as a request related to a specific academic discipline (e.g., "Can you explain the theory of supply and demand in economics?").
    
    4. **Nature of Explanation**: The respondent might wonder whether to provide a simple definition, a historical overview, or a technical breakdown of the theory.
    
    5. **Philosophical vs. Scientific Theory**: The respondent may consider whether the question refers to a scientific theory that is testable and empirical or a philosophical theory that may involve more abstract reasoning.
    
    In conclusion, the prompt's ambiguity arises from its lack of specificity regarding the theory in question, the depth of explanation needed, and the intended audience. Clarifying these aspects would help eliminate confusion and facilitate a more productive discussion.
    --------------------------------------------------
    

## Resolving Ambiguity

Now, let's explore strategies for resolving ambiguity in prompts.


```python
def resolve_ambiguity(prompt, context):
    """
    Resolve ambiguity in a prompt by providing additional context.
    
    Args:
    prompt (str): The original ambiguous prompt
    context (str): Additional context to resolve ambiguity
    
    Returns:
    str: The AI's response to the clarified prompt
    """
    clarified_prompt = f"{context}\n\nBased on this context, {prompt}"
    return llm.invoke(clarified_prompt).content

# Example usage
ambiguous_prompt = "Tell me about the bank."
contexts = [
    "You are a financial advisor discussing savings accounts.",
    "You are a geographer describing river formations."
]

for context in contexts:
    print(f"Context: {context}")
    print(f"Clarified response: {resolve_ambiguity(ambiguous_prompt, context)}")
    print("-" * 50)
```

    Context: You are a financial advisor discussing savings accounts.
    Clarified response: When discussing savings accounts, it's important to consider the role of the bank in managing these accounts. Here are some key points to understand about banks in this context:
    
    1. **Types of Banks**: Banks can be broadly categorized into commercial banks, credit unions, and online banks. Each type offers savings accounts but may have different terms, interest rates, and services.
    
    2. **Interest Rates**: Banks typically offer interest on savings accounts, which can vary widely. Online banks often provide higher interest rates compared to traditional brick-and-mortar banks due to lower overhead costs. It’s essential to compare rates when choosing a bank for your savings account.
    
    3. **Fees and Minimum Balances**: Some banks charge monthly maintenance fees or require a minimum balance to avoid these fees. It’s crucial to understand the fee structure before selecting a bank, as this can affect your overall savings.
    
    4. **FDIC Insurance**: In the United States, deposits in savings accounts at member banks are insured by the Federal Deposit Insurance Corporation (FDIC) up to $250,000 per depositor, per bank. This insurance provides security and peace of mind for your savings.
    
    5. **Accessibility and Convenience**: Consider how easy it is to access your funds. Many banks offer mobile banking apps, ATMs, and online account management, making it convenient to manage your savings. 
    
    6. **Customer Service**: Good customer service can significantly enhance your banking experience. Look for banks that offer support through multiple channels, such as phone, chat, and in-person assistance.
    
    7. **Promotions and Offers**: Banks often run promotions for new savings accounts, such as cash bonuses for opening an account or higher introductory interest rates. These can be beneficial, but always read the fine print.
    
    8. **Account Features**: Some banks provide additional features like automatic savings plans, budgeting tools, or the ability to link to other accounts for easy transfers. These can help you grow your savings more effectively.
    
    When choosing a bank for your savings account, it’s important to evaluate these factors to find the best fit for your financial goals and needs.
    --------------------------------------------------
    Context: You are a geographer describing river formations.
    Clarified response: In the context of river formations, the term "bank" refers to the land alongside a river. Banks play a crucial role in shaping the river's flow and ecosystem. There are typically two banks in a river: the left bank and the right bank, determined by the perspective of looking downstream.
    
    **Characteristics of River Banks:**
    
    1. **Composition:** River banks can be made up of various materials, including soil, sand, silt, gravel, and rocks. The composition can affect erosion rates, sediment deposition, and the types of vegetation that can thrive in the area.
    
    2. **Erosion and Deposition:** The dynamic processes of erosion and deposition significantly shape river banks. Erosion occurs when water flow removes material from the bank, often resulting in steep, undercut banks. Conversely, deposition occurs when sediment carried by the river is dropped off, usually at points where the water slows down, leading to the formation of sandbars or point bars.
    
    3. **Ecology:** River banks are often rich in biodiversity. The vegetation found along banks, such as reeds, willows, and other riparian plants, provides habitat and food for various wildlife species. These plants also help stabilize the bank, reduce erosion, and improve water quality by filtering pollutants.
    
    4. **Human Impact:** Human activities, such as urban development, agriculture, and dam construction, can significantly alter river banks. These activities may lead to increased erosion, reduced habitat quality, and changes in sediment transport, which can affect the overall health of the river ecosystem.
    
    5. **Floodplain Interaction:** River banks are often part of a larger floodplain, which is the area adjacent to the river that may be inundated during periods of high flow. The interaction between the river and its banks during flooding can lead to the deposition of nutrient-rich sediments, benefiting the surrounding ecosystem.
    
    Understanding the formation and dynamics of river banks is essential for managing and preserving riverine environments, as they are integral to the health of aquatic and terrestrial ecosystems.
    --------------------------------------------------
    

## Techniques for Writing Clearer Prompts

Let's explore some techniques for writing clearer prompts to improve AI responses.


```python
def compare_prompt_clarity(original_prompt, improved_prompt):
    """
    Compare the responses to an original prompt and an improved, clearer version.
    
    Args:
    original_prompt (str): The original, potentially unclear prompt
    improved_prompt (str): An improved, clearer version of the prompt
    
    Returns:
    tuple: Responses to the original and improved prompts
    """
    original_response = llm.invoke(original_prompt).content
    improved_response = llm.invoke(improved_prompt).content
    return original_response, improved_response

# Example usage
original_prompt = "How do I make it?"
improved_prompt = "Provide a step-by-step guide for making a classic margherita pizza, including ingredients and cooking instructions."

original_response, improved_response = compare_prompt_clarity(original_prompt, improved_prompt)

print("Original Prompt Response:")
print(original_response)
print("\nImproved Prompt Response:")
print(improved_response)
```

    Original Prompt Response:
    Could you please clarify what you would like to make? Whether it's a recipe, a DIY project, or something else, I'd be happy to help!
    
    Improved Prompt Response:
    Sure! Here’s a step-by-step guide for making a classic Margherita pizza, which features a simple yet delicious combination of fresh ingredients.
    
    ### Ingredients:
    
    #### For the Dough:
    - 2 ¼ cups (280g) all-purpose flour (plus extra for dusting)
    - 1 teaspoon salt
    - ¾ teaspoon instant yeast
    - ¾ cup (180ml) warm water (about 100°F/38°C)
    - 1 teaspoon sugar (optional, to help activate yeast)
    
    #### For the Toppings:
    - 1 cup (240ml) canned San Marzano tomatoes (or any good quality canned tomatoes)
    - 1 tablespoon olive oil (plus more for drizzling)
    - Salt to taste
    - 8 ounces (225g) fresh mozzarella cheese, preferably buffalo mozzarella
    - Fresh basil leaves
    - Freshly cracked black pepper (optional)
    
    ### Equipment:
    - A mixing bowl
    - A baking sheet or pizza stone
    - A rolling pin (optional)
    - A pizza peel (optional, for transferring to the oven)
    - An oven (preferably with a pizza stone or steel for best results)
    
    ### Instructions:
    
    #### Step 1: Make the Dough
    1. **Mix the dry ingredients**: In a mixing bowl, combine the flour, salt, and instant yeast. If you're using sugar, add it here as well.
    2. **Add water**: Slowly pour in the warm water while stirring the mixture with a spoon or your hand until it begins to come together into a shaggy dough.
    3. **Knead the dough**: Transfer the dough onto a lightly floured surface and knead for about 8-10 minutes until smooth and elastic. If the dough is too sticky, sprinkle a little more flour as needed.
    4. **Let it rise**: Form the dough into a ball and place it in a lightly greased bowl. Cover it with a damp cloth or plastic wrap and let it rise in a warm place for about 1-2 hours, or until it has doubled in size.
    
    #### Step 2: Prepare the Sauce
    1. **Blend the tomatoes**: In a bowl, crush the canned tomatoes by hand or use a blender for a smoother consistency. You want it to be a bit chunky for texture.
    2. **Season**: Add a little salt to taste and a tablespoon of olive oil to the tomato mixture. Mix well and set aside.
    
    #### Step 3: Preheat the Oven
    1. **Preheat your oven**: If using a pizza stone, place it in the oven and preheat to the highest setting (usually around 475°F to 500°F or 245°C to 260°C) for at least 30 minutes. If you don’t have a pizza stone, preheat a baking sheet.
    
    #### Step 4: Shape the Pizza
    1. **Divide the dough**: Once the dough has risen, punch it down and divide it into two equal pieces (for two pizzas). Shape each piece into a ball and let them rest for 10-15 minutes.
    2. **Shape the pizza**: On a lightly floured surface, take one dough ball and gently stretch it out with your hands or roll it out with a rolling pin into a 10-12 inch round. Make sure the edges are slightly thicker for the crust.
    
    #### Step 5: Assemble the Pizza
    1. **Add the sauce**: Spread a thin layer of the tomato sauce over the surface of the dough, leaving a small border around the edges.
    2. **Add cheese**: Tear the fresh mozzarella into small pieces and distribute them evenly over the sauce.
    3. **Add basil**: Tear a few fresh basil leaves and sprinkle them on top (you can also add them after baking for a fresher taste).
    4. **Drizzle olive oil**: Drizzle a little olive oil over the top for added flavor.
    
    #### Step 6: Bake the Pizza
    1. **Transfer to the oven**: If using a pizza peel, sprinkle it with flour or cornmeal and carefully transfer the assembled pizza onto it. Then slide the pizza onto the preheated stone or baking sheet in the oven.
    2. **Bake**: Bake for about 8-12 minutes, or until the crust is golden and the cheese is bubbling and starting to brown.
    3. **Check frequently**: Keep an eye on the pizza to avoid burning, especially if your oven runs hot.
    
    #### Step 7: Serve
    1. **Remove from oven**: Once done, carefully remove the pizza from the oven.
    2. **Garnish**: Add a few more fresh basil leaves, a drizzle of olive oil, and freshly cracked black pepper if desired.
    3. **Slice and enjoy**: Let it cool for a minute, slice it up, and enjoy your classic Margherita pizza!
    
    ### Tips:
    - For the best flavor, use high-quality ingredients, especially the tomatoes and mozzarella.
    - If you have time, letting the dough rise slowly in the refrigerator overnight can enhance the flavor and texture.
    - Experiment with the thickness of the crust to find your preferred style.
    
    Enjoy your homemade Margherita pizza!
    

## Structured Prompts for Clarity

Using structured prompts can significantly improve clarity and consistency in AI responses.


```python
structured_prompt = PromptTemplate(
    input_variables=["topic", "aspects", "tone"],
    template="""Provide an analysis of {topic} considering the following aspects:
    1. {{aspects[0]}}
    2. {{aspects[1]}}
    3. {{aspects[2]}}
    
    Present the analysis in a {tone} tone.
    """
)

# Example usage
input_variables = {
    "topic": "the impact of social media on society",
    "aspects": ["communication patterns", "mental health", "information spread"],
    "tone": "balanced and objective"
}

chain = structured_prompt | llm
response = chain.invoke(input_variables).content
print(response)
```

    To analyze the impact of social media on society, we can consider the following aspects: communication, mental health, and information dissemination. Each of these areas reveals both positive and negative consequences of social media usage.
    
    ### 1. Communication
    
    **Positive Impact:**  
    Social media has revolutionized communication by making it easier and faster for people to connect across long distances. Platforms like Facebook, Twitter, and Instagram allow users to share moments, thoughts, and experiences with friends and family, regardless of geographic barriers. This instant connectivity can foster relationships and create a sense of belonging, especially for those who may feel isolated in their physical environments.
    
    **Negative Impact:**  
    Conversely, the nature of communication on social media can lead to misunderstandings and conflicts. The absence of non-verbal cues, such as tone and body language, can result in misinterpretations of messages. Furthermore, the prevalence of online arguments and cyberbullying can create a toxic environment, leading to strained relationships and a decline in face-to-face interactions.
    
    ### 2. Mental Health
    
    **Positive Impact:**  
    Social media can serve as a supportive platform for individuals dealing with mental health issues. Online communities provide a space for individuals to share experiences and seek support from others facing similar challenges. Many organizations use social media to raise awareness about mental health, promoting resources and encouraging open discussions.
    
    **Negative Impact:**  
    On the flip side, social media can contribute to mental health issues such as anxiety, depression, and low self-esteem. The constant comparison with others' curated lives can lead to feelings of inadequacy. Additionally, the addictive nature of social media can exacerbate feelings of loneliness and isolation, as users may substitute online interactions for genuine social connections.
    
    ### 3. Information Dissemination
    
    **Positive Impact:**  
    Social media has democratized the flow of information, allowing users to access a wide range of news and perspectives that may not be covered by traditional media outlets. This accessibility can empower individuals to engage in social and political discourse, mobilize for causes, and stay informed about global events in real-time.
    
    **Negative Impact:**  
    However, the rapid spread of information can also lead to the dissemination of misinformation and disinformation. False narratives can easily go viral, leading to public confusion and mistrust in credible sources. The algorithms that govern many social media platforms often prioritize sensational content, which can skew public perception and create echo chambers that reinforce existing biases.
    
    ### Conclusion
    
    In summary, the impact of social media on society is multifaceted, encompassing both beneficial and detrimental effects. While it fosters communication, offers mental health support, and enhances information accessibility, it also presents challenges such as misunderstandings, mental health concerns, and the spread of misinformation. A balanced perspective requires recognizing these complexities and striving for responsible usage of social media to maximize its positive potential while mitigating its adverse effects.
    

## Practical Exercise: Improving Prompt Clarity

Now, let's practice improving the clarity of prompts.


```python
unclear_prompts = [
    "What's the difference?",
    "How does it work?",
    "Why is it important?"
]

def improve_prompt_clarity(unclear_prompt):
    """
    Improve the clarity of a given prompt.
    
    Args:
    unclear_prompt (str): The original unclear prompt
    
    Returns:
    str: An improved, clearer version of the prompt
    """
    improvement_prompt = f"The following prompt is unclear: '{unclear_prompt}'. Please provide a clearer, more specific version of this prompt. output just the improved prompt and nothing else." 
    return llm.invoke(improvement_prompt).content

for prompt in unclear_prompts:
    improved_prompt = improve_prompt_clarity(prompt)
    print(f"Original: {prompt}")
    print(f"Improved: {improved_prompt}")
    print("-" * 50)
```

    Original: What's the difference?
    Improved: "What are the differences between these two concepts/objects?"
    --------------------------------------------------
    Original: How does it work?
    Improved: Can you explain the process or mechanism behind how this system or product functions?
    --------------------------------------------------
    Original: Why is it important?
    Improved: "What is the significance of this topic, and how does it impact individuals or society?"
    --------------------------------------------------
    
