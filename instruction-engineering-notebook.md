# Instruction Engineering Tutorial

## Overview

This tutorial focuses on Instruction Engineering, a crucial aspect of prompt engineering that deals with crafting clear and effective instructions for language models. We'll explore techniques for creating well-structured prompts and balancing specificity with generality to achieve optimal results.

## Motivation

As language models become more advanced, the quality of instructions we provide becomes increasingly important. Well-crafted instructions can significantly improve the model's output, leading to more accurate, relevant, and useful responses. This tutorial aims to equip learners with the skills to create effective instructions that maximize the potential of AI language models.

## Key Components

1. Crafting Clear Instructions: Techniques for writing unambiguous and easily understandable prompts.
2. Effective Instruction Structures: Exploring different ways to format and organize instructions.
3. Balancing Specificity and Generality: Finding the right level of detail in instructions.
4. Iterative Refinement: Techniques for improving instructions based on model outputs.

## Method Details

We'll use the OpenAI API and LangChain library to demonstrate instruction engineering techniques. The tutorial will cover:

1. Setting up the environment and necessary libraries.
2. Creating basic instructions and analyzing their effectiveness.
3. Refining instructions for clarity and specificity.
4. Experimenting with different instruction structures.
5. Balancing specific and general instructions for versatile outputs.
6. Iterative improvement of instructions based on model responses.

Throughout the tutorial, we'll use practical examples to illustrate these concepts and provide hands-on experience in crafting effective instructions.

## Conclusion

By the end of this tutorial, learners will have gained practical skills in instruction engineering, including how to craft clear and effective instructions, balance specificity and generality, and iteratively refine prompts for optimal results. These skills are essential for anyone working with AI language models and can significantly enhance the quality and usefulness of AI-generated content across various applications.

## Setup

First, let's import the necessary libraries and set up our environment.


```python
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini")

def get_completion(prompt):
    """Helper function to get model completion."""
    return llm.invoke(prompt).content
```

## Crafting Clear Instructions

Let's start by examining the importance of clarity in instructions. We'll compare vague and clear instructions to see the difference in model outputs.


```python
vague_instruction = "Tell me about climate change conciesly."
clear_instruction = "Provide a concise summary of the primary causes and effects of climate change, focusing on scientific consensus from the past five years conciesly."

print("Vague Instruction Output:")
print(get_completion(vague_instruction))

print("\nClear Instruction Output:")
print(get_completion(clear_instruction))
```

    Vague Instruction Output:
    Climate change refers to significant and lasting changes in global temperatures and weather patterns over time. While climate change is a natural phenomenon, human activities, particularly the burning of fossil fuels, deforestation, and industrial processes, have accelerated this process since the late 19th century. This has led to increased greenhouse gas emissions, primarily carbon dioxide and methane, trapping heat in the atmosphere.
    
    Consequences of climate change include rising global temperatures, melting ice caps, sea-level rise, more frequent and severe weather events (like hurricanes and droughts), and disruptions to ecosystems and biodiversity. Addressing climate change requires global cooperation to reduce greenhouse gas emissions, transition to renewable energy sources, and promote sustainable practices.
    
    Clear Instruction Output:
    **Primary Causes of Climate Change:**
    
    1. **Greenhouse Gas Emissions:** The combustion of fossil fuels (coal, oil, and natural gas) for energy and transportation is the largest contributor, releasing carbon dioxide (CO2) and methane (CH4).
    2. **Deforestation:** Trees absorb CO2, and large-scale deforestation reduces this capacity, while also releasing stored carbon.
    3. **Agricultural Practices:** Livestock production and certain agricultural methods contribute significant greenhouse gases, particularly methane and nitrous oxide.
    4. **Industrial Processes:** Manufacturing and chemical processes release various greenhouse gases and pollutants.
    
    **Effects of Climate Change:**
    
    1. **Temperature Rise:** Global temperatures have increased, leading to more frequent and severe heatwaves.
    2. **Extreme Weather Events:** Increased intensity and frequency of hurricanes, floods, droughts, and wildfires are observed.
    3. **Sea Level Rise:** Melting ice caps and glaciers, along with thermal expansion of water, contribute to rising sea levels, threatening coastal communities.
    4. **Ecosystem Disruption:** Altered habitats lead to shifts in biodiversity, threatening species extinction and disrupting food webs.
    5. **Public Health Risks:** Increased heat and pollution levels can exacerbate health issues, while changing climates can also affect the spread of diseases.
    
    The scientific consensus emphasizes that urgent action is needed to mitigate these causes and adapt to the impacts of climate change to ensure a sustainable future.
    

## Effective Instruction Structures

Now, let's explore different structures for instructions to see how they affect the model's output.


```python
bullet_structure = """
Explain the process of photosynthesis conciesly:
- Define photosynthesis
- List the main components involved
- Describe the steps in order
- Mention its importance for life on Earth
"""

narrative_structure = """
Imagine you're a botanist explaining photosynthesis to a curious student. 
Start with a simple definition, then walk through the process step-by-step, 
highlighting the key components involved. Conclude by emphasizing why 
photosynthesis is crucial for life on Earth. Write it concisely.
"""

print("Bullet Structure Output:")
print(get_completion(bullet_structure))

print("\nNarrative Structure Output:")
print(get_completion(narrative_structure))
```

    Bullet Structure Output:
    ### Photosynthesis
    
    **Definition:**  
    Photosynthesis is the biochemical process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose, using carbon dioxide and water.
    
    **Main Components Involved:**  
    1. **Light Energy** (usually sunlight)
    2. **Chlorophyll** (pigment in chloroplasts)
    3. **Water (H₂O)**
    4. **Carbon Dioxide (CO₂)**
    5. **Glucose (C₆H₁₂O₆)**
    6. **Oxygen (O₂)**
    
    **Steps of Photosynthesis:**
    1. **Light Absorption:** Chlorophyll absorbs sunlight, primarily in the blue and red wavelengths.
    2. **Water Splitting (Photolysis):** The absorbed light energy splits water molecules into oxygen, protons, and electrons.
    3. **Oxygen Release:** Oxygen is released as a byproduct into the atmosphere.
    4. **Energy Conversion:** The electrons move through the electron transport chain, creating ATP (adenosine triphosphate) and NADPH (nicotinamide adenine dinucleotide phosphate) from ADP and NADP⁺.
    5. **Calvin Cycle:** In the stroma, ATP and NADPH are used to convert carbon dioxide into glucose through a series of reactions.
    
    **Importance for Life on Earth:**  
    Photosynthesis is crucial for life on Earth as it produces oxygen, which is essential for the respiration of most living organisms. Additionally, it forms the base of the food chain, providing energy and organic compounds for plants, animals, and humans.
    
    Narrative Structure Output:
    Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy in the form of glucose, using carbon dioxide and water.
    
    Here's how it works, step-by-step:
    
    1. **Light Absorption**: Plants have a green pigment called chlorophyll, primarily found in chloroplasts, that captures sunlight. This light energy is essential for driving the photosynthesis process.
    
    2. **Water Uptake**: Roots absorb water (H₂O) from the soil and transport it to the leaves through specialized vessels known as xylem.
    
    3. **Carbon Dioxide Intake**: Plants take in carbon dioxide (CO₂) from the atmosphere through small openings in their leaves called stomata.
    
    4. **Light Reaction**: In the chloroplasts, the absorbed light energy splits water molecules into oxygen (O₂), protons, and electrons. This reaction releases oxygen as a byproduct, which is expelled into the atmosphere.
    
    5. **Calvin Cycle**: The electrons and energy produced in the light reaction are used in the Calvin Cycle to convert carbon dioxide and protons into glucose (C₆H₁₂O₆), a simple sugar that serves as an energy source for the plant.
    
    In summary, photosynthesis is crucial for life on Earth because it produces oxygen, which is vital for the survival of most living organisms, and it forms the base of the food chain by converting solar energy into a form that can be used by other organisms for energy. Without photosynthesis, life as we know it would not exist.
    

## Balancing Specificity and Generality

Let's experiment with instructions that vary in their level of specificity to understand how this affects the model's responses.


```python
specific_instruction = """
Describe the plot of the 1985 film 'Back to the Future', focusing on:
1. The main character's name and his friendship with Dr. Brown
2. The time machine and how it works
3. The specific year the main character travels to and why it's significant
4. The main conflict involving his parents' past
5. How the protagonist resolves the issues and returns to his time
Limit your response to 150 words. 
"""

general_instruction = """
Describe the plot of a popular time travel movie from the 1980s. Include:
1. The main characters and their relationships
2. The method of time travel
3. The time period visited and its significance
4. The main conflict or challenge faced
5. How the story is resolved
Keep your response around 150 words.
"""

print("Specific Instruction Output:")
print(get_completion(specific_instruction))

print("\nGeneral Instruction Output:")
print(get_completion(general_instruction))
```

    Specific Instruction Output:
    In the 1985 film "Back to the Future," the main character, Marty McFly, is a teenager who shares a close friendship with eccentric scientist Dr. Emmett Brown. Dr. Brown invents a time machine using a DeLorean car, which operates when it reaches 88 miles per hour, powered by a flux capacitor and plutonium. Marty accidentally travels back to 1955, a significant year as it’s when his parents first met. The main conflict arises when Marty disrupts their initial encounter, threatening his own existence. To resolve this, Marty must ensure his parents fall in love while avoiding interactions with his younger self. With Dr. Brown's guidance, he orchestrates a series of events at the Enchantment Under the Sea dance, ultimately restoring his parents' romance. After succeeding, Marty returns to 1985, where he finds his life improved by the changes he made in the past.
    
    General Instruction Output:
    One of the most popular time travel movies from the 1980s is "Back to the Future." The main character, Marty McFly, is a teenager who is friends with eccentric scientist Doc Brown. Their relationship is built on mutual respect and friendship. The method of time travel is a DeLorean car modified by Doc to travel through time when it reaches 88 miles per hour. Marty travels back to 1955, a time significant for its cultural impact and the formative years of his parents.
    
    The main conflict arises when Marty accidentally interferes with his parents' first meeting, jeopardizing his own existence. He must navigate the challenges of the past, ensuring his parents fall in love. The story resolves when Marty successfully orchestrates their meeting at the school dance, restoring the timeline. He returns to 1985, finding his life improved, and Doc arrives from the future, setting the stage for further adventures.
    

## Iterative Refinement

Now, let's demonstrate how to iteratively refine instructions based on the model's output.


```python
initial_instruction = "Explain how to make a peanut butter and jelly sandwich."

print("Initial Instruction Output:")
initial_output = get_completion(initial_instruction)
print(initial_output)

refined_instruction = """
Explain how to make a peanut butter and jelly sandwich, with the following improvements:
1. Specify the type of bread, peanut butter, and jelly to use
2. Include a step about washing hands before starting
3. Mention how to deal with potential allergies
4. Add a tip for storing the sandwich if not eaten immediately
Present the instructions in a numbered list format.
"""

print("\nRefined Instruction Output:")
refined_output = get_completion(refined_instruction)
print(refined_output)
```

    Initial Instruction Output:
    Making a peanut butter and jelly sandwich is quick and easy! Here’s a step-by-step guide:
    
    ### Ingredients:
    - 2 slices of bread (white, whole wheat, or your choice)
    - Peanut butter (creamy or crunchy)
    - Jelly or jam (flavor of your choice, such as grape, strawberry, or raspberry)
    
    ### Tools:
    - Butter knife or spreading knife
    - Spoon (optional, for jelly)
    - Plate
    
    ### Instructions:
    
    1. **Gather Your Ingredients and Tools**: Make sure you have everything you need within reach.
    
    2. **Spread Peanut Butter**:
       - Place one slice of bread on the plate.
       - Use the butter knife to scoop out a generous amount of peanut butter.
       - Spread the peanut butter evenly over one side of the bread, covering it from edge to edge.
    
    3. **Spread Jelly**:
       - Take the second slice of bread and place it next to the first slice.
       - If using a spoon, scoop out some jelly and place it on the second slice. If using the knife, you can use the clean side or a different knife.
       - Spread the jelly evenly over one side of the second slice of bread, ensuring it covers the bread well.
    
    4. **Combine the Slices**:
       - Carefully place the slice with jelly on top of the slice with peanut butter, jelly side down, to create a sandwich.
    
    5. **Cut the Sandwich (Optional)**:
       - If desired, you can cut the sandwich in half or into quarters for easier eating. Use the butter knife to slice through the sandwich.
    
    6. **Serve and Enjoy**: Your peanut butter and jelly sandwich is ready to be enjoyed! You can serve it with a glass of milk, some fruit, or your favorite snack.
    
    ### Tips:
    - For added flavor, consider adding banana slices or honey.
    - If you’re making it for kids, you might want to use a cookie cutter to make fun shapes.
    
    Enjoy your delicious peanut butter and jelly sandwich!
    
    Refined Instruction Output:
    Here’s a step-by-step guide on how to make a delicious peanut butter and jelly sandwich, incorporating your requested improvements:
    
    ### How to Make a Peanut Butter and Jelly Sandwich
    
    1. **Wash Your Hands**: Before you start, wash your hands thoroughly with soap and water for at least 20 seconds to ensure cleanliness.
    
    2. **Gather Your Ingredients**:
       - **Bread**: Choose whole grain or white bread, depending on your preference.
       - **Peanut Butter**: Use creamy or crunchy natural peanut butter for a wholesome taste.
       - **Jelly**: Opt for grape or strawberry jelly for a classic flavor.
    
    3. **Prepare Your Workspace**: Clear a clean surface on your kitchen counter and gather the following tools:
       - A butter knife or spreading tool
       - A clean plate
       - A spoon (if needed for the jelly)
    
    4. **Spread the Peanut Butter**: Take one slice of bread and use the butter knife to spread an even layer of peanut butter over one side. Be generous, but don’t overdo it—about 2 tablespoons is a good amount.
    
    5. **Spread the Jelly**: On the second slice of bread, use the clean side of your butter knife or a spoon to spread jelly evenly over the surface. Again, about 2 tablespoons should suffice.
    
    6. **Combine the Slices**: Carefully place the peanut butter slice on top of the jelly slice, peanut butter side facing the jelly side, to create your sandwich.
    
    7. **Cut the Sandwich (Optional)**: If you prefer, you can cut the sandwich in half diagonally or vertically for easier handling.
    
    8. **Address Allergies**: Be mindful of potential allergies. If you or someone you are serving has a peanut allergy, consider using an alternative like sunflower seed butter or almond butter, and ensure that the jelly is free from any allergens.
    
    9. **Storage Tip**: If you’re not eating the sandwich immediately, wrap it in plastic wrap or place it in an airtight container to keep it fresh. Store it in the refrigerator if you want to extend its shelf life, especially if using perishable ingredients.
    
    10. **Enjoy**: Your peanut butter and jelly sandwich is ready to be enjoyed! Pair it with a glass of milk or a piece of fruit for a complete meal.
    
    By following these steps, you can create a tasty and safe peanut butter and jelly sandwich!
    

## Practical Application

Let's apply what we've learned to create a well-structured, balanced instruction for a more complex task.


```python
final_instruction = """
Task: Create a brief lesson plan for teaching basic personal finance to high school students.

Instructions:
1. Start with a concise introduction explaining the importance of personal finance.
2. List 3-5 key topics to cover (e.g., budgeting, saving, understanding credit).
3. For each topic:
   a) Provide a brief explanation suitable for teenagers.
   b) Suggest one practical activity or exercise to reinforce the concept.
4. Conclude with a summary and a suggestion for further learning resources.

Format your response as a structured outline. Aim for clarity and engagement, 
balancing specific examples with general principles that can apply to various 
financial situations. Keep the entire lesson plan to approximately 300 words.
"""

print("Final Instruction Output:")
print(get_completion(final_instruction))
```

    Final Instruction Output:
    ### Lesson Plan: Introduction to Personal Finance for High School Students
    
    #### I. Introduction
    Personal finance is crucial for making informed decisions about money. Understanding how to budget, save, and manage credit can empower students to achieve their financial goals, avoid debt, and build a secure future. This lesson will introduce key concepts that every teenager should know to establish a strong financial foundation.
    
    #### II. Key Topics to Cover
    
    1. **Budgeting**
       - **Explanation**: Budgeting involves tracking income and expenses to ensure that you live within your means. It helps you allocate funds for necessary expenses and savings.
       - **Activity**: Create a simple monthly budget using a template. Students will list hypothetical income (e.g., allowance, part-time job) and expenses (e.g., entertainment, food) to see how they can plan their spending.
    
    2. **Saving**
       - **Explanation**: Saving money is setting aside a portion of your income for future needs or emergencies. It teaches discipline and prepares you for unexpected expenses.
       - **Activity**: Set a savings goal. Students will choose a short-term goal (e.g., a new phone) and calculate how much they need to save each week to reach that goal in three months.
    
    3. **Understanding Credit**
       - **Explanation**: Credit is the ability to borrow money with the promise to pay it back later. Understanding credit scores is essential, as they can impact loan approvals and interest rates.
       - **Activity**: Discuss common credit scenarios (like using a credit card) and have students role-play responsible versus irresponsible credit management.
    
    4. **Investing Basics**
       - **Explanation**: Investing involves putting money into assets (like stocks) with the expectation of generating a profit over time. It’s important for building wealth.
       - **Activity**: Simulate a stock market game where students choose stocks to "invest" in and track their performance over a week.
    
    #### III. Conclusion
    Understanding personal finance is key to making smart financial decisions. By budgeting, saving, and learning about credit, students can build a secure financial future. For further learning, consider resources like "The Millionaire Next Door" by Thomas J. Stanley or online platforms like Khan Academy’s personal finance section.
    
