# Agents with Praison AI
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MervinPraison/PraisonAI/blob/main/cookbooks/notebooks/agents_with_praison_ai_googlecolab.ipynb)

## Dependencies


```python
# Install dependencies without output
%pip install praisonai > /dev/null
%pip install praisonai[crewai] > /dev/null
```

## Tools


```python
# This specific code does not leverage any of the tools
```

## YAML Prompt


```python
agent_yaml = """
framework: crewai
topic: Artificial Intelligence
roles:
  movie_concept_creator:
    backstory: 'Creative thinker with a deep understanding of cinematic storytelling,
      capable of using AI-generated storylines to create unique and compelling movie
      ideas.'
    goal: Generate engaging movie concepts using AI storylines
    role: Movie Concept Creator
    tasks:
      movie_concept_development:
        description: 'Develop movie concepts from AI-generated storylines, ensuring
          they are engaging and have strong narrative arcs.'
        expected_output: 'Well-structured movie concept document with character
          bios, settings, and plot outlines.'
  screenwriter:
    backstory: 'Expert in writing engaging dialogue and script structure, able to
      turn movie concepts into production-ready scripts.'
    goal: Write compelling scripts based on movie concepts
    role: Screenwriter
    tasks:
      scriptwriting_task:
        description: 'Turn movie concepts into polished scripts with well-developed
          characters, strong dialogue, and effective scene transitions.'
        expected_output: 'Production-ready script with a beginning, middle, and
          end, along with character development and engaging dialogues.'
  editor:
    backstory: 'Adept at identifying inconsistencies, improving language usage,
      and maintaining the overall flow of the script.'
    goal: Refine the scripts and ensure continuity of the movie storyline
    role: Editor
    tasks:
      editing_task:
        description: 'Review, edit, and refine the scripts to ensure they are cohesive
          and follow a well-structured narrative.'
        expected_output: 'A polished final draft of the script with no inconsistencies,
          strong character development, and effective dialogue.'
dependencies: []
"""
```

## Main


```python
import os
from praisonai import PraisonAI
from google.colab import userdata

# Create a PraisonAI instance with the agent_yaml content
praisonai = PraisonAI(agent_yaml=agent_yaml)

# Add OPENAI_API_KEY Secrets to Google Colab on the Left Hand Side 🔑 or Enter Manually Below
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

# Run PraisonAI
result = praisonai.run()

# Print the result
print(result)
```

    [1m[95m [2024-11-01 09:42:18][DEBUG]: == Working Agent: Movie Concept Creator[00m
    [1m[95m [2024-11-01 09:42:18][INFO]: == Starting Task: Develop movie concepts from AI-generated storylines, ensuring they are engaging and have strong narrative arcs.[00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI now can give a great answer.
    
    Final Answer: 
    
    **Title: Whispering Shadows**
    
    **Genre:** Supernatural Thriller
    
    **Setting:** A small, isolated coastal town with a rich history and an eerie atmosphere. The town is surrounded by dense forests and cliffs overlooking a tumultuous sea, where fog often blankets the area, adding to its mysterious aura.
    
    **Plot Outline:**
    
    **Act 1:**
    
    - **Introduction:** The story begins with the arrival of **Clara Thompson**, a young journalist seeking solace from a recent heartbreak. She moves into an old, inherited family home on the outskirts of the coastal town, hoping to find peace and write a novel.
    
    - **Inciting Incident:** Clara discovers a hidden room in the house, filled with old journals and artifacts from the town's past. Her curiosity leads her to uncover the story of **Edwin Blackwood**, a reclusive artist rumored to have vanished under mysterious circumstances decades ago.
    
    - **Rising Action:** As Clara delves deeper into Edwin's life, she begins experiencing strange occurrences: whispers in the walls, shadows that move on their own, and vivid dreams of the past. The townspeople are tight-lipped, hinting at dark secrets best left buried.
    
    **Act 2:**
    
    - **Complications:** Clara befriends **Eli Sanders**, a local historian who shares her intrigue about Edwin Blackwood. Together, they piece together clues suggesting Edwin's disappearance is linked to a series of supernatural events involving the town's founding families.
    
    - **Midpoint:** Clara and Eli discover that Edwin's paintings, hidden in the basement, depict the town's darkest secrets and a mysterious entity known as "The Whisperer" that seems to feed on guilt and fear.
    
    - **Tension:** The entity begins to manifest more aggressively, targeting Clara and those around her. Clara's dreams become more intense, revealing that Edwin tried to trap The Whisperer in his artwork but failed, resulting in his disappearance.
    
    **Act 3:**
    
    - **Climax:** Clara and Eli devise a plan to finish Edwin's last painting, hoping to capture The Whisperer once and for all. As they work, the entity attacks, leading to a tense confrontation where Clara must confront her own fears and guilt.
    
    - **Resolution:** With Eli's help, Clara completes the painting, trapping The Whisperer. The town begins to heal as Clara decides to stay, turning Edwin's house into a gallery to honor his memory and the town's history.
    
    - **Denouement:** In the final scene, Clara finds a hidden journal entry from Edwin, thanking whoever set him free. As she reads, a gentle whisper echoes through the room, leaving the audience with a sense of lingering mystery.
    
    **Character Bios:**
    
    - **Clara Thompson:** A determined and introspective journalist in her late 20s. She is resourceful, with a deep sense of curiosity, driven by both her professional ambitions and personal quest for healing.
    
    - **Eli Sanders:** A knowledgeable and slightly eccentric historian in his early 30s. Eli is passionate about the town's history and becomes Clara's ally. He possesses a keen intellect and a hidden past that connects him to the town's mysteries.
    
    - **Edwin Blackwood:** A talented but troubled artist from the early 20th century. Edwin was obsessed with capturing the supernatural on canvas, which led to his mysterious disappearance. His legacy becomes the key to unraveling the town's secrets.
    
    **Themes:** The film explores themes of redemption, the power of art, the weight of history, and confronting one's inner demons. It combines elements of suspense, mystery, and the supernatural to create an engaging narrative that keeps audiences on the edge of their seats.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-01 09:42:34][DEBUG]: == [Movie Concept Creator] Task output: **Title: Whispering Shadows**
    
    **Genre:** Supernatural Thriller
    
    **Setting:** A small, isolated coastal town with a rich history and an eerie atmosphere. The town is surrounded by dense forests and cliffs overlooking a tumultuous sea, where fog often blankets the area, adding to its mysterious aura.
    
    **Plot Outline:**
    
    **Act 1:**
    
    - **Introduction:** The story begins with the arrival of **Clara Thompson**, a young journalist seeking solace from a recent heartbreak. She moves into an old, inherited family home on the outskirts of the coastal town, hoping to find peace and write a novel.
    
    - **Inciting Incident:** Clara discovers a hidden room in the house, filled with old journals and artifacts from the town's past. Her curiosity leads her to uncover the story of **Edwin Blackwood**, a reclusive artist rumored to have vanished under mysterious circumstances decades ago.
    
    - **Rising Action:** As Clara delves deeper into Edwin's life, she begins experiencing strange occurrences: whispers in the walls, shadows that move on their own, and vivid dreams of the past. The townspeople are tight-lipped, hinting at dark secrets best left buried.
    
    **Act 2:**
    
    - **Complications:** Clara befriends **Eli Sanders**, a local historian who shares her intrigue about Edwin Blackwood. Together, they piece together clues suggesting Edwin's disappearance is linked to a series of supernatural events involving the town's founding families.
    
    - **Midpoint:** Clara and Eli discover that Edwin's paintings, hidden in the basement, depict the town's darkest secrets and a mysterious entity known as "The Whisperer" that seems to feed on guilt and fear.
    
    - **Tension:** The entity begins to manifest more aggressively, targeting Clara and those around her. Clara's dreams become more intense, revealing that Edwin tried to trap The Whisperer in his artwork but failed, resulting in his disappearance.
    
    **Act 3:**
    
    - **Climax:** Clara and Eli devise a plan to finish Edwin's last painting, hoping to capture The Whisperer once and for all. As they work, the entity attacks, leading to a tense confrontation where Clara must confront her own fears and guilt.
    
    - **Resolution:** With Eli's help, Clara completes the painting, trapping The Whisperer. The town begins to heal as Clara decides to stay, turning Edwin's house into a gallery to honor his memory and the town's history.
    
    - **Denouement:** In the final scene, Clara finds a hidden journal entry from Edwin, thanking whoever set him free. As she reads, a gentle whisper echoes through the room, leaving the audience with a sense of lingering mystery.
    
    **Character Bios:**
    
    - **Clara Thompson:** A determined and introspective journalist in her late 20s. She is resourceful, with a deep sense of curiosity, driven by both her professional ambitions and personal quest for healing.
    
    - **Eli Sanders:** A knowledgeable and slightly eccentric historian in his early 30s. Eli is passionate about the town's history and becomes Clara's ally. He possesses a keen intellect and a hidden past that connects him to the town's mysteries.
    
    - **Edwin Blackwood:** A talented but troubled artist from the early 20th century. Edwin was obsessed with capturing the supernatural on canvas, which led to his mysterious disappearance. His legacy becomes the key to unraveling the town's secrets.
    
    **Themes:** The film explores themes of redemption, the power of art, the weight of history, and confronting one's inner demons. It combines elements of suspense, mystery, and the supernatural to create an engaging narrative that keeps audiences on the edge of their seats.
    
    [00m
    [1m[95m [2024-11-01 09:42:34][DEBUG]: == Working Agent: Screenwriter[00m
    [1m[95m [2024-11-01 09:42:34][INFO]: == Starting Task: Turn movie concepts into polished scripts with well-developed characters, strong dialogue, and effective scene transitions.[00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mThought: I now can give a great answer  
    Final Answer:  
    
    **Title: Whispering Shadows**
    
    **Genre:** Supernatural Thriller
    
    **Setting:** A small, isolated coastal town with a rich history and an eerie atmosphere. The town is surrounded by dense forests and cliffs overlooking a tumultuous sea, where fog often blankets the area, adding to its mysterious aura.
    
    ---
    
    **ACT 1**
    
    **INT. INHERITED FAMILY HOME - DAY**
    
    *The camera follows CLARA THOMPSON, late 20s, as she steps into the dusty, old house. Her eyes scan the surroundings, taking in the antique furniture and faded photographs lining the walls. She sets her suitcase down, letting out a deep sigh.*
    
    **CLARA**  
    (softly to herself)  
    A fresh start.
    
    *She moves through the house, her footsteps echoing. Suddenly, a draft catches her attention, leading her to a hidden door, slightly ajar.*
    
    **INT. HIDDEN ROOM - DAY**
    
    *Clara pushes the door open, revealing a room covered in dust. In the center, a collection of journals and artifacts are stacked haphazardly. Curiosity piqued, Clara picks up a journal, flipping through its pages.*
    
    **JOURNAL ENTRY (V.O.)**  
    "The whispers grow louder each night. Edwin Blackwood, 1923."
    
    *Clara's eyes widen, a mix of fear and intrigue washing over her.*
    
    ---
    
    **ACT 2**
    
    **EXT. COASTAL TOWN - DAY**
    
    *Clara walks through the town, absorbing its quaint charm. She approaches a small bookstore, where she meets ELI SANDERS, early 30s, sorting through historical texts.*
    
    **CLARA**  
    Excuse me, are you Eli Sanders?
    
    **ELI**  
    (looking up, smiling)  
    The one and only. What can I do for you?
    
    *Clara explains her discovery, and Eli's interest is piqued. They discuss the mysterious Edwin Blackwood and the town's history.*
    
    **INT. EDWIN'S BASEMENT - NIGHT**
    
    *Clara and Eli descend into the dimly lit basement, flashlights in hand. Paintings of shadowy figures line the walls, their haunting presence undeniable.*
    
    **ELI**  
    (pointing at a painting)  
    This... this is "The Whisperer." It's said to thrive on guilt and fear.
    
    *Clara shivers, feeling the weight of the entity's power.*
    
    ---
    
    **ACT 3**
    
    **INT. INHERITED FAMILY HOME - NIGHT**
    
    *Clara and Eli work tirelessly on a canvas, replicating Edwin's unfinished work. The room is thick with tension as shadows dance around them.*
    
    **CLARA**  
    (paintbrush in hand, determined)  
    We have to do this. For Edwin. For the town.
    
    *The entity manifests, a swirling darkness that threatens to engulf them. Clara confronts her deepest fears, memories of her heartbreak flashing before her eyes.*
    
    **CLARA**  
    (shouting)  
    You won't control me!
    
    *With a final stroke, the painting is complete. A blinding light envelops the room, and the entity's presence dissipates.*
    
    **EXT. COASTAL TOWN - DAY**
    
    *The town feels lighter, the oppressive atmosphere lifted. Clara stands outside Edwin's house, now transformed into a gallery.*
    
    **INT. GALLERY - DAY**
    
    *Clara reads the hidden journal entry from Edwin, a tear rolling down her cheek.*
    
    **EDWIN'S JOURNAL (V.O.)**  
    "To the one who set me free, thank you."
    
    *As she closes the journal, a gentle whisper echoes, leaving Clara with a sense of peace.*
    
    **FADE OUT.**
    
    ---
    
    **Themes:** The film explores themes of redemption, the power of art, the weight of history, and confronting one's inner demons. It combines elements of suspense, mystery, and the supernatural to create an engaging narrative that keeps audiences on the edge of their seats.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-01 09:42:44][DEBUG]: == [Screenwriter] Task output: **Title: Whispering Shadows**
    
    **Genre:** Supernatural Thriller
    
    **Setting:** A small, isolated coastal town with a rich history and an eerie atmosphere. The town is surrounded by dense forests and cliffs overlooking a tumultuous sea, where fog often blankets the area, adding to its mysterious aura.
    
    ---
    
    **ACT 1**
    
    **INT. INHERITED FAMILY HOME - DAY**
    
    *The camera follows CLARA THOMPSON, late 20s, as she steps into the dusty, old house. Her eyes scan the surroundings, taking in the antique furniture and faded photographs lining the walls. She sets her suitcase down, letting out a deep sigh.*
    
    **CLARA**  
    (softly to herself)  
    A fresh start.
    
    *She moves through the house, her footsteps echoing. Suddenly, a draft catches her attention, leading her to a hidden door, slightly ajar.*
    
    **INT. HIDDEN ROOM - DAY**
    
    *Clara pushes the door open, revealing a room covered in dust. In the center, a collection of journals and artifacts are stacked haphazardly. Curiosity piqued, Clara picks up a journal, flipping through its pages.*
    
    **JOURNAL ENTRY (V.O.)**  
    "The whispers grow louder each night. Edwin Blackwood, 1923."
    
    *Clara's eyes widen, a mix of fear and intrigue washing over her.*
    
    ---
    
    **ACT 2**
    
    **EXT. COASTAL TOWN - DAY**
    
    *Clara walks through the town, absorbing its quaint charm. She approaches a small bookstore, where she meets ELI SANDERS, early 30s, sorting through historical texts.*
    
    **CLARA**  
    Excuse me, are you Eli Sanders?
    
    **ELI**  
    (looking up, smiling)  
    The one and only. What can I do for you?
    
    *Clara explains her discovery, and Eli's interest is piqued. They discuss the mysterious Edwin Blackwood and the town's history.*
    
    **INT. EDWIN'S BASEMENT - NIGHT**
    
    *Clara and Eli descend into the dimly lit basement, flashlights in hand. Paintings of shadowy figures line the walls, their haunting presence undeniable.*
    
    **ELI**  
    (pointing at a painting)  
    This... this is "The Whisperer." It's said to thrive on guilt and fear.
    
    *Clara shivers, feeling the weight of the entity's power.*
    
    ---
    
    **ACT 3**
    
    **INT. INHERITED FAMILY HOME - NIGHT**
    
    *Clara and Eli work tirelessly on a canvas, replicating Edwin's unfinished work. The room is thick with tension as shadows dance around them.*
    
    **CLARA**  
    (paintbrush in hand, determined)  
    We have to do this. For Edwin. For the town.
    
    *The entity manifests, a swirling darkness that threatens to engulf them. Clara confronts her deepest fears, memories of her heartbreak flashing before her eyes.*
    
    **CLARA**  
    (shouting)  
    You won't control me!
    
    *With a final stroke, the painting is complete. A blinding light envelops the room, and the entity's presence dissipates.*
    
    **EXT. COASTAL TOWN - DAY**
    
    *The town feels lighter, the oppressive atmosphere lifted. Clara stands outside Edwin's house, now transformed into a gallery.*
    
    **INT. GALLERY - DAY**
    
    *Clara reads the hidden journal entry from Edwin, a tear rolling down her cheek.*
    
    **EDWIN'S JOURNAL (V.O.)**  
    "To the one who set me free, thank you."
    
    *As she closes the journal, a gentle whisper echoes, leaving Clara with a sense of peace.*
    
    **FADE OUT.**
    
    ---
    
    **Themes:** The film explores themes of redemption, the power of art, the weight of history, and confronting one's inner demons. It combines elements of suspense, mystery, and the supernatural to create an engaging narrative that keeps audiences on the edge of their seats.
    
    [00m
    [1m[95m [2024-11-01 09:42:44][DEBUG]: == Working Agent: Editor[00m
    [1m[95m [2024-11-01 09:42:44][INFO]: == Starting Task: Review, edit, and refine the scripts to ensure they are cohesive and follow a well-structured narrative.[00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI now can give a great answer.
    
    Final Answer:
    
    **Title: Whispering Shadows**
    
    **Genre:** Supernatural Thriller
    
    **Setting:** A small, isolated coastal town with a rich history and an eerie atmosphere. The town is surrounded by dense forests and cliffs overlooking a tumultuous sea, where fog often blankets the area, adding to its mysterious aura.
    
    ---
    
    **ACT 1**
    
    **INT. INHERITED FAMILY HOME - DAY**
    
    *The camera follows CLARA THOMPSON, late 20s, as she steps into the dusty, old house. Her eyes scan the surroundings, taking in the antique furniture and faded photographs lining the walls. She sets her suitcase down, letting out a deep sigh.*
    
    **CLARA**  
    (softly to herself)  
    A fresh start.
    
    *She moves through the house, her footsteps echoing. Suddenly, a draft catches her attention, leading her to a hidden door, slightly ajar.*
    
    **INT. HIDDEN ROOM - DAY**
    
    *Clara pushes the door open, revealing a room covered in dust. In the center, a collection of journals and artifacts are stacked haphazardly. Curiosity piqued, Clara picks up a journal, flipping through its pages.*
    
    **JOURNAL ENTRY (V.O.)**  
    "The whispers grow louder each night. Edwin Blackwood, 1923."
    
    *Clara's eyes widen, a mix of fear and intrigue washing over her.*
    
    ---
    
    **ACT 2**
    
    **EXT. COASTAL TOWN - DAY**
    
    *Clara walks through the town, absorbing its quaint charm. She approaches a small bookstore, where she meets ELI SANDERS, early 30s, sorting through historical texts.*
    
    **CLARA**  
    Excuse me, are you Eli Sanders?
    
    **ELI**  
    (looking up, smiling)  
    The one and only. What can I do for you?
    
    *Clara explains her discovery, and Eli's interest is piqued. They discuss the mysterious Edwin Blackwood and the town's history.*
    
    **INT. EDWIN'S BASEMENT - NIGHT**
    
    *Clara and Eli descend into the dimly lit basement, flashlights in hand. Paintings of shadowy figures line the walls, their haunting presence undeniable.*
    
    **ELI**  
    (pointing at a painting)  
    This... this is "The Whisperer." It's said to thrive on guilt and fear.
    
    *Clara shivers, feeling the weight of the entity's power.*
    
    ---
    
    **ACT 3**
    
    **INT. INHERITED FAMILY HOME - NIGHT**
    
    *Clara and Eli work tirelessly on a canvas, replicating Edwin's unfinished work. The room is thick with tension as shadows dance around them.*
    
    **CLARA**  
    (paintbrush in hand, determined)  
    We have to do this. For Edwin. For the town.
    
    *The entity manifests, a swirling darkness that threatens to engulf them. Clara confronts her deepest fears, memories of her heartbreak flashing before her eyes.*
    
    **CLARA**  
    (shouting)  
    You won't control me!
    
    *With a final stroke, the painting is complete. A blinding light envelops the room, and the entity's presence dissipates.*
    
    **EXT. COASTAL TOWN - DAY**
    
    *The town feels lighter, the oppressive atmosphere lifted. Clara stands outside Edwin's house, now transformed into a gallery.*
    
    **INT. GALLERY - DAY**
    
    *Clara reads the hidden journal entry from Edwin, a tear rolling down her cheek.*
    
    **EDWIN'S JOURNAL (V.O.)**  
    "To the one who set me free, thank you."
    
    *As she closes the journal, a gentle whisper echoes, leaving Clara with a sense of peace.*
    
    **FADE OUT.**
    
    ---
    
    **Themes:** The film explores themes of redemption, the power of art, the weight of history, and confronting one's inner demons. It combines elements of suspense, mystery, and the supernatural to create an engaging narrative that keeps audiences on the edge of their seats.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-01 09:42:55][DEBUG]: == [Editor] Task output: **Title: Whispering Shadows**
    
    **Genre:** Supernatural Thriller
    
    **Setting:** A small, isolated coastal town with a rich history and an eerie atmosphere. The town is surrounded by dense forests and cliffs overlooking a tumultuous sea, where fog often blankets the area, adding to its mysterious aura.
    
    ---
    
    **ACT 1**
    
    **INT. INHERITED FAMILY HOME - DAY**
    
    *The camera follows CLARA THOMPSON, late 20s, as she steps into the dusty, old house. Her eyes scan the surroundings, taking in the antique furniture and faded photographs lining the walls. She sets her suitcase down, letting out a deep sigh.*
    
    **CLARA**  
    (softly to herself)  
    A fresh start.
    
    *She moves through the house, her footsteps echoing. Suddenly, a draft catches her attention, leading her to a hidden door, slightly ajar.*
    
    **INT. HIDDEN ROOM - DAY**
    
    *Clara pushes the door open, revealing a room covered in dust. In the center, a collection of journals and artifacts are stacked haphazardly. Curiosity piqued, Clara picks up a journal, flipping through its pages.*
    
    **JOURNAL ENTRY (V.O.)**  
    "The whispers grow louder each night. Edwin Blackwood, 1923."
    
    *Clara's eyes widen, a mix of fear and intrigue washing over her.*
    
    ---
    
    **ACT 2**
    
    **EXT. COASTAL TOWN - DAY**
    
    *Clara walks through the town, absorbing its quaint charm. She approaches a small bookstore, where she meets ELI SANDERS, early 30s, sorting through historical texts.*
    
    **CLARA**  
    Excuse me, are you Eli Sanders?
    
    **ELI**  
    (looking up, smiling)  
    The one and only. What can I do for you?
    
    *Clara explains her discovery, and Eli's interest is piqued. They discuss the mysterious Edwin Blackwood and the town's history.*
    
    **INT. EDWIN'S BASEMENT - NIGHT**
    
    *Clara and Eli descend into the dimly lit basement, flashlights in hand. Paintings of shadowy figures line the walls, their haunting presence undeniable.*
    
    **ELI**  
    (pointing at a painting)  
    This... this is "The Whisperer." It's said to thrive on guilt and fear.
    
    *Clara shivers, feeling the weight of the entity's power.*
    
    ---
    
    **ACT 3**
    
    **INT. INHERITED FAMILY HOME - NIGHT**
    
    *Clara and Eli work tirelessly on a canvas, replicating Edwin's unfinished work. The room is thick with tension as shadows dance around them.*
    
    **CLARA**  
    (paintbrush in hand, determined)  
    We have to do this. For Edwin. For the town.
    
    *The entity manifests, a swirling darkness that threatens to engulf them. Clara confronts her deepest fears, memories of her heartbreak flashing before her eyes.*
    
    **CLARA**  
    (shouting)  
    You won't control me!
    
    *With a final stroke, the painting is complete. A blinding light envelops the room, and the entity's presence dissipates.*
    
    **EXT. COASTAL TOWN - DAY**
    
    *The town feels lighter, the oppressive atmosphere lifted. Clara stands outside Edwin's house, now transformed into a gallery.*
    
    **INT. GALLERY - DAY**
    
    *Clara reads the hidden journal entry from Edwin, a tear rolling down her cheek.*
    
    **EDWIN'S JOURNAL (V.O.)**  
    "To the one who set me free, thank you."
    
    *As she closes the journal, a gentle whisper echoes, leaving Clara with a sense of peace.*
    
    **FADE OUT.**
    
    ---
    
    **Themes:** The film explores themes of redemption, the power of art, the weight of history, and confronting one's inner demons. It combines elements of suspense, mystery, and the supernatural to create an engaging narrative that keeps audiences on the edge of their seats.
    
    [00m
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">### Task Output ###
**Title: Whispering Shadows**

**Genre:** Supernatural Thriller

**Setting:** A small, isolated coastal town with a rich history and an eerie atmosphere. The town is surrounded by 
dense forests and cliffs overlooking a tumultuous sea, where fog often blankets the area, adding to its mysterious 
aura.

---

**ACT <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>**

**INT. INHERITED FAMILY HOME - DAY**

*The camera follows CLARA THOMPSON, late 20s, as she steps into the dusty, old house. Her eyes scan the 
surroundings, taking in the antique furniture and faded photographs lining the walls. She sets her suitcase down, 
letting out a deep sigh.*

**CLARA**  
<span style="font-weight: bold">(</span>softly to herself<span style="font-weight: bold">)</span>  
A fresh start.

*She moves through the house, her footsteps echoing. Suddenly, a draft catches her attention, leading her to a 
hidden door, slightly ajar.*

**INT. HIDDEN ROOM - DAY**

*Clara pushes the door open, revealing a room covered in dust. In the center, a collection of journals and 
artifacts are stacked haphazardly. Curiosity piqued, Clara picks up a journal, flipping through its pages.*

**JOURNAL ENTRY <span style="font-weight: bold">(</span>V.O.<span style="font-weight: bold">)</span>**  
<span style="color: #008000; text-decoration-color: #008000">"The whispers grow louder each night. Edwin Blackwood, 1923."</span>

*Clara's eyes widen, a mix of fear and intrigue washing over her.*

---

**ACT <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>**

**EXT. COASTAL TOWN - DAY**

*Clara walks through the town, absorbing its quaint charm. She approaches a small bookstore, where she meets ELI 
SANDERS, early 30s, sorting through historical texts.*

**CLARA**  
Excuse me, are you Eli Sanders?

**ELI**  
<span style="font-weight: bold">(</span>looking up, smiling<span style="font-weight: bold">)</span>  
The one and only. What can I do for you?

*Clara explains her discovery, and Eli's interest is piqued. They discuss the mysterious Edwin Blackwood and the 
town's history.*

**INT. EDWIN'S BASEMENT - NIGHT**

*Clara and Eli descend into the dimly lit basement, flashlights in hand. Paintings of shadowy figures line the 
walls, their haunting presence undeniable.*

**ELI**  
<span style="font-weight: bold">(</span>pointing at a painting<span style="font-weight: bold">)</span>  
This<span style="color: #808000; text-decoration-color: #808000">...</span> this is <span style="color: #008000; text-decoration-color: #008000">"The Whisperer."</span> It's said to thrive on guilt and fear.

*Clara shivers, feeling the weight of the entity's power.*

---

**ACT <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>**

**INT. INHERITED FAMILY HOME - NIGHT**

*Clara and Eli work tirelessly on a canvas, replicating Edwin's unfinished work. The room is thick with tension as 
shadows dance around them.*

**CLARA**  
<span style="font-weight: bold">(</span>paintbrush in hand, determined<span style="font-weight: bold">)</span>  
We have to do this. For Edwin. For the town.

*The entity manifests, a swirling darkness that threatens to engulf them. Clara confronts her deepest fears, 
memories of her heartbreak flashing before her eyes.*

**CLARA**  
<span style="font-weight: bold">(</span>shouting<span style="font-weight: bold">)</span>  
You won't control me!

*With a final stroke, the painting is complete. A blinding light envelops the room, and the entity's presence 
dissipates.*

**EXT. COASTAL TOWN - DAY**

*The town feels lighter, the oppressive atmosphere lifted. Clara stands outside Edwin's house, now transformed into
a gallery.*

**INT. GALLERY - DAY**

*Clara reads the hidden journal entry from Edwin, a tear rolling down her cheek.*

**EDWIN'S JOURNAL <span style="font-weight: bold">(</span>V.O.<span style="font-weight: bold">)</span>**  
<span style="color: #008000; text-decoration-color: #008000">"To the one who set me free, thank you."</span>

*As she closes the journal, a gentle whisper echoes, leaving Clara with a sense of peace.*

**FADE OUT.**

---

**Themes:** The film explores themes of redemption, the power of art, the weight of history, and confronting one's 
inner demons. It combines elements of suspense, mystery, and the supernatural to create an engaging narrative that 
keeps audiences on the edge of their seats.
</pre>



    None
    
