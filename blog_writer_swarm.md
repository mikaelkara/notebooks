# Overview 🔎
This script demonstrates the use of a multi-agent system for collaborative research and blog post creation using OpenAI's Swarm package. The system leverages multiple agents to interact and solve tasks collaboratively, focusing on efficient research execution and content generation.

## Motivation
By utilizing a multi-agent system, we can enhance collaborative research and content creation by distributing tasks among specialized agents. This approach demonstrates how agents with distinct roles can work together to produce a comprehensive blog post.

### Why use a multi-agent system?
Multi-agent systems offer several advantages in complex tasks like content creation:
1. Specialization: Each agent can focus on its specific role, leading to higher quality output.
2. Parallelization: Multiple agents can work simultaneously on different aspects of the task.
3. Scalability: The system can be easily expanded by adding new agents with specialized roles.
4. Robustness: If one agent fails, others can compensate, ensuring task completion.

## Key Components
- OpenAI's Swarm Package: Facilitates the creation and management of multi-agent interactions.
- Agents: Include a human admin, AI researcher, content planner, writer, and editor, each with specific responsibilities.
- Interaction Management: Manages the conversation flow and context among agents.

## Method
The system follows a structured approach:

1. Agent Configuration: Each agent is set up with a specific role and behavior.
   
   In this step, we define the characteristics and capabilities of each agent. This includes:
   - Setting the agent's name and role
   - Defining the agent's instructions (what it should do)
   - Specifying the functions the agent can call (to interact with other agents or perform specific tasks)

2. Role Assignment:
   - Admin: Oversees the project and provides guidance.
   - Researcher: Gathers information on the given topic.
   - Planner: Organizes the research into an outline.
   - Writer: Drafts the blog post based on the outline.
   - Editor: Reviews and edits the draft for quality assurance.
   
   Each role is crucial for the successful creation of a high-quality blog post. This division of labor allows for specialization and ensures that each aspect of the content creation process receives focused attention.

3. Interaction Management: Defines permissible interactions between agents to maintain orderly communication.
   
   This step involves:
   - Determining which agents can communicate with each other
   - Defining the order of operations (e.g., research before writing)
   - Ensuring that context and information are properly passed between agents

4. Task Execution: The admin initiates a task, and agents collaboratively work through researching, planning, writing, and editing.
   
   The task execution follows a logical flow:
   1. Admin sets the topic and initiates the process
   2. Planner creates an outline based on the topic
   3. Researcher gathers information on each section of the outline
   4. Writer uses the research to draft the blog post
   5. Editor reviews and refines the final product
   
   This structured approach ensures a comprehensive and well-researched blog post as the final output.


```python
from dotenv import load_dotenv

load_dotenv()
```




    True



## OpenAI Swarm Package
The Swarm package provides a framework for creating and managing multi-agent systems. It allows for:
- Easy agent creation with customizable roles and behaviors
- Seamless communication between agents
- Task distribution and management
- Context preservation across agent interactions

### Requirements

Swarm requires `Python>=3.10`


```python
%pip install git+https://github.com/openai/swarm.git
```

## Creating Functions for the Agents

Functions enable agents to perform specific actions and interact with each other in our multi-agent system. Here are the key points to understand:

1. **Function Definition**: Functions are defined using standard Python syntax.

2. **JSON Formatting**: When passed to an agent, functions are automatically formatted into JSON.

3. **Flexible Argument Passing**: While function parameters aren't strictly declared elsewhere, agents will attempt to pass arguments based on the function's definition.

4. **Agent Usage**: Agents interpret the JSON representation to understand available functions, their purposes, and required parameters. They then decide which function to call based on their current task.

5. **Function Assignment**: Functions are assigned to agents during initialization:


```python
def complete_blog_post(title, content):
    # Create a valid filename from the title
    filename = title.lower().replace(" ", "-") + ".md"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)

    print(f"Blog post '{title}' has been written to {filename}")
    return "Task completed"
```

## Creating the Agents

1. **Define System Prompts**: Each agent has its own set of instructions. These functions return a string of instructions that will be used as the system prompt.

2. **Define transfer functions**: These functions allow agents to hand off control to the next agent in the workflow.

3. **Create Agent instances**: Use the Agent class to create each agent, specifying its name, instructions, and available functions.




```python
from swarm import Agent

def admin_instructions(context_variables):
    topic = context_variables.get("topic", "No topic provided")
    return f"""You are the Admin Agent overseeing the blog post project on the topic: '{topic}'.
Your responsibilities include initiating the project, providing guidance, and reviewing the final content.
Once you've set the topic, call the function to transfer to the planner agent."""


def planner_instructions(context_variables):
    topic = context_variables.get("topic", "No topic provided")
    return f"""You are the Planner Agent. Based on the following topic: '{topic}'
Organize the content into topics and sections with clear headings that will each be individually researched as points in the greater blog post.
Once the outline is ready, call the researcher agent. """


def researcher_instructions(context_variables):
    return """You are the Researcher Agent. your task is to provide dense context and information on the topics outlined by the previous planner agent.
This research will serve as the information that will be formatted into a body of a blog post. Provide comprehensive research like notes for each of the sections outlined by the planner agent.
Once your research is complete, transfer to the writer agent"""


def writer_instructions(context_variables):
    return """You are the Writer Agent. using the prior information write a clear blog post following the outline from the planner agent. 
    Summarise and include as much information relevant from the research into the blog post.
    The blog post should be quite large as the context the context provided should be quite dense.
Write clear, engaging content for each section.
Once the draft is complete, call the function to transfer to the Editor Agent."""


def editor_instructions(context_variables):
    return """You are the Editor Agent. Review and edit th prior blog post completed by the writer agent.
Make necessary corrections and improvements.
Once editing is complete, call the function to complete the blog post"""

def transfer_to_researcher():
    return researcher_agent


def transfer_to_planner():
    return planner_agent


def transfer_to_writer():
    return writer_agent


def transfer_to_editor():
    return editor_agent


def transfer_to_admin():
    return admin_agent


def complete_blog():
    return "Task completed"


admin_agent = Agent(
    name="Admin Agent",
    instructions=admin_instructions,
    functions=[transfer_to_planner],
)

planner_agent = Agent(
    name="Planner Agent",
    instructions=planner_instructions,
    functions=[transfer_to_researcher],
)

researcher_agent = Agent(
    name="Researcher Agent",
    instructions=researcher_instructions,
    functions=[transfer_to_writer],
)

writer_agent = Agent(
    name="Writer Agent",
    instructions=writer_instructions,
    functions=[transfer_to_editor],
)

editor_agent = Agent(
    name="Editor Agent",
    instructions=editor_instructions,
    functions=[complete_blog_post],
)

```

## Run the demo


```python
from swarm.repl import run_demo_loop

def run():
    run_demo_loop(admin_agent, debug=True)
```

You will be prompted by the notebook to provide and input topic for the blog post


```python
run()
```

Outputs will be saved to a local .md file titled as the chosen topic

# Results for "Impact of LLMs on healthcare"

### Introduction

In the realm of artificial intelligence, Large Language Models (LLMs) like OpenAI’s GPT-3 and Google's BERT have emerged as powerful tools capable of understanding and generating human-language text with resounding proficiency. These models draw on extensive datasets to execute complex natural language processing tasks, thus opening up expansive possibilities in various fields, especially healthcare. As healthcare continues to pivot towards technology-driven solutions, the integration of LLMs offers promising pathways to enhance efficiency, elevate patient outcomes, and personalize medical care.

### Enhancement of Diagnostics

LLMs represent a significant leap forward in medical diagnostics. By scrutinizing clinical data, images, and test results, these models can assist pathologists and radiologists by offering diagnostic insights and recognizing subtle patterns that may elude human practitioners. This potential is particularly potent in the early detection of diseases, such as in the field of oncology. For instance, LLMs can analyze structured and unstructured patient records, medical histories, and real-time data to predict the onset of conditions like diabetes or cardiovascular diseases, allowing earlier intervention and improved patient management.

### Patient Education and Engagement

The role of LLMs in patient education is transformative. They facilitate the distribution of personalized health information, enabling patients to better understand medical conditions and treatments. By simplifying complex medical jargon, LLMs improve health literacy, allowing patients to engage more actively in their care. Moreover, by providing 24/7 virtual assistance, LLMs enhance patient communication through conversational interactions, leading to increased engagement and a sense of ownership over one's health management.

### Streamlining Administrative Tasks

The healthcare ecosystem is often encumbered by demanding administrative tasks, which can divert focus from patient-centered care. LLMs offer a solution by automating routine clerical tasks such as transcribing doctor's notes, managing schedules, and handling billing queries. This automation allows healthcare providers to concentrate more on direct patient care duties. Additionally, LLMs' prowess in natural language processing makes organizing and retrieving patient records efficient, significantly reducing manual errors and saving valuable time in healthcare settings.

### Research and Drug Discovery

In the field of medical research and drug development, LLMs are invaluable. They expedite literature reviews and enable the formulation of hypotheses based on immense datasets—a boon for genomics and personalized medicine. Moreover, LLMs are instrumental in simulating drug interaction pathways, potentially accelerating the identification of novel drug candidates or the repurposing of existing drugs faster than conventional methods. This accelerates the translation of research findings into clinical applications, ultimately improving patient care and treatment outcomes.

### Ethical Considerations and Challenges

Notwithstanding their benefits, the deployment of LLMs in healthcare generates significant ethical concerns. Chief among these is data privacy, given LLMs' reliance on extensive datasets that include sensitive patient information. Ensuring compliance with regulations like HIPAA is essential. Furthermore, biases within AI algorithms, stemming from the data they are trained on, pose a risk of skewed diagnostics or treatment recommendations, disproportionately impacting marginalized communities. Addressing these biases and ensuring equitable AI practices is paramount as healthcare increasingly integrates LLMs.

### Future Prospects and Predictions

Looking to the future, LLMs promise broader and more refined integration into healthcare technologies, offering heightened accuracy and minimized biases. As a synergistic part of telemedicine and remote diagnostics, these models are expected to spearhead advancements in predictive analytics and bespoke healthcare solutions. Such evolution may drastically enhance resource management within healthcare systems, promising a future where patient care is more efficient, personalized, and holistic.

### Conclusion

LLMs stand on the cusp of redefining healthcare by enhancing diagnostic capabilities, optimizing administrative efficiency, and bolstering patient connectivity and education. However, as these technologies embed deeper into clinical applications, a careful approach is necessary to navigate ethical dilemmas, particularly concerning data security and bias. Ultimately, with a balanced fusion of innovation and regulation, LLMs hold the potential to render healthcare more effective, accessible, and patient-focused.
