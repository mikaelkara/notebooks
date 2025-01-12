# History and Data Analysis Collaboration System

## Overview
This notebook implements a multi-agent collaboration system that combines historical research with data analysis to answer complex historical questions. It leverages the power of large language models to simulate specialized agents working together to provide comprehensive answers.

## Motivation
Historical analysis often requires both deep contextual understanding and quantitative data interpretation. By creating a system that combines these two aspects, we aim to provide more robust and insightful answers to complex historical questions. This approach mimics real-world collaboration between historians and data analysts, potentially leading to more nuanced and data-driven historical insights.

## Key Components
1. **Agent Class**: A base class for creating specialized AI agents.
2. **HistoryResearchAgent**: Specialized in historical context and trends.
3. **DataAnalysisAgent**: Focused on interpreting numerical data and statistics.
4. **HistoryDataCollaborationSystem**: Orchestrates the collaboration between agents.

## Method Details
The collaboration system follows these steps:
1. **Historical Context**: The History Agent provides relevant historical background.
2. **Data Needs Identification**: The Data Agent determines what quantitative information is needed.
3. **Historical Data Provision**: The History Agent supplies relevant historical data.
4. **Data Analysis**: The Data Agent interprets the provided historical data.
5. **Final Synthesis**: The History Agent combines all insights into a comprehensive answer.

This iterative process allows for a back-and-forth between historical context and data analysis, mimicking real-world collaborative research.

## Conclusion
The History and Data Analysis Collaboration System demonstrates the potential of multi-agent AI systems in tackling complex, interdisciplinary questions. By combining the strengths of historical research and data analysis, it offers a novel approach to understanding historical trends and events. This system could be valuable for researchers, educators, and anyone interested in gaining deeper insights into historical topics.

Future improvements could include adding more specialized agents, incorporating external data sources, and refining the collaboration process for even more nuanced analyses.

### Import required libraries


```python
import os
import time

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

### Initialize the language model


```python
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0.7)
```

### Define the base Agent class


```python
class Agent:
    def __init__(self, name: str, role: str, skills: List[str]):
        self.name = name
        self.role = role
        self.skills = skills
        self.llm = llm

    def process(self, task: str, context: List[Dict] = None) -> str:
        messages = [
            SystemMessage(content=f"You are {self.name}, a {self.role}. Your skills include: {', '.join(self.skills)}. Respond to the task based on your role and skills.")
        ]
        
        if context:
            for msg in context:
                if msg['role'] == 'human':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'ai':
                    messages.append(AIMessage(content=msg['content']))
        
        messages.append(HumanMessage(content=task))
        response = self.llm.invoke(messages)
        return response.content
```

### Define specialized agents: HistoryResearchAgent and DataAnalysisAgent


```python
class HistoryResearchAgent(Agent):
    def __init__(self):
        super().__init__("Clio", "History Research Specialist", ["deep knowledge of historical events", "understanding of historical contexts", "identifying historical trends"])

class DataAnalysisAgent(Agent):
    def __init__(self):
        super().__init__("Data", "Data Analysis Expert", ["interpreting numerical data", "statistical analysis", "data visualization description"])
```

### Define the different functions for the collaboration system

#### Research Historical Context


```python
def research_historical_context(history_agent, task: str, context: list) -> list:
    print("🏛️ History Agent: Researching historical context...")
    history_task = f"Provide relevant historical context and information for the following task: {task}"
    history_result = history_agent.process(history_task)
    context.append({"role": "ai", "content": f"History Agent: {history_result}"})
    print(f"📜 Historical context provided: {history_result[:100]}...\n")
    return context
```

#### Identify Data Needs


```python
def identify_data_needs(data_agent, task: str, context: list) -> list:
    print("📊 Data Agent: Identifying data needs based on historical context...")
    historical_context = context[-1]["content"]
    data_need_task = f"Based on the historical context, what specific data or statistical information would be helpful to answer the original question? Historical context: {historical_context}"
    data_need_result = data_agent.process(data_need_task, context)
    context.append({"role": "ai", "content": f"Data Agent: {data_need_result}"})
    print(f"🔍 Data needs identified: {data_need_result[:100]}...\n")
    return context
```

#### Provide Historical Data


```python
def provide_historical_data(history_agent, task: str, context: list) -> list:
    print("🏛️ History Agent: Providing relevant historical data...")
    data_needs = context[-1]["content"]
    data_provision_task = f"Based on the data needs identified, provide relevant historical data or statistics. Data needs: {data_needs}"
    data_provision_result = history_agent.process(data_provision_task, context)
    context.append({"role": "ai", "content": f"History Agent: {data_provision_result}"})
    print(f"📊 Historical data provided: {data_provision_result[:100]}...\n")
    return context
```

#### Analyze Data


```python
def analyze_data(data_agent, task: str, context: list) -> list:
    print("📈 Data Agent: Analyzing historical data...")
    historical_data = context[-1]["content"]
    analysis_task = f"Analyze the historical data provided and describe any trends or insights relevant to the original task. Historical data: {historical_data}"
    analysis_result = data_agent.process(analysis_task, context)
    context.append({"role": "ai", "content": f"Data Agent: {analysis_result}"})
    print(f"💡 Data analysis results: {analysis_result[:100]}...\n")
    return context
```

#### Synthesize Final Answer


```python
def synthesize_final_answer(history_agent, task: str, context: list) -> str:
    print("🏛️ History Agent: Synthesizing final answer...")
    synthesis_task = "Based on all the historical context, data, and analysis, provide a comprehensive answer to the original task."
    final_result = history_agent.process(synthesis_task, context)
    return final_result
```

### HistoryDataCollaborationSystem Class


```python
class HistoryDataCollaborationSystem:
    def __init__(self):
        self.history_agent = Agent("Clio", "History Research Specialist", ["deep knowledge of historical events", "understanding of historical contexts", "identifying historical trends"])
        self.data_agent = Agent("Data", "Data Analysis Expert", ["interpreting numerical data", "statistical analysis", "data visualization description"])

    def solve(self, task: str, timeout: int = 300) -> str:
        print(f"\n👥 Starting collaboration to solve: {task}\n")
        
        start_time = time.time()
        context = []
        
        steps = [
            (research_historical_context, self.history_agent),
            (identify_data_needs, self.data_agent),
            (provide_historical_data, self.history_agent),
            (analyze_data, self.data_agent),
            (synthesize_final_answer, self.history_agent)
        ]
        
        for step_func, agent in steps:
            if time.time() - start_time > timeout:
                return "Operation timed out. The process took too long to complete."
            try:
                result = step_func(agent, task, context)
                if isinstance(result, str):
                    return result  # This is the final answer
                context = result
            except Exception as e:
                return f"Error during collaboration: {str(e)}"
        
        print("\n✅ Collaboration complete. Final answer synthesized.\n")
        return context[-1]["content"]
```

### Example usage


```python
# Create an instance of the collaboration system
collaboration_system = HistoryDataCollaborationSystem()

# Define a complex historical question that requires both historical knowledge and data analysis
question = "How did urbanization rates in Europe compare to those in North America during the Industrial Revolution, and what were the main factors influencing these trends?"

# Solve the question using the collaboration system
result = collaboration_system.solve(question)

# Print the result
print(result)
```

    
    👥 Starting collaboration to solve: How did urbanization rates in Europe compare to those in North America during the Industrial Revolution, and what were the main factors influencing these trends?
    
    🏛️ History Agent: Researching historical context...
    📜 Historical context provided: During the Industrial Revolution, which generally spanned from the late 18th century to the mid-19th...
    
    📊 Data Agent: Identifying data needs based on historical context...
    🔍 Data needs identified: To analyze the urbanization phenomenon during the Industrial Revolution in Europe and North America ...
    
    🏛️ History Agent: Providing relevant historical data...
    📊 Historical data provided: Here is some relevant historical data and statistics that pertain to the urbanization phenomenon dur...
    
    📈 Data Agent: Analyzing historical data...
    💡 Data analysis results: Data Agent: Analyzing the historical data provided reveals several key trends and insights regarding...
    
    🏛️ History Agent: Synthesizing final answer...
    ### Urbanization During the Industrial Revolution: A Comparative Analysis of Europe and North America
    
    The Industrial Revolution, spanning from the late 18th century to the mid-19th century, marked a transformative era characterized by significant changes in economic structures, social dynamics, and urban development. Urbanization emerged as a crucial phenomenon during this period, particularly in Europe and North America, albeit with notable differences in the pace, scale, and nature of urban growth between the two regions.
    
    #### Urbanization in Europe
    
    1. **Origins and Growth**: The Industrial Revolution began in Britain around the 1760s, leading to rapid industrial growth and a shift from agrarian to industrial economies. Cities such as Manchester, Birmingham, and London witnessed explosive population growth. For example, London’s population surged from approximately 1 million in 1801 to 2.5 million by 1851, while Manchester grew from 75,000 to 300,000 during the same period.
    
    2. **Rate of Urbanization**: By 1851, about 50% of Britain's population lived in urban areas, reflecting a significant urbanization trend. The annual growth rates in major cities were substantial, with Manchester experiencing an approximate 4.6% growth rate. This rapid urbanization was driven by the promise of jobs in factories and improved transportation networks, such as railways and canals, which facilitated the movement of goods and people.
    
    3. **Social and Economic Shifts**: The urban workforce transformed dramatically, with roughly 50% of the British workforce engaged in manufacturing by mid-century. This shift led to the emergence of a distinct working class and significant social changes, including increased labor organization and political activism, exemplified by movements like Chartism.
    
    4. **Challenges**: Urbanization brought about severe social challenges, including overcrowding, poor living conditions, and public health crises. For instance, cholera outbreaks in London during the 1840s underscored the dire consequences of rapid urban growth, as many urban areas lacked adequate sanitation and housing.
    
    #### Urbanization in North America
    
    1. **Emergence and Growth**: North America, particularly the United States, began its industrialization later, gaining momentum in the early to mid-19th century. Cities like New York and Chicago became pivotal industrial and urban centers. New York City's population grew from around 60,000 in 1800 to over 1.1 million by 1860, showcasing a remarkable urban expansion.
    
    2. **Urbanization Rates**: By 1860, approximately 20% of the U.S. population lived in urban areas, indicating a lower urbanization level compared to Europe. However, the growth rate of urban populations was high, with New York experiencing an annual growth rate of about 7.6%. This growth was fueled by substantial immigration, primarily from Europe, which contributed significantly to urban demographics.
    
    3. **Economic and Labor Dynamics**: The U.S. saw about 20% of its workforce in manufacturing by 1860, with approximately 110,000 manufacturing establishments, marking a burgeoning industrial sector. The influx of immigrants provided a labor force that was essential for the growth of industries and urban centers, significantly diversifying the population.
    
    4. **Social Issues**: Like their European counterparts, urban areas in the U.S. faced challenges related to overcrowding and inadequate infrastructure. In New York, some neighborhoods had population densities exceeding 135,000 people per square mile. These conditions often led to public health concerns and social unrest, prompting the rise of labor movements advocating for workers’ rights and improved living conditions.
    
    5. **Legislative Responses**: The response to urbanization in the U.S. included the formation of labor unions and early labor movements, such as the National Labor Union established in 1866, which aimed to address workers' rights and working conditions. This reflected a growing awareness of the need for social and economic reforms amidst the rapid urban and industrial expansion.
    
    #### Conclusion
    
    In conclusion, urbanization during the Industrial Revolution was a defining characteristic of both Europe and North America, driven by industrialization, economic opportunities, and transportation advancements. Europe, particularly Britain, experienced an earlier and more advanced stage of urbanization, while North America, fueled by immigration and rapid industrial growth, showed a remarkable increase in urban populations. Despite their differences, both regions faced similar challenges related to overcrowding, public health, and labor rights, leading to social changes and movements advocating for reforms. The complexities of urbanization during this transformative era laid the groundwork for the modern urban landscape, shaping socioeconomic structures and influencing future developments in both regions.
    
