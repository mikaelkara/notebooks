# Job Candidate Evaluation and Follow Up Agents

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MervinPraison/PraisonAI/blob/main/cookbooks/notebooks/job_candidate_evaluation_agents.ipynb)


# Dependencies


```python
# Install dependencies without output
%pip install praisonai[crewai] > /dev/null
```

# Tools




```python
# This specific code does not leverage any of the tools (may require FileInputTool when the job application/CV input mechanism has been defined in code)
# This program requires updating so as to accomodate one or several input for the Job Application based on which the Agents will manage the evaluation & follow up

```

## YAML Prompt



```python
agent_yaml = """
framework: "crewai"
topic: "Candidate Evaluation and Follow-up Process"
roles:
  email_followup_agent:
    role: "HR Coordinator"
    backstory: |
      An HR professional named Sarah at CrewAI with excellent communication skills, dedicated to crafting personalized, thoughtful emails for job candidates.
    goal: |
      Compose personalized follow-up emails based on the candidate's bio and whether they are being pursued for the job, maintaining a positive and professional tone.
    verbose: true
    allow_delegation: false
    tasks:
      send_followup_email:
        description: |
          Compose personalized follow-up emails for candidates based on their bio and job progress status.
          If the candidate is being pursued, request their availability for a Zoom call; otherwise, send a polite rejection email.
        expected_output: |
          A respectful, personalized email inviting the candidate for a Zoom call or politely declining their application.
  hr_evaluation_agent:
    role: "Senior HR Evaluation Expert"
    backstory: |
      An experienced HR expert known for effectively assessing candidate profiles, scoring them based on skills, experience, and alignment with the job description.
    goal: |
      Evaluate candidates' qualifications against the job description, providing detailed reasoning for each score.
    verbose: true
    allow_delegation: false
    tasks:
      evaluate_candidate:
        description: |
          Evaluate a candidate's bio against the job description, considering skill alignment, experience, cultural fit, and growth potential.
        expected_output: |
          A specific score (1 to 100) and detailed reasoning covering the candidate’s skill match, experience, cultural fit, and growth potential.
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
print(result) # 8/10

```

    [1m[95m [2024-11-01 15:23:25][DEBUG]: == Working Agent: HR Coordinator[00m
    [1m[95m [2024-11-01 15:23:25][INFO]: == Starting Task: Compose personalized follow-up emails for candidates based on their bio and job progress status.
    If the candidate is being pursued, request their availability for a Zoom call; otherwise, send a polite rejection email.
    [00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI now can give a great answer.
    
    Final Answer:
    
    **Email for a Candidate Being Pursued:**
    
    Subject: Exciting Next Steps in Your Application Process with CrewAI
    
    Dear [Candidate's Name],
    
    I hope this message finds you well. I want to extend my gratitude for your interest in the [Job Title] position at CrewAI and for sharing your impressive background with us. Your experience in [specific skill or experience related to the job] truly stands out, and we believe your talents could be a great match for our team.
    
    We are excited to move forward in the interview process and would like to schedule a Zoom call to discuss this opportunity further. Could you please share your availability for the coming week? We are eager to learn more about your experiences and discuss how you can contribute to CrewAI's success.
    
    Thank you once again for considering a career with us. We look forward to the possibility of working together and will do our best to accommodate your schedule.
    
    Warm regards,
    
    Sarah [Your Last Name]  
    HR Coordinator  
    CrewAI  
    [Your Contact Information]  
    
    ---
    
    **Email for a Candidate Not Being Pursued:**
    
    Subject: Update on Your Application with CrewAI
    
    Dear [Candidate's Name],
    
    I hope you are doing well. I want to sincerely thank you for applying for the [Job Title] position at CrewAI and for taking the time to share your qualifications with us. We genuinely appreciate the effort you put into your application.
    
    After careful consideration, we have decided to move forward with other candidates who more closely match the specific skills and experience required for this role. This decision was not easy due to the high caliber of applicants like yourself.
    
    We encourage you to apply for future openings that align with your expertise, as we would love to see your application again. Thank you once again for your interest in joining CrewAI, and we wish you the very best in your job search and future professional endeavors.
    
    Warm regards,
    
    Sarah [Your Last Name]  
    HR Coordinator  
    CrewAI  
    [Your Contact Information][0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-01 15:23:32][DEBUG]: == [HR Coordinator] Task output: **Email for a Candidate Being Pursued:**
    
    Subject: Exciting Next Steps in Your Application Process with CrewAI
    
    Dear [Candidate's Name],
    
    I hope this message finds you well. I want to extend my gratitude for your interest in the [Job Title] position at CrewAI and for sharing your impressive background with us. Your experience in [specific skill or experience related to the job] truly stands out, and we believe your talents could be a great match for our team.
    
    We are excited to move forward in the interview process and would like to schedule a Zoom call to discuss this opportunity further. Could you please share your availability for the coming week? We are eager to learn more about your experiences and discuss how you can contribute to CrewAI's success.
    
    Thank you once again for considering a career with us. We look forward to the possibility of working together and will do our best to accommodate your schedule.
    
    Warm regards,
    
    Sarah [Your Last Name]  
    HR Coordinator  
    CrewAI  
    [Your Contact Information]  
    
    ---
    
    **Email for a Candidate Not Being Pursued:**
    
    Subject: Update on Your Application with CrewAI
    
    Dear [Candidate's Name],
    
    I hope you are doing well. I want to sincerely thank you for applying for the [Job Title] position at CrewAI and for taking the time to share your qualifications with us. We genuinely appreciate the effort you put into your application.
    
    After careful consideration, we have decided to move forward with other candidates who more closely match the specific skills and experience required for this role. This decision was not easy due to the high caliber of applicants like yourself.
    
    We encourage you to apply for future openings that align with your expertise, as we would love to see your application again. Thank you once again for your interest in joining CrewAI, and we wish you the very best in your job search and future professional endeavors.
    
    Warm regards,
    
    Sarah [Your Last Name]  
    HR Coordinator  
    CrewAI  
    [Your Contact Information]
    
    [00m
    [1m[95m [2024-11-01 15:23:32][DEBUG]: == Working Agent: Senior HR Evaluation Expert[00m
    [1m[95m [2024-11-01 15:23:32][INFO]: == Starting Task: Evaluate a candidate's bio against the job description, considering skill alignment, experience, cultural fit, and growth potential.
    [00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI now can give a great answer.   
    Final Answer: My best complete final answer to the task is as follows:  
    
    To evaluate a candidate's qualifications against the job description, we need to consider several key factors: skill match, experience, cultural fit, and growth potential. Here's a detailed assessment based on these criteria:
    
    1. **Skill Match:**  
       The candidate's skills should align closely with the requirements listed in the job description. For instance, if the position requires expertise in data analysis and the candidate has a strong background in statistical software and data interpretation, this would be a significant positive. Conversely, if there are gaps between the candidate's skills and the job requirements, it would result in a lower score in this area.
    
    2. **Experience:**  
       This involves evaluating the candidate's past roles and responsibilities to determine how well their experience aligns with the job at CrewAI. Relevant past experiences, such as working in similar industries or roles, managing projects of similar scale, or having a track record of achievements in the field, would enhance the candidate's score. A lack of relevant experience or a mismatch in the level of responsibility can lower the score.
    
    3. **Cultural Fit:**  
       Assessing cultural fit is about understanding how well the candidate's values, work style, and personality align with CrewAI's company culture. For example, if CrewAI values innovation and the candidate has demonstrated creativity and a penchant for innovative solutions in their previous roles, this would be a positive indicator.
    
    4. **Growth Potential:**  
       This considers the candidate's ability to grow within the company and adapt to future challenges. A candidate who has shown continuous professional development and a willingness to learn new skills would score higher in this area.
    
    Based on these criteria, the candidate can be scored on a scale from 1 to 100. Let's assign a hypothetical score for illustration:
    
    - Skill Match: 85  
    - Experience: 80  
    - Cultural Fit: 90  
    - Growth Potential: 88  
    
    **Overall Score: 86/100**
    
    **Detailed Reasoning:**  
    The candidate demonstrates a strong alignment with the required skills (85/100), indicating they possess most of the technical abilities and knowledge necessary for the role. Their experience (80/100) is relevant, though there may be some areas where they've had less exposure compared to the ideal candidate profile. The cultural fit (90/100) is excellent, suggesting they share similar values and work ethics with CrewAI's team, which is crucial for long-term satisfaction and performance. Finally, their growth potential (88/100) is high, showing a readiness to take on more responsibilities and adapt to new challenges, making them a valuable long-term investment for the company.
    
    These factors together lead to an overall score of 86, indicating a strong candidate likely to succeed in the role at CrewAI.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-01 15:23:48][DEBUG]: == [Senior HR Evaluation Expert] Task output: My best complete final answer to the task is as follows:  
    
    To evaluate a candidate's qualifications against the job description, we need to consider several key factors: skill match, experience, cultural fit, and growth potential. Here's a detailed assessment based on these criteria:
    
    1. **Skill Match:**  
       The candidate's skills should align closely with the requirements listed in the job description. For instance, if the position requires expertise in data analysis and the candidate has a strong background in statistical software and data interpretation, this would be a significant positive. Conversely, if there are gaps between the candidate's skills and the job requirements, it would result in a lower score in this area.
    
    2. **Experience:**  
       This involves evaluating the candidate's past roles and responsibilities to determine how well their experience aligns with the job at CrewAI. Relevant past experiences, such as working in similar industries or roles, managing projects of similar scale, or having a track record of achievements in the field, would enhance the candidate's score. A lack of relevant experience or a mismatch in the level of responsibility can lower the score.
    
    3. **Cultural Fit:**  
       Assessing cultural fit is about understanding how well the candidate's values, work style, and personality align with CrewAI's company culture. For example, if CrewAI values innovation and the candidate has demonstrated creativity and a penchant for innovative solutions in their previous roles, this would be a positive indicator.
    
    4. **Growth Potential:**  
       This considers the candidate's ability to grow within the company and adapt to future challenges. A candidate who has shown continuous professional development and a willingness to learn new skills would score higher in this area.
    
    Based on these criteria, the candidate can be scored on a scale from 1 to 100. Let's assign a hypothetical score for illustration:
    
    - Skill Match: 85  
    - Experience: 80  
    - Cultural Fit: 90  
    - Growth Potential: 88  
    
    **Overall Score: 86/100**
    
    **Detailed Reasoning:**  
    The candidate demonstrates a strong alignment with the required skills (85/100), indicating they possess most of the technical abilities and knowledge necessary for the role. Their experience (80/100) is relevant, though there may be some areas where they've had less exposure compared to the ideal candidate profile. The cultural fit (90/100) is excellent, suggesting they share similar values and work ethics with CrewAI's team, which is crucial for long-term satisfaction and performance. Finally, their growth potential (88/100) is high, showing a readiness to take on more responsibilities and adapt to new challenges, making them a valuable long-term investment for the company.
    
    These factors together lead to an overall score of 86, indicating a strong candidate likely to succeed in the role at CrewAI.
    
    [00m
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">### Task Output ###
My best complete final answer to the task is as follows:  

To evaluate a candidate's qualifications against the job description, we need to consider several key factors: 
skill match, experience, cultural fit, and growth potential. Here's a detailed assessment based on these criteria:

<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>. **Skill Match:**  
   The candidate's skills should align closely with the requirements listed in the job description. For instance, 
if the position requires expertise in data analysis and the candidate has a strong background in statistical 
software and data interpretation, this would be a significant positive. Conversely, if there are gaps between the 
candidate's skills and the job requirements, it would result in a lower score in this area.

<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>. **Experience:**  
   This involves evaluating the candidate's past roles and responsibilities to determine how well their experience 
aligns with the job at CrewAI. Relevant past experiences, such as working in similar industries or roles, managing 
projects of similar scale, or having a track record of achievements in the field, would enhance the candidate's 
score. A lack of relevant experience or a mismatch in the level of responsibility can lower the score.

<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>. **Cultural Fit:**  
   Assessing cultural fit is about understanding how well the candidate's values, work style, and personality align
with CrewAI's company culture. For example, if CrewAI values innovation and the candidate has demonstrated 
creativity and a penchant for innovative solutions in their previous roles, this would be a positive indicator.

<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>. **Growth Potential:**  
   This considers the candidate's ability to grow within the company and adapt to future challenges. A candidate 
who has shown continuous professional development and a willingness to learn new skills would score higher in this 
area.

Based on these criteria, the candidate can be scored on a scale from <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span> to <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span>. Let's assign a hypothetical score 
for illustration:

- Skill Match: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">85</span>  
- Experience: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">80</span>  
- Cultural Fit: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">90</span>  
- Growth Potential: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">88</span>  

**Overall Score: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">86</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span>**

**Detailed Reasoning:**  
The candidate demonstrates a strong alignment with the required skills <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">85</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span><span style="font-weight: bold">)</span>, indicating they possess most of 
the technical abilities and knowledge necessary for the role. Their experience <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">80</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span><span style="font-weight: bold">)</span> is relevant, though there 
may be some areas where they've had less exposure compared to the ideal candidate profile. The cultural fit 
<span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">90</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span><span style="font-weight: bold">)</span> is excellent, suggesting they share similar values and work ethics with CrewAI's team, which is crucial 
for long-term satisfaction and performance. Finally, their growth potential <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">88</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span><span style="font-weight: bold">)</span> is high, showing a readiness 
to take on more responsibilities and adapt to new challenges, making them a valuable long-term investment for the 
company.

These factors together lead to an overall score of <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">86</span>, indicating a strong candidate likely to succeed in the role 
at CrewAI.
</pre>



    None
    


```python

```
