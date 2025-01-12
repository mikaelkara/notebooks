# Job Profile Matching Agents
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MervinPraison/PraisonAI/blob/main/cookbooks/notebooks/job_profile_matching_agents.ipynb)

## Dependencies


```python
# Install dependencies without output
%pip install langchain_community > /dev/null
%pip install praisonai[crewai] > /dev/null

```

## Tools


```python
from praisonai_tools import BaseTool, FileReadTool, CSVSearchTool
```

## YAML Prompt


```python
agent_yaml = """
framework: "crewai"
topic: "CV Matching to Job Opportunities"
roles:
  cv_reader:
    role: "CV Reader"
    goal: |
      Extract relevant information from the CV, such as skills, experience, and education.
    backstory: |
      With years of experience in HR, you excel at quickly identifying key qualifications in resumes.
    verbose: true
    allow_delegation: false
    tools:
      - "FileReadTool"
    tasks:
      read_cv_task:
        description: |
          Extract relevant information from the given CV. Focus on skills, experience, education, and key achievements.
          Ensure to capture the candidate's professional summary, technical skills, work history, and educational background.

          CV file: /content/cv.md
        expected_output: |
          A structured summary of the CV, including:
          - Professional Summary
          - Technical Skills
          - Work History
          - Education
          - Key Achievements

  matcher:
    role: "Matcher"
    goal: |
      Match the CV to the job opportunities based on skills and experience.
    backstory: |
      A seasoned recruiter, you specialize in finding the perfect fit between candidates and job roles.
    verbose: true
    allow_delegation: false
    tools:
      - "FileReadTool"
      - "CSVSearchTool"
    tasks:
      match_cv_task:
        description: |
          Match the CV to the job opportunities based on skills, experience, and key achievements.
          Evaluate how well the candidate's profile fits each job description, focusing on the alignment of skills, work history, and key achievements with the job requirements.

          Jobs CSV file: /content/jobs.csv
          CV file: /content/cv.md
        expected_output: |
          A ranked list of job opportunities that best match the CV, including:
          - Job Title
          - Match Score (based on skills and experience)
          - Key Matching Points

  job_opportunities_parser:
    role: "Job Opportunities Parser"
    goal: |
      Extract job descriptions from the CSV file, including job title, required skills, and responsibilities.
    backstory: |
      A data analyst who has transitioned into HR, you have a knack for organizing and interpreting job data.
    verbose: true
    allow_delegation: false
    tools:
      - "CSVSearchTool"
dependencies: []

"""
```

## Main


```python
import os
from praisonai import PraisonAI
from google.colab import userdata

# Create a PraisonAI instance with the agent_yaml content
praisonai = PraisonAI(agent_yaml=agent_yaml, tools=[CSVSearchTool, FileReadTool])

# Add OPENAI_API_KEY Secrets to Google Colab on the Left Hand Side 🔑 or Enter Manually Below
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY') or "ENTER OPENAI_API_KEY HERE"
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

# Run PraisonAI
result = praisonai.run()

# Print the result
print(result) # 10/10

```

    [1m[95m [2024-11-03 18:16:01][DEBUG]: == Working Agent: CV Reader[00m
    [1m[95m [2024-11-03 18:16:01][INFO]: == Starting Task: Extract relevant information from the given CV. Focus on skills, experience, education, and key achievements.
    Ensure to capture the candidate's professional summary, technical skills, work history, and educational background.
    
    CV file: /content/cv.md
    [00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mAction: Read a file's content  
    Action Input: {"file_path": "/content/cv.md"}  [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    # John Doe
    
    ## Professional Summary
    Experienced Software Engineer with a strong background in Python, JavaScript, and RESTful APIs. Proven ability to develop and maintain software applications, collaborate with cross-functional teams, and ensure code quality.
    
    ## Technical Skills
    - **Programming Languages**: Python, JavaScript
    - **Web Development**: RESTful APIs, HTML, CSS
    - **Frameworks/Libraries**: Django, Flask, React
    - **Tools**: Git, Docker, Jenkins
    - **Databases**: MySQL, PostgreSQL
    
    ## Work History
    
    ### Senior Software Engineer
    **Tech Innovations Inc., San Francisco, CA**
    *January 2020 - Present*
    - Developed and maintained multiple web applications using Python and JavaScript.
    - Collaborated with cross-functional teams to design and implement RESTful APIs.
    - Ensured code quality through rigorous testing and code reviews.
    - Automated deployment processes using Docker and Jenkins.
    
    ### Software Engineer
    **Innovative Solutions LLC, San Francisco, CA**
    *June 2017 - December 2019*
    - Worked on full-stack web development projects using Python and JavaScript.
    - Built and maintained RESTful APIs for various applications.
    - Participated in Agile development processes, including sprint planning and daily stand-ups.
    - Mentored junior developers and conducted code reviews.
    
    ## Education
    **Bachelor of Science in Computer Science**
    *University of California, Berkeley*
    *Graduated: May 2017*
    
    ## Key Achievements
    - Successfully led a project to migrate a legacy system to a modern web-based platform, resulting in a 30% increase in performance.
    - Recognized as Employee of the Month for outstanding performance and dedication.
    - Implemented a CI/CD pipeline that reduced deployment times by 50%.
    
    ## Contact Information
    - **Email**: john.doe@example.com
    - **Phone**: (123) 456-7890
    - **LinkedIn**: linkedin.com/in/johndoe
    
    [00m
    [32;1m[1;3mThought: I now know the final answer  
    Final Answer: 
    - **Professional Summary**  
    Experienced Software Engineer with a strong background in Python, JavaScript, and RESTful APIs. Proven ability to develop and maintain software applications, collaborate with cross-functional teams, and ensure code quality.
    
    - **Technical Skills**  
      - **Programming Languages**: Python, JavaScript  
      - **Web Development**: RESTful APIs, HTML, CSS  
      - **Frameworks/Libraries**: Django, Flask, React  
      - **Tools**: Git, Docker, Jenkins  
      - **Databases**: MySQL, PostgreSQL  
    
    - **Work History**  
      ### Senior Software Engineer  
      **Tech Innovations Inc., San Francisco, CA**  
      *January 2020 - Present*  
      - Developed and maintained multiple web applications using Python and JavaScript.  
      - Collaborated with cross-functional teams to design and implement RESTful APIs.  
      - Ensured code quality through rigorous testing and code reviews.  
      - Automated deployment processes using Docker and Jenkins.  
    
      ### Software Engineer  
      **Innovative Solutions LLC, San Francisco, CA**  
      *June 2017 - December 2019*  
      - Worked on full-stack web development projects using Python and JavaScript.  
      - Built and maintained RESTful APIs for various applications.  
      - Participated in Agile development processes, including sprint planning and daily stand-ups.  
      - Mentored junior developers and conducted code reviews.  
    
    - **Education**  
      **Bachelor of Science in Computer Science**  
      *University of California, Berkeley*  
      *Graduated: May 2017*  
    
    - **Key Achievements**  
      - Successfully led a project to migrate a legacy system to a modern web-based platform, resulting in a 30% increase in performance.  
      - Recognized as Employee of the Month for outstanding performance and dedication.  
      - Implemented a CI/CD pipeline that reduced deployment times by 50%.  
    
    - **Contact Information**  
      - **Email**: john.doe@example.com  
      - **Phone**: (123) 456-7890  
      - **LinkedIn**: linkedin.com/in/johndoe  [0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-03 18:16:12][DEBUG]: == [CV Reader] Task output: - **Professional Summary**  
    Experienced Software Engineer with a strong background in Python, JavaScript, and RESTful APIs. Proven ability to develop and maintain software applications, collaborate with cross-functional teams, and ensure code quality.
    
    - **Technical Skills**  
      - **Programming Languages**: Python, JavaScript  
      - **Web Development**: RESTful APIs, HTML, CSS  
      - **Frameworks/Libraries**: Django, Flask, React  
      - **Tools**: Git, Docker, Jenkins  
      - **Databases**: MySQL, PostgreSQL  
    
    - **Work History**  
      ### Senior Software Engineer  
      **Tech Innovations Inc., San Francisco, CA**  
      *January 2020 - Present*  
      - Developed and maintained multiple web applications using Python and JavaScript.  
      - Collaborated with cross-functional teams to design and implement RESTful APIs.  
      - Ensured code quality through rigorous testing and code reviews.  
      - Automated deployment processes using Docker and Jenkins.  
    
      ### Software Engineer  
      **Innovative Solutions LLC, San Francisco, CA**  
      *June 2017 - December 2019*  
      - Worked on full-stack web development projects using Python and JavaScript.  
      - Built and maintained RESTful APIs for various applications.  
      - Participated in Agile development processes, including sprint planning and daily stand-ups.  
      - Mentored junior developers and conducted code reviews.  
    
    - **Education**  
      **Bachelor of Science in Computer Science**  
      *University of California, Berkeley*  
      *Graduated: May 2017*  
    
    - **Key Achievements**  
      - Successfully led a project to migrate a legacy system to a modern web-based platform, resulting in a 30% increase in performance.  
      - Recognized as Employee of the Month for outstanding performance and dedication.  
      - Implemented a CI/CD pipeline that reduced deployment times by 50%.  
    
    - **Contact Information**  
      - **Email**: john.doe@example.com  
      - **Phone**: (123) 456-7890  
      - **LinkedIn**: linkedin.com/in/johndoe
    
    [00m
    [1m[95m [2024-11-03 18:16:12][DEBUG]: == Working Agent: Matcher[00m
    [1m[95m [2024-11-03 18:16:12][INFO]: == Starting Task: Match the CV to the job opportunities based on skills, experience, and key achievements.
    Evaluate how well the candidate's profile fits each job description, focusing on the alignment of skills, work history, and key achievements with the job requirements.
    
    Jobs CSV file: /content/jobs.csv
    CV file: /content/cv.md
    [00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI need to gather the content of the job opportunities from the CSV file and the candidate's CV to evaluate how well the candidate's profile fits each job description. 
    
    Action: Read a file's content  
    Action Input: {"file_path": "/content/cv.md"}  
    [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    # John Doe
    
    ## Professional Summary
    Experienced Software Engineer with a strong background in Python, JavaScript, and RESTful APIs. Proven ability to develop and maintain software applications, collaborate with cross-functional teams, and ensure code quality.
    
    ## Technical Skills
    - **Programming Languages**: Python, JavaScript
    - **Web Development**: RESTful APIs, HTML, CSS
    - **Frameworks/Libraries**: Django, Flask, React
    - **Tools**: Git, Docker, Jenkins
    - **Databases**: MySQL, PostgreSQL
    
    ## Work History
    
    ### Senior Software Engineer
    **Tech Innovations Inc., San Francisco, CA**
    *January 2020 - Present*
    - Developed and maintained multiple web applications using Python and JavaScript.
    - Collaborated with cross-functional teams to design and implement RESTful APIs.
    - Ensured code quality through rigorous testing and code reviews.
    - Automated deployment processes using Docker and Jenkins.
    
    ### Software Engineer
    **Innovative Solutions LLC, San Francisco, CA**
    *June 2017 - December 2019*
    - Worked on full-stack web development projects using Python and JavaScript.
    - Built and maintained RESTful APIs for various applications.
    - Participated in Agile development processes, including sprint planning and daily stand-ups.
    - Mentored junior developers and conducted code reviews.
    
    ## Education
    **Bachelor of Science in Computer Science**
    *University of California, Berkeley*
    *Graduated: May 2017*
    
    ## Key Achievements
    - Successfully led a project to migrate a legacy system to a modern web-based platform, resulting in a 30% increase in performance.
    - Recognized as Employee of the Month for outstanding performance and dedication.
    - Implemented a CI/CD pipeline that reduced deployment times by 50%.
    
    ## Contact Information
    - **Email**: john.doe@example.com
    - **Phone**: (123) 456-7890
    - **LinkedIn**: linkedin.com/in/johndoe
    
    [00m
    [32;1m[1;3mThought:
    I need to gather the job opportunities from the CSV file to evaluate how well the candidate's profile fits each job description. 
    
    Action: Read a file's content  
    Action Input: {"file_path": "/content/jobs.csv"}  
    [0m

    /usr/local/lib/python3.10/dist-packages/pydantic/main.py:214: UserWarning: A custom validator is returning a value other than `self`.
    Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.
    See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.
      warnings.warn(
    

    [95m 
    
    Job Title,Required Skills,Key Responsibilities,Company Name,Company Location
    Software Engineer,"Python,JavaScript,RESTful APIs","Develop and maintain software applications, collaborate with cross-functional teams, ensure code quality",TechCorp,San Francisco
    Data Scientist,"Python,R,Machine Learning,Data Analysis","Analyze large datasets to extract insights, build predictive models, collaborate with stakeholders",DataTech,New York
    Project Manager,"Agile, Scrum,Leadership,Communication","Manage project timelines, coordinate with team members, ensure project goals are met",ProjectPlus,Chicago
    DevOps Engineer,"Docker,Kubernetes,CI/CD","Implement and manage CI/CD pipelines, automate deployment processes, monitor system performance",DevOpsCo,Seattle
    Marketing Specialist,"SEO,Content Creation,Social Media","Develop and execute marketing campaigns, create content for various platforms, analyze campaign performance",MarketGenius,Los Angeles
    
    [00m
    [32;1m[1;3mThought: Now that I have both the CV and job descriptions, I need to evaluate how well the candidate's skills and experience match each job opportunity. I will analyze the required skills and responsibilities for each job in relation to the candidate's qualifications. 
    
    1. **Software Engineer** at TechCorp
       - **Required Skills**: Python, JavaScript, RESTful APIs
       - **Key Responsibilities**: Develop and maintain software applications, collaborate with cross-functional teams, ensure code quality
       - **Match Evaluation**:
         - The candidate has strong experience in Python and JavaScript, having developed and maintained web applications using these technologies.
         - The candidate has also collaborated with cross-functional teams and ensured code quality, which aligns with the job responsibilities.
    
    2. **Data Scientist** at DataTech
       - **Required Skills**: Python, R, Machine Learning, Data Analysis
       - **Key Responsibilities**: Analyze large datasets to extract insights, build predictive models, collaborate with stakeholders
       - **Match Evaluation**:
         - The candidate has Python experience but does not have skills in R, Machine Learning, or Data Analysis, which are crucial for this role. Therefore, the match is low.
    
    3. **Project Manager** at ProjectPlus
       - **Required Skills**: Agile, Scrum, Leadership, Communication
       - **Key Responsibilities**: Manage project timelines, coordinate with team members, ensure project goals are met
       - **Match Evaluation**:
         - The candidate does not have direct project management experience or specific skills in Agile or Scrum, making this role a poor fit.
    
    4. **DevOps Engineer** at DevOpsCo
       - **Required Skills**: Docker, Kubernetes, CI/CD
       - **Key Responsibilities**: Implement and manage CI/CD pipelines, automate deployment processes, monitor system performance
       - **Match Evaluation**:
         - The candidate has experience with Docker and has implemented a CI/CD pipeline, which aligns well with this position. However, there is no mention of Kubernetes experience, which may be a requirement.
    
    5. **Marketing Specialist** at MarketGenius
       - **Required Skills**: SEO, Content Creation, Social Media
       - **Key Responsibilities**: Develop and execute marketing campaigns, create content for various platforms, analyze campaign performance
       - **Match Evaluation**:
         - The candidate’s background is unrelated to marketing, thus making this position an unsuitable match.
    
    Based on this analysis, I will now assign a match score to each job based on the alignment of the candidate's skills and experiences to the job requirements.
    
    Thought: I will now compile the ranked list of job opportunities that best match the CV, including job titles, match scores, and key matching points.
    
    Final Answer:
    1. **Job Title**: Software Engineer  
       **Match Score**: 90%  
       **Key Matching Points**: Strong experience in Python and JavaScript, developed and maintained applications, collaborated with cross-functional teams, ensured code quality.
    
    2. **Job Title**: DevOps Engineer  
       **Match Score**: 75%  
       **Key Matching Points**: Experience with Docker, implemented CI/CD pipeline, automated deployment processes. (Lack of Kubernetes experience noted)
    
    3. **Job Title**: Data Scientist  
       **Match Score**: 30%  
       **Key Matching Points**: Proficiency in Python, but lack of skills in R, Machine Learning, and Data Analysis.
    
    4. **Job Title**: Project Manager  
       **Match Score**: 20%  
       **Key Matching Points**: No direct project management experience or Agile/Scrum skills.
    
    5. **Job Title**: Marketing Specialist  
       **Match Score**: 0%  
       **Key Matching Points**: Background unrelated to marketing.[0m
    
    [1m> Finished chain.[0m
    [1m[92m [2024-11-03 18:16:23][DEBUG]: == [Matcher] Task output: 1. **Job Title**: Software Engineer  
       **Match Score**: 90%  
       **Key Matching Points**: Strong experience in Python and JavaScript, developed and maintained applications, collaborated with cross-functional teams, ensured code quality.
    
    2. **Job Title**: DevOps Engineer  
       **Match Score**: 75%  
       **Key Matching Points**: Experience with Docker, implemented CI/CD pipeline, automated deployment processes. (Lack of Kubernetes experience noted)
    
    3. **Job Title**: Data Scientist  
       **Match Score**: 30%  
       **Key Matching Points**: Proficiency in Python, but lack of skills in R, Machine Learning, and Data Analysis.
    
    4. **Job Title**: Project Manager  
       **Match Score**: 20%  
       **Key Matching Points**: No direct project management experience or Agile/Scrum skills.
    
    5. **Job Title**: Marketing Specialist  
       **Match Score**: 0%  
       **Key Matching Points**: Background unrelated to marketing.
    
    [00m
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">### Task Output ###
<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>. **Job Title**: Software Engineer  
   **Match Score**: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">90</span>%  
   **Key Matching Points**: Strong experience in Python and JavaScript, developed and maintained applications, 
collaborated with cross-functional teams, ensured code quality.

<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>. **Job Title**: DevOps Engineer  
   **Match Score**: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">75</span>%  
   **Key Matching Points**: Experience with Docker, implemented CI/CD pipeline, automated deployment processes. 
<span style="font-weight: bold">(</span>Lack of Kubernetes experience noted<span style="font-weight: bold">)</span>

<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>. **Job Title**: Data Scientist  
   **Match Score**: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">30</span>%  
   **Key Matching Points**: Proficiency in Python, but lack of skills in R, Machine Learning, and Data Analysis.

<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>. **Job Title**: Project Manager  
   **Match Score**: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">20</span>%  
   **Key Matching Points**: No direct project management experience or Agile/Scrum skills.

<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>. **Job Title**: Marketing Specialist  
   **Match Score**: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>%  
   **Key Matching Points**: Background unrelated to marketing.
</pre>



    None
    


```python

```
