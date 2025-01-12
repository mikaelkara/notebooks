# Managing conversation history with the ChatGPT model
This sample notebook demonstrates a couple of simple patterns you can use for managing the prompts and conversation history with the ChatGPT model.


```python
# if needed, install and/or upgrade to the latest version of the OpenAI Python library
%pip install --upgrade openai
```


```python
# import os module & the OpenAI Python library for calling the OpenAI API
# please make sure you have installed required libraries via pip install -r requirements.txt
import os, dotenv
import openai

dotenv.load_dotenv()
```




    True




```python
# Setting up the deployment name
deployment_name = os.environ['CHAT_COMPLETIONS_MODEL']

# The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
# This is the value of the endpoint for your Azure OpenAI resource
azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT']

# The API key for your Azure OpenAI resource.
api_key = os.environ['AZURE_OPENAI_API_KEY']

# Currently OPENAI API have the following versions available: https://learn.microsoft.com/azure/ai-services/openai/reference
api_version = os.environ['OPENAI_API_VERSION']

client = openai.AzureOpenAI(
  azure_endpoint = azure_endpoint, 
  api_key=api_key,  
  api_version=api_version
)
```

## 1.0 Create the system message for ChatGPT


```python
base_system_message = "You are a helpful assistant."

system_message = f"{base_system_message.strip()}"
print(system_message)
```

    You are a helpful assistant.
    

## 2.0 Define helper functions




```python
import tiktoken

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens
```


```python
# Defining a function to send the prompt to the ChatGPT model
# More info : https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/chatgpt?pivots=programming-language-chat-completions
def send_message(messages, model_name, max_response_tokens=500):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.5,
        max_tokens=max_response_tokens,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content

# Defining a function to print out the conversation in a readable format
def print_conversation(messages):
    for message in messages:
        print(f"[{message['role'].upper()}]")
        print(message['content'])
        print()
```

## 3.0 Start the conversation


```python
# This is the first user message that will be sent to the model. Feel free to update this.
user_message = "I want to write a blog post about the impact of AI on the future of work."
```


```python
# Create the list of messages. role can be either "user" or "assistant" 
messages=[
    {"role": "system", "content": system_message},
    {"role": "user", "name":"example_user", "content": user_message}
]
```


```python
token_count = num_tokens_from_messages(messages)
print(f"Token count: {token_count}")
```

    Token count: 37
    


```python
max_response_tokens = 500

response = send_message(messages, deployment_name, max_response_tokens)
messages.append({"role": "assistant", "content": response})

print_conversation(messages)
```

    [SYSTEM]
    You are a helpful assistant.
    
    [USER]
    I want to write a blog post about the impact of AI on the future of work.
    
    [ASSISTANT]
    Great topic choice! The impact of AI on the future of work is indeed a fascinating and relevant subject. Here's a structure you can follow to write your blog post:
    
    1. Introduction:
       - Start with a captivating opening statement or a thought-provoking question to grab the reader's attention.
       - Provide a brief overview of AI and its growing significance in various industries.
    
    2. The current state of AI in the workplace:
       - Discuss how AI is already being used in different sectors, such as healthcare, finance, manufacturing, and customer service.
       - Highlight some specific examples of AI technologies and their applications.
    
    3. Advantages of AI in the workplace:
       - Explain the potential benefits that AI brings to businesses and employees.
       - Discuss how AI can automate repetitive tasks, enhance productivity, and improve decision-making processes.
       - Mention how AI-powered tools can assist workers in performing their jobs more efficiently.
    
    4. Potential challenges and concerns:
       - Address the concerns surrounding AI's impact on jobs and the workforce.
       - Discuss the fear of job displacement and the need for upskilling and reskilling to adapt to the changing job market.
       - Explore ethical considerations, such as bias in AI algorithms and privacy concerns.
    
    5. Opportunities for collaboration between humans and AI:
       - Emphasize the importance of humans and AI working together as a team.
       - Discuss how AI can augment human capabilities and free up time for more complex and creative tasks.
       - Highlight examples of successful collaborations between humans and AI.
    
    6. Future implications and predictions:
       - Explore the potential long-term effects of AI on the job market and employment landscape.
       - Discuss different viewpoints and predictions from experts regarding the future of work in an AI-driven world.
    
    7. Conclusion:
       - Summarize the key points discussed in the blog post.
       - End with a thought-provoking statement or call-to-action, encouraging readers to stay informed and adapt to the changing work environment.
    
    Remember to support your arguments with relevant statistics, case studies, and credible sources. Good luck with your blog post!
    
    

## 4.0 Continue the conversation

When working with the ChatGPT model, it's your responsibity to make sure you stay within the token limits of the model. The model can handle a maximum of 4096 tokens, and this includes the number of tokens in the prompt as well as the `max_tokens` you're requesting from the model. If you exceed these limits, the model will return an error.

You should also consider the trade-off between maintaining more of the conversation history and the cost/latency that you'll incur by including those tokens in the prompt. Shorter prompts are cheaper and faster. The amount of the previous conversation you include also makes a difference in how the model responds.

In this notebook, we'll show two strategies for managing the conversation history when working with the ChatGPT model.
- Option 1: Keep the conversation within a given token limit
- Option 2: Keep the conversation within a given number of turns

### Option 1: Keep the conversation within a given token limit

`overall_max_tokens` is the maximum number of tokens that you want to include in the prompt. Th maximum number this can be set to is 4096 but you can also consider reducing this number to reduce the cost and latency of the request.


```python
overall_max_tokens = 4096
prompt_max_tokens = overall_max_tokens - max_response_tokens
```

You can continue the conversation below by editing the user_message and running the cell as many times as you would like.


```python
user_message = "The target audience for the blog post should be business leaders working in the tech industry."
#user_message = "Let's talk about generative AI and keep the tone informational but also friendly."
#user_message = "Show me a few more examples"
messages.append({"role": "user", "content": user_message})

token_count = num_tokens_from_messages(messages)
print(f"Token count: {token_count}")

# remove first message while over the token limit
while token_count > prompt_max_tokens:
    messages.pop(0)
    token_count = num_tokens_from_messages(messages)

response = send_message(messages, deployment_name, max_response_tokens)

messages.append({"role": "assistant", "content": response})
print_conversation(messages)
```

    Token count: 865
    [SYSTEM]
    You are a helpful assistant.
    
    [USER]
    I want to write a blog post about the impact of AI on the future of work.
    
    [ASSISTANT]
    Great topic choice! The impact of AI on the future of work is indeed a fascinating and relevant subject. Here's a structure you can follow to write your blog post:
    
    1. Introduction:
       - Start with a captivating opening statement or a thought-provoking question to grab the reader's attention.
       - Provide a brief overview of AI and its growing significance in various industries.
    
    2. The current state of AI in the workplace:
       - Discuss how AI is already being used in different sectors, such as healthcare, finance, manufacturing, and customer service.
       - Highlight some specific examples of AI technologies and their applications.
    
    3. Advantages of AI in the workplace:
       - Explain the potential benefits that AI brings to businesses and employees.
       - Discuss how AI can automate repetitive tasks, enhance productivity, and improve decision-making processes.
       - Mention how AI-powered tools can assist workers in performing their jobs more efficiently.
    
    4. Potential challenges and concerns:
       - Address the concerns surrounding AI's impact on jobs and the workforce.
       - Discuss the fear of job displacement and the need for upskilling and reskilling to adapt to the changing job market.
       - Explore ethical considerations, such as bias in AI algorithms and privacy concerns.
    
    5. Opportunities for collaboration between humans and AI:
       - Emphasize the importance of humans and AI working together as a team.
       - Discuss how AI can augment human capabilities and free up time for more complex and creative tasks.
       - Highlight examples of successful collaborations between humans and AI.
    
    6. Future implications and predictions:
       - Explore the potential long-term effects of AI on the job market and employment landscape.
       - Discuss different viewpoints and predictions from experts regarding the future of work in an AI-driven world.
    
    7. Conclusion:
       - Summarize the key points discussed in the blog post.
       - End with a thought-provoking statement or call-to-action, encouraging readers to stay informed and adapt to the changing work environment.
    
    Remember to support your arguments with relevant statistics, case studies, and credible sources. Good luck with your blog post!
    
    [USER]
    The target audience for the blog post should be business leaders working in the tech industry.
    
    [ASSISTANT]
    That's a great target audience! When writing your blog post for business leaders in the tech industry, you can tailor your content to address their specific interests and concerns. Here are a few suggestions to make your blog post more relevant to this audience:
    
    1. Use industry-specific examples: Incorporate case studies and examples from the tech industry to demonstrate how AI is already impacting businesses in their sector. This will help them relate to the topic and understand its relevance.
    
    2. Highlight the competitive advantage: Emphasize how AI adoption can give businesses a competitive edge in the tech industry. Discuss how AI can improve efficiency, streamline processes, and drive innovation.
    
    3. Address industry-specific challenges: Discuss the unique challenges that tech companies may face when implementing AI, such as data privacy and security concerns, talent acquisition, or regulatory compliance. Offer insights and strategies to overcome these challenges.
    
    4. Discuss emerging trends: Provide insights into the latest trends and advancements in AI technology that are specifically relevant to the tech industry. This can include topics like machine learning, natural language processing, or computer vision.
    
    5. Discuss the future of tech jobs: Explore the potential impact of AI on job roles and skills required in the tech industry. Discuss how business leaders can prepare their workforce for the future by investing in upskilling and reskilling programs.
    
    6. Provide actionable advice: Offer practical tips and recommendations for business leaders on how to successfully integrate AI into their organizations. This can include guidance on selecting the right AI solutions, ensuring ethical AI practices, and fostering a culture of collaboration between humans and AI.
    
    By tailoring your content to address the specific interests and challenges faced by business leaders in the tech industry, you can make your blog post more engaging and valuable to your target audience.
    
    [USER]
    The target audience for the blog post should be business leaders working in the tech industry.
    
    [ASSISTANT]
    That's a great target audience! When writing your blog post for business leaders in the tech industry, you can tailor your content to address their specific interests and concerns. Here are a few suggestions to make your blog post more relevant to this audience:
    
    1. Start with a compelling hook: Begin your blog post with a statement or statistic that highlights the impact of AI on the tech industry. This will grab the attention of business leaders and make them eager to continue reading.
    
    2. Discuss industry-specific applications: Explore how AI is transforming various aspects of the tech industry, such as software development, cybersecurity, data analysis, or product innovation. Provide real-world examples and success stories to showcase the practical applications of AI in these areas.
    
    3. Highlight the competitive advantage: Emphasize how AI adoption can give tech companies a competitive edge. Discuss how AI can enhance product development, improve customer experience, or optimize business operations. Illustrate the benefits of AI in terms of increased efficiency, cost savings, or revenue growth.
    
    4. Address concerns and challenges: Acknowledge the concerns that business leaders in the tech industry may have regarding AI. Discuss potential challenges such as job displacement, ethical considerations, or regulatory compliance. Offer insights and strategies to mitigate these concerns and navigate the evolving landscape.
    
    5. Explore future opportunities: Discuss the future potential of AI in the tech industry. Highlight emerging trends and technologies that business leaders should be aware of, such as machine learning, robotics, or augmented reality. Discuss how these advancements may shape the future of work and present new opportunities for innovation.
    
    6. Provide actionable insights: Offer practical advice and recommendations for business leaders to leverage AI effectively. Discuss strategies for implementing AI solutions, building AI-driven teams, or fostering a culture of innovation. Provide resources and tools that can help them stay updated on the latest AI advancements.
    
    7. Conclusion: Summarize the key takeaways from your blog post and leave the readers with a call-to-action. Encourage business leaders to embrace AI as a transformative force and to proactively explore its potential within their organizations.
    
    By tailoring your content to address the specific interests and challenges faced by business leaders in the tech industry, you can make your blog post more engaging and valuable to your target audience.
    
    

### Option 2: Keep the conversation within a given number of turns


```python
max_messages = 10

overall_max_tokens = 4096
prompt_max_tokens = overall_max_tokens - max_response_tokens
```

You can continue the conversation below by editing the user_message and running the cell as many times as you would like.


```python
user_message = "Make the post about generative AI aimed at business leaders who have some knowledge of the technology."
messages.append({"role": "user", "content": user_message})

while len(messages) > max_messages:
    messages.pop(0)

token_count = num_tokens_from_messages(messages)

while token_count > prompt_max_tokens:
    # remove first message from messages
    messages.pop(0)
    token_count = num_tokens_from_messages(messages)

response = send_message(messages, deployment_name)
messages.append({"role": "assistant", "content": response})
print_conversation(messages)

```

    [SYSTEM]
    You are a helpful assistant.
    
    [USER]
    I want to write a blog post about the impact of AI on the future of work.
    
    [ASSISTANT]
    Great topic choice! The impact of AI on the future of work is indeed a fascinating and relevant subject. Here's a structure you can follow to write your blog post:
    
    1. Introduction:
       - Start with a captivating opening statement or a thought-provoking question to grab the reader's attention.
       - Provide a brief overview of AI and its growing significance in various industries.
    
    2. The current state of AI in the workplace:
       - Discuss how AI is already being used in different sectors, such as healthcare, finance, manufacturing, and customer service.
       - Highlight some specific examples of AI technologies and their applications.
    
    3. Advantages of AI in the workplace:
       - Explain the potential benefits that AI brings to businesses and employees.
       - Discuss how AI can automate repetitive tasks, enhance productivity, and improve decision-making processes.
       - Mention how AI-powered tools can assist workers in performing their jobs more efficiently.
    
    4. Potential challenges and concerns:
       - Address the concerns surrounding AI's impact on jobs and the workforce.
       - Discuss the fear of job displacement and the need for upskilling and reskilling to adapt to the changing job market.
       - Explore ethical considerations, such as bias in AI algorithms and privacy concerns.
    
    5. Opportunities for collaboration between humans and AI:
       - Emphasize the importance of humans and AI working together as a team.
       - Discuss how AI can augment human capabilities and free up time for more complex and creative tasks.
       - Highlight examples of successful collaborations between humans and AI.
    
    6. Future implications and predictions:
       - Explore the potential long-term effects of AI on the job market and employment landscape.
       - Discuss different viewpoints and predictions from experts regarding the future of work in an AI-driven world.
    
    7. Conclusion:
       - Summarize the key points discussed in the blog post.
       - End with a thought-provoking statement or call-to-action, encouraging readers to stay informed and adapt to the changing work environment.
    
    Remember to support your arguments with relevant statistics, case studies, and credible sources. Good luck with your blog post!
    
    [USER]
    The target audience for the blog post should be business leaders working in the tech industry.
    
    [ASSISTANT]
    That's a great target audience! When writing your blog post for business leaders in the tech industry, you can tailor your content to address their specific interests and concerns. Here are a few suggestions to make your blog post more relevant to this audience:
    
    1. Use industry-specific examples: Incorporate case studies and examples from the tech industry to demonstrate how AI is already impacting businesses in their sector. This will help them relate to the topic and understand its relevance.
    
    2. Highlight the competitive advantage: Emphasize how AI adoption can give businesses a competitive edge in the tech industry. Discuss how AI can improve efficiency, streamline processes, and drive innovation.
    
    3. Address industry-specific challenges: Discuss the unique challenges that tech companies may face when implementing AI, such as data privacy and security concerns, talent acquisition, or regulatory compliance. Offer insights and strategies to overcome these challenges.
    
    4. Discuss emerging trends: Provide insights into the latest trends and advancements in AI technology that are specifically relevant to the tech industry. This can include topics like machine learning, natural language processing, or computer vision.
    
    5. Discuss the future of tech jobs: Explore the potential impact of AI on job roles and skills required in the tech industry. Discuss how business leaders can prepare their workforce for the future by investing in upskilling and reskilling programs.
    
    6. Provide actionable advice: Offer practical tips and recommendations for business leaders on how to successfully integrate AI into their organizations. This can include guidance on selecting the right AI solutions, ensuring ethical AI practices, and fostering a culture of collaboration between humans and AI.
    
    By tailoring your content to address the specific interests and challenges faced by business leaders in the tech industry, you can make your blog post more engaging and valuable to your target audience.
    
    [USER]
    The target audience for the blog post should be business leaders working in the tech industry.
    
    [ASSISTANT]
    That's a great target audience! When writing your blog post for business leaders in the tech industry, you can tailor your content to address their specific interests and concerns. Here are a few suggestions to make your blog post more relevant to this audience:
    
    1. Start with a compelling hook: Begin your blog post with a statement or statistic that highlights the impact of AI on the tech industry. This will grab the attention of business leaders and make them eager to continue reading.
    
    2. Discuss industry-specific applications: Explore how AI is transforming various aspects of the tech industry, such as software development, cybersecurity, data analysis, or product innovation. Provide real-world examples and success stories to showcase the practical applications of AI in these areas.
    
    3. Highlight the competitive advantage: Emphasize how AI adoption can give tech companies a competitive edge. Discuss how AI can enhance product development, improve customer experience, or optimize business operations. Illustrate the benefits of AI in terms of increased efficiency, cost savings, or revenue growth.
    
    4. Address concerns and challenges: Acknowledge the concerns that business leaders in the tech industry may have regarding AI. Discuss potential challenges such as job displacement, ethical considerations, or regulatory compliance. Offer insights and strategies to mitigate these concerns and navigate the evolving landscape.
    
    5. Explore future opportunities: Discuss the future potential of AI in the tech industry. Highlight emerging trends and technologies that business leaders should be aware of, such as machine learning, robotics, or augmented reality. Discuss how these advancements may shape the future of work and present new opportunities for innovation.
    
    6. Provide actionable insights: Offer practical advice and recommendations for business leaders to leverage AI effectively. Discuss strategies for implementing AI solutions, building AI-driven teams, or fostering a culture of innovation. Provide resources and tools that can help them stay updated on the latest AI advancements.
    
    7. Conclusion: Summarize the key takeaways from your blog post and leave the readers with a call-to-action. Encourage business leaders to embrace AI as a transformative force and to proactively explore its potential within their organizations.
    
    By tailoring your content to address the specific interests and challenges faced by business leaders in the tech industry, you can make your blog post more engaging and valuable to your target audience.
    
    [USER]
    Make the post about generative AI aimed at business leaders who have some knowledge of the technology.
    
    [ASSISTANT]
    Certainly! Here's an updated outline for your blog post, focusing on generative AI for business leaders with some knowledge of the technology:
    
    1. Introduction:
       - Start with an attention-grabbing statement about the power and potential of generative AI.
       - Briefly explain what generative AI is and its significance in various industries.
    
    2. The current state of generative AI:
       - Discuss the advancements in generative AI technology and its applications in fields like art, design, content creation, and product development.
       - Highlight specific examples of generative AI tools and platforms that have gained traction in the business world.
    
    3. Benefits and opportunities of generative AI:
       - Explain how generative AI can enhance creativity and innovation in business processes.
       - Discuss how generative AI can assist in generating new ideas, designing unique products, or creating personalized customer experiences.
       - Highlight the potential for cost and time savings through automation and optimization with generative AI.
    
    4. Industry-specific applications:
       - Explore how generative AI is being used in specific industries, such as marketing, fashion, gaming, or architecture.
       - Discuss the impact of generative AI on these industries, including improved design workflows, personalized marketing campaigns, or virtual prototyping.
    
    5. Challenges and considerations:
       - Address the challenges and limitations of generative AI, such as ethical concerns, bias in AI-generated content, or the need for human oversight.
       - Discuss strategies for mitigating these challenges and ensuring responsible use of generative AI.
    
    6. Implications for business leaders:
       - Discuss the implications of generative AI on business models and processes.
       - Explore how business leaders can leverage generative AI to gain a competitive edge, drive innovation, and create unique value propositions.
       - Highlight the importance of fostering a culture of experimentation and collaboration with generative AI tools and teams.
    
    7. Looking ahead:
       - Discuss emerging trends and future developments in generative AI.
       - Explore the potential impact of generative AI on job roles, workforce skills, and the need for upskilling or reskilling.
    
    8. Conclusion:
       - Summarize the key points discussed in the blog post.
       - Encourage business leaders to embrace generative AI as a powerful tool for creativity, innovation, and competitive advantage.
       - Provide resources or recommendations for further exploration of generative AI in their respective industries.
    
    Remember to support your arguments with relevant case studies, examples, and credible sources.
    
    [USER]
    Make the post about generative AI aimed at business leaders who have some knowledge of the technology.
    
    [ASSISTANT]
    Title: Unleashing Creativity and Innovation: Exploring the Power of Generative AI for Tech-Savvy Business Leaders
    
    Introduction:
    - Captivating opening: "Imagine a world where machines can generate unique and captivating designs, write compelling content, or even create personalized customer experiences. Welcome to the realm of generative AI."
    - Briefly define generative AI and its growing significance in various industries.
    - Highlight the relevance of generative AI for tech-savvy business leaders who already have some knowledge of the technology.
    
    1. The Evolution of Generative AI:
    - Discuss the advancements in generative AI technology and its journey from basic rule-based systems to sophisticated deep learning models.
    - Highlight key breakthroughs and techniques like GANs (Generative Adversarial Networks) and VAEs (Variational Autoencoders) that enable generative AI to produce realistic and creative outputs.
    
    2. Applications of Generative AI:
    - Explore the diverse range of applications where generative AI is making an impact, such as art, design, content creation, and product development.
    - Provide real-world examples of successful generative AI applications, showcasing how businesses are leveraging this technology to streamline workflows, generate innovative ideas, and enhance customer experiences.
    
    3. Unlocking Creativity and Innovation:
    - Discuss how generative AI can augment human creativity and unlock new possibilities.
    - Highlight how generative AI tools can assist in brainstorming sessions, idea generation, and concept exploration, providing a fresh perspective and pushing the boundaries of innovation.
    - Showcase how generative AI can automate repetitive tasks, freeing up valuable time for creative problem-solving and strategic thinking.
    
    4. Industry-Specific Impacts:
    - Dive into industry-specific impacts of generative AI, focusing on sectors like marketing, fashion, gaming, or architecture.
    - Explore how generative AI is revolutionizing these industries, from personalized marketing campaigns and virtual try-on experiences to procedural content generation and architectural design optimization.
    
    5. Ethical Considerations and Human Oversight:
    - Address the ethical considerations surrounding generative AI, such as potential biases in AI-generated content or the risk of AI replacing human creativity.
    - Discuss the importance of human oversight and collaboration in the generative AI process to ensure responsible and ethical outcomes.
    - Highlight the need for business leaders to establish guidelines and frameworks that prioritize ethical and inclusive generative AI practices.
    
    6. Embracing Generative AI for Competitive Advantage:
    - Discuss the strategic implications of generative AI for businesses and their competitive advantage.
    - Explore
    
    
