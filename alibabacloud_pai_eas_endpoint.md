# Alibaba Cloud PAI EAS

>[Machine Learning Platform for AI of Alibaba Cloud](https://www.alibabacloud.com/help/en/pai) is a machine learning or deep learning engineering platform intended for enterprises and developers. It provides easy-to-use, cost-effective, high-performance, and easy-to-scale plug-ins that can be applied to various industry scenarios. With over 140 built-in optimization algorithms, `Machine Learning Platform for AI` provides whole-process AI engineering capabilities including data labeling (`PAI-iTAG`), model building (`PAI-Designer` and `PAI-DSW`), model training (`PAI-DLC`), compilation optimization, and inference deployment (`PAI-EAS`). `PAI-EAS` supports different types of hardware resources, including CPUs and GPUs, and features high throughput and low latency. It allows you to deploy large-scale complex models with a few clicks and perform elastic scale-ins and scale-outs in real time. It also provides a comprehensive O&M and monitoring system.


```python
##Installing the langchain packages needed to use the integration
%pip install -qU langchain-community
```


```python
from langchain.chains import LLMChain
from langchain_community.llms.pai_eas_endpoint import PaiEasEndpoint
from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)
```

One who wants to use EAS LLMs must set up EAS service first. When the EAS service is launched, `EAS_SERVICE_URL` and `EAS_SERVICE_TOKEN` can be obtained. Users can refer to https://www.alibabacloud.com/help/en/pai/user-guide/service-deployment/ for more information,


```python
import os

os.environ["EAS_SERVICE_URL"] = "Your_EAS_Service_URL"
os.environ["EAS_SERVICE_TOKEN"] = "Your_EAS_Service_Token"
llm = PaiEasEndpoint(
    eas_service_url=os.environ["EAS_SERVICE_URL"],
    eas_service_token=os.environ["EAS_SERVICE_TOKEN"],
)
```


```python
llm_chain = prompt | llm

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
llm_chain.invoke({"question": question})
```




    '  Thank you for asking! However, I must respectfully point out that the question contains an error. Justin Bieber was born in 1994, and the Super Bowl was first played in 1967. Therefore, it is not possible for any NFL team to have won the Super Bowl in the year Justin Bieber was born.\n\nI hope this clarifies things! If you have any other questions, please feel free to ask.'


