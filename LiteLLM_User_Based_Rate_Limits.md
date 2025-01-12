## User Based Rate Limiting Using LiteLLM
- LiteLLM allows you to set budgets per user
- Check if a given user has cross their allocated budget

In this notebook we create a $0.0002 daily budget per user and make completion calls using the litellm budget manager


```python
!pip install litellm uuid
```

## Imports & Env variables


```python
import uuid
import os
os.environ['OPENAI_API_KEY'] = ""
```

## completion() with the budget manager

This code does the following
- Initializes a litellm.BudgetManager()
- Checks if a budget exists for a user
  - Creates a $0.0002 budget if the user does not exisr
- Makes a `litellm.completion()` request only if the user is under their budget


```python
from litellm import BudgetManager, completion

# Initializes a litellm.BudgetManager()
budget_manager = BudgetManager(project_name="liteLLM_project", client_type="hosted") # see https://docs.litellm.ai/docs/budget_manager

user_id = str(uuid.uuid4()) # create a new user id
daily_budget = 0.0002

# Checks if a budget exists for a user
if not budget_manager.is_valid_user(user_id):
    # Creates a $0.0002 budget if the user does not exisr
    print(f"No budget exists for user: {user_id}\n")
    print(f"Creating a budget for user: {user_id}, daily budget ${daily_budget}\n")
    budget_manager.create_budget(total_budget=daily_budget, user=user_id, duration="daily") # duration can be daily, weekly, monthly


# Makes a `litellm.completion()` request only if the user is under their budget
current_spend_for_user = budget_manager.get_current_cost(user=user_id)
budget_for_user = budget_manager.get_total_budget(user_id)
print(f"User: {user_id} has spent ${current_spend_for_user}, budget for user: ${budget_for_user}\n")

if current_spend_for_user <= budget_for_user:
    response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hey, how's it going?"}])
    budget_manager.update_cost(completion_obj=response, user=user_id)
else:
    response = "Sorry - no budget!"

print(response)
```

    No budget exists for user: 29af95f8-c3c6-4c8c-b080-8b2d18d25432
    
    Creating a budget for user: 29af95f8-c3c6-4c8c-b080-8b2d18d25432, daily budget $0.0002
    
    User: 29af95f8-c3c6-4c8c-b080-8b2d18d25432 has spent $0, budget for user: $0.0002
    
    {
      "id": "chatcmpl-7yAUkHQV8xdfldzzZnnnuVU8pl31b",
      "object": "chat.completion",
      "created": 1694574378,
      "model": "gpt-3.5-turbo-0613",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Hello! I'm an AI, so I don't have emotions, but I'm here to assist you. How can I help you today?"
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 14,
        "completion_tokens": 29,
        "total_tokens": 43
      }
    }
    




    {'status': 'success'}



## Make 10 calls to cross the budget per user
- Code fails after user crossed their budget


```python
user_id = "29af95f8-c3c6-4c8c-b080-8b2d18d25432" # set in the previous cell

for _ in range(10):
  # check if a given call can be made
  current_spend_for_user = budget_manager.get_current_cost(user=user_id)
  budget_for_user = budget_manager.get_total_budget(user_id)
  print(f"User: {user_id} has spent ${current_spend_for_user}, budget for user: ${budget_for_user}\n")
  if current_spend_for_user <= budget_for_user:
      response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hey, how's it going?"}])
      budget_manager.update_cost(completion_obj=response, user=user_id)
  else:
      response = "Sorry - no budget!"
      print(f"User: {user_id} has exceeded budget, current spend ${current_spend_for_user}, budget for user: ${budget_for_user}\n")
      break # no more requests

  # print(response)
```

    User: 29af95f8-c3c6-4c8c-b080-8b2d18d25432 has spent $7.9e-05, budget for user: $0.0002
    
    User: 29af95f8-c3c6-4c8c-b080-8b2d18d25432 has spent $0.00015999999999999999, budget for user: $0.0002
    
    User: 29af95f8-c3c6-4c8c-b080-8b2d18d25432 has spent $0.00023899999999999998, budget for user: $0.0002
    
    User: 29af95f8-c3c6-4c8c-b080-8b2d18d25432 has exceeded budget, current spend $0.00023899999999999998, budget for user: $0.0002
    
    
