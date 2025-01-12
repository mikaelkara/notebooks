##### Copyright 2024 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Gemini API: Agents and Automatic Function Calling with Barista Bot

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Agents_Function_Calling_Barista_Bot.ipynb"><img src="../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

This notebook shows a practical example of using automatic function calling with the Gemini API's Python SDK to build an agent. You will define some functions that comprise a café's ordering system, connect them to the Gemini API and write an agent loop that interacts with the user to order café drinks.

The guide was inspired by the ReAct-style [Barista bot](https://aistudio.google.com/app/prompts/barista-bot) prompt available through AI Studio.


```
!pip install -qU "google-generativeai>=0.7.2"
```

To run this notebook, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you are running in a different environment, you can store your key in an environment variable. See [Authentication](../quickstarts/Authentication.ipynb) to learn more.


```
from random import randint
from typing import Iterable

import google.generativeai as genai
from google.api_core import retry

from google.colab import userdata
genai.configure(api_key=userdata.get('GOOGLE_API_KEY'))
```

## Define the API

To emulate a café's ordering system, define functions for managing the customer's order: adding, editing, clearing, confirming and fulfilling.

These functions track the customer's order using the global variables `order` (the in-progress order) and `placed_order` (the confirmed order sent to the kitchen). Each of the order-editing functions updates the `order`, and once placed, `order` is copied to `placed_order` and cleared.

In the Python SDK you can pass functions directly to the model constructor, where the SDK will inspect the type signatures and docstrings to define the `tools`. For this reason it's important that you correctly type each of the parameters, give the functions sensible names and detailed docstrings.


```
order = []  # The in-progress order.
placed_order = []  # The confirmed, completed order.

def add_to_order(drink: str, modifiers: Iterable[str] = ()) -> None:
  """Adds the specified drink to the customer's order, including any modifiers."""
  order.append((drink, modifiers))


def get_order() -> Iterable[tuple[str, Iterable[str]]]:
  """Returns the customer's order."""
  return order


def remove_item(n: int) -> str:
  """Remove the nth (one-based) item from the order.

  Returns:
    The item that was removed.
  """
  item, modifiers = order.pop(int(n) - 1)
  return item


def clear_order() -> None:
  """Removes all items from the customer's order."""
  order.clear()


def confirm_order() -> str:
  """Asks the customer if the order is correct.

  Returns:
    The user's free-text response.
  """

  print('Your order:')
  if not order:
    print('  (no items)')

  for drink, modifiers in order:
    print(f'  {drink}')
    if modifiers:
      print(f'   - {", ".join(modifiers)}')

  return input('Is this correct? ')


def place_order() -> int:
  """Submit the order to the kitchen.

  Returns:
    The estimated number of minutes until the order is ready.
  """
  placed_order[:] = order.copy()
  clear_order()

  # TODO(you!): Implement coffee fulfilment.
  return randint(1, 10)
```

## Test the API

With the functions written, test that they work as expected.


```
# Test it out!

clear_order()
add_to_order('Latte', ['Extra shot'])
add_to_order('Tea')
remove_item(2)
add_to_order('Tea', ['Earl Grey', 'hot'])
confirm_order();
```

    Your order:
      Latte
       - Extra shot
      Tea
       - Earl Grey, hot
    

    Is this correct?  yes
    

## Define the prompt

Here you define the full Barista-bot prompt. This prompt contains the café's menu items and modifiers and some instructions.

The instructions include guidance on how functions should be called (e.g. "Always `confirm_order` with the user before calling `place_order`"). You can modify this to add your own interaction style to the bot, for example if you wanted to have the bot repeat every request back before adding to the order, you could provide that instruction here.

The end of the prompt includes some jargon the bot might encounter, and instructions _du jour_ - in this case it notes that the café has run out of soy milk.


```
COFFEE_BOT_PROMPT = """\You are a coffee order taking system and you are restricted to talk only about drinks on the MENU. Do not talk about anything but ordering MENU drinks for the customer, ever.
Your goal is to do place_order after understanding the menu items and any modifiers the customer wants.
Add items to the customer's order with add_to_order, remove specific items with remove_item, and reset the order with clear_order.
To see the contents of the order so far, call get_order (by default this is shown to you, not the user)
Always confirm_order with the user (double-check) before calling place_order. Calling confirm_order will display the order items to the user and returns their response to seeing the list. Their response may contain modifications.
Always verify and respond with drink and modifier names from the MENU before adding them to the order.
If you are unsure a drink or modifier matches those on the MENU, ask a question to clarify or redirect.
You only have the modifiers listed on the menu below: Milk options, espresso shots, caffeine, sweeteners, special requests.
Once the customer has finished ordering items, confirm_order and then place_order.

Hours: Tues, Wed, Thurs, 10am to 2pm
Prices: All drinks are free.

MENU:
Coffee Drinks:
Espresso
Americano
Cold Brew

Coffee Drinks with Milk:
Latte
Cappuccino
Cortado
Macchiato
Mocha
Flat White

Tea Drinks:
English Breakfast Tea
Green Tea
Earl Grey

Tea Drinks with Milk:
Chai Latte
Matcha Latte
London Fog

Other Drinks:
Steamer
Hot Chocolate

Modifiers:
Milk options: Whole, 2%, Oat, Almond, 2% Lactose Free; Default option: whole
Espresso shots: Single, Double, Triple, Quadruple; default: Double
Caffeine: Decaf, Regular; default: Regular
Hot-Iced: Hot, Iced; Default: Hot
Sweeteners (option to add one or more): vanilla sweetener, hazelnut sweetener, caramel sauce, chocolate sauce, sugar free vanilla sweetener
Special requests: any reasonable modification that does not involve items not on the menu, for example: 'extra hot', 'one pump', 'half caff', 'extra foam', etc.

"dirty" means add a shot of espresso to a drink that doesn't usually have it, like "Dirty Chai Latte".
"Regular milk" is the same as 'whole milk'.
"Sweetened" means add some regular sugar, not a sweetener.

Soy milk has run out of stock today, so soy is not available.
"""
```

## Set up the model

In this step you collate the functions into a "system" that is passed as `tools`, instantiate the model and start the chat session.

This block includes two options for interacting with the Gemini API. By toggling `use_sys_inst`, you can switch between using Gemini 1.5 Pro with a system instruction (highest quality but free-tier quota may be insufficient for a long chat session) or Gemini 1.0 Pro (higher free quota but does not support system instructions).

A retriable `send_message` function is also defined to help with low-quota conversations.


```
ordering_system = [add_to_order, get_order, remove_item, clear_order, confirm_order, place_order]

# Toggle this to switch between Gemini 1.5 with a system instruction, or Gemini 1.0 Pro.
use_sys_inst = False

model_name = 'gemini-1.5-flash' if use_sys_inst else 'gemini-1.0-pro'

if use_sys_inst:
  model = genai.GenerativeModel(
      model_name, tools=ordering_system, system_instruction=COFFEE_BOT_PROMPT)
  convo = model.start_chat(enable_automatic_function_calling=True)

else:
  model = genai.GenerativeModel(model_name, tools=ordering_system)
  convo = model.start_chat(
      history=[
          {'role': 'user', 'parts': [COFFEE_BOT_PROMPT]},
          {'role': 'model', 'parts': ['OK I understand. I will do my best!']}
        ],
      enable_automatic_function_calling=True)


@retry.Retry(initial=30)
def send_message(message):
  return convo.send_message(message)


placed_order = []
order = []
```

## Chat with Barista Bot

With the model defined and chat created, all that's left is to connect the user input to the model and display the output, in a loop. This loop continues until an order is placed.

When run in Colab, any fixed-width text originates from your Python code (e.g. `print` calls in the ordering system), regular text comes the Gemini API, and the outlined boxes allow for user input that is rendered with a leading `>`.

Try it out!


```
from IPython.display import display, Markdown

print('Welcome to Barista bot!\n\n')

while not placed_order:
  response = send_message(input('> '))
  display(Markdown(response.text))


print('\n\n')
print('[barista bot session over]')
print()
print('Your order:')
print(f'  {placed_order}\n')
print('- Thanks for using Barista Bot!')
```

    Welcome to Barista bot!
    
    
    

    >  I would like a capuccino with almond milk
    


I have added a Cappuccino with Almond Milk to your order.


    >  do you have stone milk?
    


I'm sorry, we do not have Stone Milk on the menu. Would you like any other type of milk with your cappuccino?


    >  no, that's all
    

    Your order:
      Cappuccino
       - Almond Milk
    

    Is this correct?  yes
    


Ok, I will place the order now.


    >  thanks
    


Your order will be ready in about 10 minutes.


    
    
    
    [barista bot session over]
    
    Your order:
      [('Cappuccino', ['Almond Milk'])]
    
    - Thanks for using Barista Bot!
    

Some things to try:
* Ask about the menu (e.g. "what coffee drinks are available?")
* Use terms that are not specified in the prompt (e.g. "a strong latte" or "an EB tea")
* Change your mind part way through ("uhh cancel the latte sorry")
* Go off-menu ("a babycino")

## See also

This sample app showed you how to integrate a traditional software system (the coffee ordering functions) and an AI agent powered by the Gemini API. This is a simple, practical way to use LLMs that allows for open-ended human language input and output that feels natural, but still keeps a human in the loop to ensure correct operation.

To learn more about how Barista Bot works, check out:

* The [Barista Bot](https://aistudio.google.com/app/prompts/barista-bot) prompt
* [System instructions](../quickstarts/System_instructions.ipynb)
* [Automatic function calling](../quickstarts/Function_calling.ipynb)

