## Generate a mock auto owner's manual using Gemini

Megan O'Keefe, 2024


```
%pip install "google-cloud-aiplatform>=1.38"
```


```
! gcloud config set project YOUR_PROJECT_ID
```


```
! gcloud auth application-default login
```


```
import vertexai
from vertexai.generative_models import ChatSession, GenerativeModel
```


```
# Set to your project and location
PROJECT_ID = "your-project-id"
REGION = "us-central1"  # change region as needed
MODEL = "gemini-1.5-pro"  # change model as needed
```


```
vertexai.init(project=PROJECT_ID, location=REGION)
model = GenerativeModel(MODEL)
chat = model.start_chat()


def get_chat_response(chat: ChatSession, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)
```


```
manual = []
system = "You are an automobile owner's manual generator for the brand Cymbal. The car model is a Cymbal Starlight 2024. Your job is to generate a 30-page owner's manual. you will be given a topic, which represents one chapter of the manual. Generate one page of material in as much detail as possible. Use specific numbers and details. The topic is: "
```


```
topics = [
    "Safety",
    "Child safety",
    "Emergency Assistance",
    "Instrument cluster",
    "Warning lights",
    "Doors, windows, and locks",
    "Adjusting the seats and steering wheel",
    "Towing, cargo, and luggage",
    "Driving procedures with automatic transmission",
    "Lights and windshield wipers",
    "Refueling",
    "Cruise control and automatic support system",
    "Inclement weather driving",
    "Audio and Bluetooth system",
    "Heating and air conditioning",
    "Maintenance and care",
    "Emergencies",
]
```


```
for t in topics:
    print(t)
    p = system + " " + t
    res = get_chat_response(chat, p)
    manual.append(res)
```


```
spl = " ".join(manual).split(" ")
print(len(spl))
```


```
final_text = "".join(manual)
```


```
with open("manual.txt", "w") as m:
    m.write(final_text)
```
