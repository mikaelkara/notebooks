# Vision Models

Vision models can look at pictures and then tell you what's in them using words. These are called vision-to-text models. They bring together the power of understanding images and language. Using fancy neural networks, these models can look at pictures and describe them in a way that makes sense. They're like a bridge between what you see and what you can read. 

This is super useful for things like making captions for images, helping people who can't see well understand what's in a picture, and organizing information. As these models get even smarter, they're going to make computers even better at understanding and talking about what they "see" in pictures. It's like teaching computers to understand and describe the visual world around us.

<img src="./images/llm_prompt_req_resp.png" height="35%" width="%65">

**Note**: 
To run any of these relevant notebooks you will need an account on Anyscale Endpoints, Anthropic, or OpenAI, depending on what model you elect, along with the respective environment file. Use the template environment files to create respective `.env` file for Anyscale Endpoints, Anthropic, Gemini, or OpenAI.


```python
import warnings
import os

from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
```


```python
BOLD_BEGIN = "\033[1m"
BOLD_END = "\033[0m"
```


```python
system_content = """You are an expert analyzing images and provide accurate descriptions.
You do not make descriptions."""
```

Load the environment


```python
_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings('ignore')
api_key = os.getenv("GOOLE_API_KEY")
MODEL = os.getenv("MODEL")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL)
print(f"Using MODEL={MODEL}")
```

    Using MODEL=gemini-1.5-flash
    

### Create our Google Gemini client


```python
from llm_clnt_factory_api import ClientFactory, get_commpletion

client_factory = ClientFactory()
client_factory.register_client('google', genai.GenerativeModel)
client_type = 'google'
client_kwargs = {"model_name": "gemini-1.5-flash",
                  "generation_config": {"temperature": 0.8,},
                  "system_instruction": None,
                }

client = client_factory.create_client(client_type, **client_kwargs)
```


```python
def display_image(img_path):
    
    from IPython.display import Image, display

    # Display the image in the notebook cell
    display(Image(filename=img_path))
```


```python
image_paths = ["./images/pexels-photo-14690500.jpeg",
               "./images/pexels-photo-313782.jpeg"]

```


```python
user_content = """Describe this picture, landscape, buildings, country, settings, and art style if any dictated. 
                  Identify any signs and indicate what they may suggest."""
```


```python

from PIL import Image
for image_path in image_paths: 
    img = Image.open(image_path)
    response = client.generate_content([user_content, img], stream=True)
    response.resolve()
    display_image(image_path)
    print(f"Image description: {response.text}\n")
```


    
![jpeg](output_11_0.jpg)
    


    Image description: The picture shows a narrow alleyway in a city. The alleyway is lined with buildings on both sides, and the walls are painted white. There are many brightly colored rugs hanging on the walls of the buildings. The rugs are all different shapes and sizes, and they have a variety of patterns.  The alleyway is paved with brick, and there is a sign hanging from one of the buildings that reads “dar baba restaurante cuisine italienne” in Italian. 
    
    The buildings are likely in a country in North Africa or the Middle East, as the architecture and the rugs are typical of those regions.  There is a sign for a restaurant in Italian, suggesting that the country may be a popular tourist destination.
    
    The picture is taken in a realistic style, with a focus on the details of the rugs and the architecture of the alleyway. The colors are rich and vibrant, and the image has a warm, inviting atmosphere.
    
    


    
![jpeg](output_11_2.jpg)
    


    Image description: The image is an aerial photograph of a city skyline.  A bright sunset is in the background with wispy clouds.  The photo was taken from above the city looking down.  The sky is a blend of orange, purple, and blue, and there is a bright light emanating from behind a cloud in the upper center of the frame.  A river flows through the city, its waters are dark blue.  A bridge spans the river. 
    
    The buildings in the foreground are tall, sleek, and modern in style.  The city is dense and packed with skyscrapers and the majority of the buildings are made of glass and concrete. There are a lot of trees and green areas within the city, but they are largely obscured by the buildings.  The overall effect is one of vastness and scale, a sense of a city that is both powerful and beautiful. 
    
    The sign on the building reads “MetLife."  It suggests that the photograph was taken in a large metropolitan city like New York City, where insurance companies have offices.  The photo is likely taken from one of the tall buildings in the cityscape.
    
    

## Modality Model is your eyes & ears! 😜  Feel the wizardy prompt power 🧙‍♀️
