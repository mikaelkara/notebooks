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

# Gemini API: Function calling with REST

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/rest/Function_calling_REST.ipynb"><img src="../../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

This notebook provides quick code examples that show you how to get started with function calling using `curl`.

You can run this in Google Colab, or you can copy/paste the `curl` commands into your terminal.

To run this notebook, your API key must be stored it in a Colab Secret named GOOGLE_API_KEY. If you are running in a different environment, you can store your key in an environment variable. See [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) to learn more.


```
import os
from google.colab import userdata

os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

## How function calling works

Function calling lets developers create a description of a function in their code, then pass that description to a language model in a request. The response from the model includes the name of a function that matches the description and the arguments to call it with. Function calling lets you use functions as tools in generative AI applications, and you can define more than one function within a single request. Function calling returns JSON with the name of a function and the arguments to use in your code.

Functions are described using *function declarations*. After you pass a list of
function declarations in a query to a language model, the model returns an
object in an [OpenAPI compatible schema](https://spec.openapis.org/oas/v3.0.3#schema)
format that includes the names of functions and their arguments and tries to
answer the user query with one of the returned functions. The language model
understands the purpose of a function by analyzing its function declaration. The
model doesn't actually call the function. Instead, a developer uses the
[OpenAPI compatible schema](https://spec.openapis.org/oas/v3.0.3#schema) object
in the response to call the function that the model returns.

When you implement function calling, you create one or more *function
declarations*, then add the function declarations to a `tools` object that's
passed to the model. Each function declaration contains information about one
function that includes the following:

* Function name
* Function parameters in an
  [OpenAPI compatible schema](https://spec.openapis.org/oas/v3.0.3#schemawr) format.
  A [select subset](https://ai.google.dev/api/rest/v1beta/Tool#Schema) is
  supported. When using curl, the schema is specified using JSON.
* Function description (optional). For the best results, we recommend that you
  include a description.

This notebook includes curl examples that make REST calls with the
`GenerativeModel` class and its methods.

## Supported models

The following model supports function calling:

* `gemini-pro`

## Function calling cURL samples

When you use cURL, the function and parameter information is included in the
`tools` element. Each function declaration in the `tools` element contains the
function name, its parameters specified using the
[OpenAPI compatible schema](https://spec.openapis.org/oas/v3.0.3#schema), and
a function description. The following samples demonstrate how to use curl
commands with function calling:

### Single-turn curl sample

Single-turn is when you call the language model one time. With function calling,
a single-turn use case might be when you provide the model a natural language
query and a list of functions. In this case, the model uses the function
declaration, which includes the function name, parameters, and description, to
predict which function to call and the arguments to call it with.

The following curl sample is an example of passing in a description of a
function that returns information about where a movie is playing. Several
function declarations are included in the request, such as `find_movies` and
`find_theaters`.


```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=$GOOGLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": {
      "role": "user",
      "parts": {
        "text": "Which theaters in Mountain View show Barbie movie?"
    }
  },
  "tools": [
    {
      "function_declarations": [
        {
          "name": "find_movies",
          "description": "find movie titles currently playing in theaters based on any description, genre, title words, etc.",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
              },
              "description": {
                "type": "string",
                "description": "Any kind of description including category or genre, title words, attributes, etc."
              }
            },
            "required": [
              "description"
            ]
          }
        },
        {
          "name": "find_theaters",
          "description": "find theaters based on location and optionally movie title which are is currently playing in theaters",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
              },
              "movie": {
                "type": "string",
                "description": "Any movie title"
              }
            },
            "required": [
              "location"
            ]
          }
        },
        {
          "name": "get_showtimes",
          "description": "Find the start times for movies playing in a specific theater",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
              },
              "movie": {
                "type": "string",
                "description": "Any movie title"
              },
              "theater": {
                "type": "string",
                "description": "Name of the theater"
              },
              "date": {
                "type": "string",
                "description": "Date for requested showtime"
              }
            },
            "required": [
              "location",
              "movie",
              "theater",
              "date"
            ]
          }
        }
      ]
    }
  ]
}' 2> /dev/null
```

    {
      "candidates": [
        {
          "content": {
            "parts": [
              {
                "functionCall": {
                  "name": "find_theaters",
                  "args": {
                    "movie": "Barbie",
                    "location": "Mountain View, CA"
                  }
                }
              }
            ],
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0,
          "safetyRatings": [
            {
              "category": "HARM_CATEGORY_HATE_SPEECH",
              "probability": "NEGLIGIBLE"
            },
            {
              "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
              "probability": "NEGLIGIBLE"
            },
            {
              "category": "HARM_CATEGORY_HARASSMENT",
              "probability": "NEGLIGIBLE"
            },
            {
              "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
              "probability": "NEGLIGIBLE"
            }
          ]
        }
      ],
      "promptFeedback": {
        "safetyRatings": [
          {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "probability": "NEGLIGIBLE"
          },
          {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "probability": "NEGLIGIBLE"
          },
          {
            "category": "HARM_CATEGORY_HARASSMENT",
            "probability": "NEGLIGIBLE"
          },
          {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "probability": "NEGLIGIBLE"
          }
        ]
      }
    }
    

### Multi-turn curl examples

You can implement a multi-turn function calling scenario by doing the following:

1. Get a function call response by calling the language model. This is the first
   turn.
1. Call the language model using the function call response from the first turn
   and the function response you get from calling that function. This is the
   second turn.

The response from the second turn either summarizes the results to answer your
query in the first turn, or contains a second function call you can use to get
more information for your query.

This topic includes two multi-turn curl examples:

* Curl example that uses a function response from a previous turn
* Curl example that calls a language model multiple times

#### Curl example that uses a response from a previous turn

The following curl sample calls the function and arguments returned by the
previous single-turn example to get a response. The method and parameters
returned by the single-turn example are in this JSON.

```json
"functionCall": {
  "name": "find_theaters",
  "args": {
    "movie": "Barbie",
    "location": "Mountain View, CA"
  }
}
```


```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=$GOOGLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": [{
      "role": "user",
      "parts": [{
        "text": "Which theaters in Mountain View show Barbie movie?"
    }]
  }, {
    "role": "model",
    "parts": [{
      "functionCall": {
        "name": "find_theaters",
        "args": {
          "location": "Mountain View, CA",
          "movie": "Barbie"
        }
      }
    }]
  }, {
    "role": "function",
    "parts": [{
      "functionResponse": {
        "name": "find_theaters",
        "response": {
          "name": "find_theaters",
          "content": {
            "movie": "Barbie",
            "theaters": [{
              "name": "AMC Mountain View 16",
              "address": "2000 W El Camino Real, Mountain View, CA 94040"
            }, {
              "name": "Regal Edwards 14",
              "address": "245 Castro St, Mountain View, CA 94040"
            }]
          }
        }
      }
    }]
  }],
  "tools": [{
    "functionDeclarations": [{
      "name": "find_movies",
      "description": "find movie titles currently playing in theaters based on any description, genre, title words, etc.",
      "parameters": {
        "type": "OBJECT",
        "properties": {
          "location": {
            "type": "STRING",
            "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
          },
          "description": {
            "type": "STRING",
            "description": "Any kind of description including category or genre, title words, attributes, etc."
          }
        },
        "required": ["description"]
      }
    }, {
      "name": "find_theaters",
      "description": "find theaters based on location and optionally movie title which are is currently playing in theaters",
      "parameters": {
        "type": "OBJECT",
        "properties": {
          "location": {
            "type": "STRING",
            "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
          },
          "movie": {
            "type": "STRING",
            "description": "Any movie title"
          }
        },
        "required": ["location"]
      }
    }, {
      "name": "get_showtimes",
      "description": "Find the start times for movies playing in a specific theater",
      "parameters": {
        "type": "OBJECT",
        "properties": {
          "location": {
            "type": "STRING",
            "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
          },
          "movie": {
            "type": "STRING",
            "description": "Any movie title"
          },
          "theater": {
            "type": "STRING",
            "description": "Name of the theater"
          },
          "date": {
            "type": "STRING",
            "description": "Date for requested showtime"
          }
        },
        "required": ["location", "movie", "theater", "date"]
      }
    }]
  }]
}' 2> /dev/null
```

    {
      "candidates": [
        {
          "content": {
            "parts": [
              {
                "text": "OK. I found two theaters in Mountain View that are showing the Barbie movie: AMC Mountain View 16 and Regal Edwards 14."
              }
            ],
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0,
          "safetyRatings": [
            {
              "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
              "probability": "NEGLIGIBLE"
            },
            {
              "category": "HARM_CATEGORY_HATE_SPEECH",
              "probability": "NEGLIGIBLE"
            },
            {
              "category": "HARM_CATEGORY_HARASSMENT",
              "probability": "NEGLIGIBLE"
            },
            {
              "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
              "probability": "NEGLIGIBLE"
            }
          ]
        }
      ]
    }
    

#### Curl example that calls a language model multiple times

The following curl example calls the language model multiple times to call a
function. Each time the model calls the function, it can use a different
function to answer a different user query in the request.


```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=$GOOGLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": [{
      "role": "user",
      "parts": [{
        "text": "Which theaters in Mountain View show Barbie movie?"
    }]
  }, {
    "role": "model",
    "parts": [{
      "functionCall": {
        "name": "find_theaters",
        "args": {
          "location": "Mountain View, CA",
          "movie": "Barbie"
        }
      }
    }]
  }, {
    "role": "function",
    "parts": [{
      "functionResponse": {
        "name": "find_theaters",
        "response": {
          "name": "find_theaters",
          "content": {
            "movie": "Barbie",
            "theaters": [{
              "name": "AMC Mountain View 16",
              "address": "2000 W El Camino Real, Mountain View, CA 94040"
            }, {
              "name": "Regal Edwards 14",
              "address": "245 Castro St, Mountain View, CA 94040"
            }]
          }
        }
      }
    }]
  },
  {
    "role": "model",
    "parts": [{
      "text": " OK. Barbie is showing in two theaters in Mountain View, CA: AMC Mountain View 16 and Regal Edwards 14."
    }]
  },{
    "role": "user",
    "parts": [{
      "text": "Can we recommend some comedy movies on show in Mountain View?"
    }]
  }],
  "tools": [{
    "functionDeclarations": [{
      "name": "find_movies",
      "description": "find movie titles currently playing in theaters based on any description, genre, title words, etc.",
      "parameters": {
        "type": "OBJECT",
        "properties": {
          "location": {
            "type": "STRING",
            "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
          },
          "description": {
            "type": "STRING",
            "description": "Any kind of description including category or genre, title words, attributes, etc."
          }
        },
        "required": ["description"]
      }
    }, {
      "name": "find_theaters",
      "description": "find theaters based on location and optionally movie title which are is currently playing in theaters",
      "parameters": {
        "type": "OBJECT",
        "properties": {
          "location": {
            "type": "STRING",
            "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
          },
          "movie": {
            "type": "STRING",
            "description": "Any movie title"
          }
        },
        "required": ["location"]
      }
    }, {
      "name": "get_showtimes",
      "description": "Find the start times for movies playing in a specific theater",
      "parameters": {
        "type": "OBJECT",
        "properties": {
          "location": {
            "type": "STRING",
            "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
          },
          "movie": {
            "type": "STRING",
            "description": "Any movie title"
          },
          "theater": {
            "type": "STRING",
            "description": "Name of the theater"
          },
          "date": {
            "type": "STRING",
            "description": "Date for requested showtime"
          }
        },
        "required": ["location", "movie", "theater", "date"]
      }
    }]
  }]
}'  2> /dev/null
```

    {
      "candidates": [
        {
          "content": {
            "parts": [
              {
                "functionCall": {
                  "name": "find_movies",
                  "args": {
                    "location": "Mountain View, CA",
                    "description": "comedy"
                  }
                }
              }
            ],
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0,
          "safetyRatings": [
            {
              "category": "HARM_CATEGORY_HARASSMENT",
              "probability": "NEGLIGIBLE"
            },
            {
              "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
              "probability": "NEGLIGIBLE"
            },
            {
              "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
              "probability": "NEGLIGIBLE"
            },
            {
              "category": "HARM_CATEGORY_HATE_SPEECH",
              "probability": "NEGLIGIBLE"
            }
          ]
        }
      ],
      "promptFeedback": {
        "safetyRatings": [
          {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "probability": "NEGLIGIBLE"
          },
          {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "probability": "NEGLIGIBLE"
          },
          {
            "category": "HARM_CATEGORY_HARASSMENT",
            "probability": "NEGLIGIBLE"
          },
          {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "probability": "NEGLIGIBLE"
          }
        ]
      }
    }
    
