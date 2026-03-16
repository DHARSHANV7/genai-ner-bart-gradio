## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
### Design Steps

### Step 1: Import Required Libraries
First, we import the necessary Python libraries such as os, json, requests, gradio, and dotenv. These libraries help us handle environment variables, send API requests, and create the user interface. We also load the .env file to securely access the Hugging Face API key and model endpoint.

### Step 2: Create Helper Function for API Request
Next, we create a function called get_completion(). This function sends a POST request to the Hugging Face Inference API. It also includes the Authorization header using the API token so that the request can securely access the model.

### Step 3: Implement the Named Entity Recognition (NER) Function
Then we define the NER function which takes the input text from the user. This function calls the get_completion() helper function and sends the text to the NER model endpoint. After receiving the response, it processes the JSON data and extracts the named entities such as person names, locations, and organizations.

### Step 4: Merge Tokens (Optional Step)
Sometimes the model splits words into smaller parts called tokens (for example: “Cal” and “##ifornia”). To make the output easier to read, we implement a merge_tokens() function which combines these tokens into a single word like “California”.

### Step 5: Create the Gradio User Interface
Finally, we build a simple Gradio interface using gr.Interface().

The input is a textbox where users can enter the text.

The output is a highlighted text display that shows the detected entities.

We also add example sentences so users can quickly test the application.

After setting up the interface, we run the application using demo.launch(share=True), which generates a public link so others can access and test the application easily.

### PROGRAM:
```
import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
hf_api_key = os.environ['HF_API_KEY']
API_URL = os.environ['HF_API_NER_BASE']

def get_completion(inputs, parameters=None, ENDPOINT_URL=API_URL):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters:
        data.update({"parameters": parameters})

    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))
    text = response.content.decode("utf-8").strip()

    # Handle extra data safely
    try:
        # Try parsing as normal JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If response contains multiple JSON objects, take the first valid one
        parts = text.split("\n")
        for part in parts:
            try:
                return json.loads(part)
            except Exception:
                continue
        raise ValueError(f"Invalid JSON returned from model: {text}")

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            last = merged_tokens[-1]
            last['word'] += token['word'].replace('##', '')
            last['end'] = token['end']
            last['score'] = (last['score'] + token['score']) / 2
        else:
            merged_tokens.append(token)
    return merged_tokens

def ner(input_text):
    output = get_completion(input_text)
    if not isinstance(output, list):
        raise ValueError(f"Unexpected model output: {output}")
    merged_tokens = merge_tokens(output)
    return {"text": input_text, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(
    fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="NER with dslim/bert-base-NER",
    description="Find named entities using the dslim/bert-base-NER model via Hugging Face Inference API.",
    allow_flagging="never",
    examples=[
        "My name is Dharshan V, I work at DeepLearningAI and live in Chennai.",
        "Dharshan lives in Chennai and works at HuggingFace."
    ]
)

demo.launch(share=True, server_port=int(os.environ.get("PORT3", 7860)))
```
### OUTPUT:

### RESULT:
The Named Entity Recognition (NER) prototype was successfully developed using the fine-tuned BERT model (dslim/bert-base-NER) and deployed through the Gradio interface.
The system efficiently identifies and highlights entities such as names, locations, and organizations from user-provided text input.
