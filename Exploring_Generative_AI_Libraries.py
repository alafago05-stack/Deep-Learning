from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Selecting the model. You will be using "facebook/blenderbot-400M-distill" in this example.
model_name = "facebook/blenderbot-400M-distill"

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Define the chat function
def chat_bot1():
    while True:
        # Get user input
        input_text = input("You: ")

        # Exit conditions
        if input_text.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        # Tokenize input and generate response
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(inputs, max_new_tokens=150) 
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Display bot's response
        print("Chatbot:", response)

#another model  "flan-t5-base" model from Google

import sentencepiece
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

### Let's chat with another bot
def chat_bot2():
    while True:
        # Get user input
        input_text = input("You: ")

        # Exit conditions
        if input_text.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        # Tokenize input and generate response
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(inputs, max_new_tokens=150) 
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Display bot's response
        print("Chatbot:", response)

print('pick 1 to chat with chatbot1 or 2 to chat with chatbot2')  
a = input()
if a == 1:
    chat_bot1()
elif a == 2:
    chat_bot2()
    