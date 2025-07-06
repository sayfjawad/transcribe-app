import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set the cache directory explicitly
os.environ['TRANSFORMERS_CACHE'] = '~/.cache/T5'

# Load the T5-large model and tokenizer
model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, local_files_only=True)

# Determine the device and log it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Using GPU for inference.")
else:
    print("Using CPU for inference.")

# Move the model to the selected device
model = model.to(device)

# Define the input sentence
input_sentence = "In the midsummer of 1993 in the afternoon of the first day of August a small package was delivered."

# Formulate the task for the model as a question
task = f"Extract the year from this sentence: {input_sentence}"

# Tokenize the input and move it to the selected device
inputs = tokenizer(task, return_tensors="pt").to(device)

# Generate multiple outputs using beam search
num_sequences = 3  # Set the number of answers to generate
print("Generating output...")
outputs = model.generate(
    inputs["input_ids"],
    max_length=50,
    num_return_sequences=num_sequences,
    num_beams=num_sequences,  # Use beam search
    early_stopping=True       # Stop early if all beams reach the end
)

# Decode and print all the generated outputs
print("Formatted Dates Output:")
for i, output in enumerate(outputs):
    formatted_dates = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Output {i + 1}: {formatted_dates}")
