from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the Flan-T5 model and tokenizer
model_name = "google/flan-t5-small"  # You can use "flan-t5-base" if you want a slightly larger model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define the input sentence
input_sentence = "In the midsummer of 1993 in the afternoon of the first day of august a small package was delivered"

# Formulate the task for the model as a question
task = f"Extract the package delivery date and time from this sentence: {input_sentence} "

# Tokenize the input
inputs = tokenizer(task, return_tensors="pt")

# Generate the output
outputs = model.generate(inputs["input_ids"], max_length=50)

# Decode the output
formatted_dates = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the formatted dates
print("Formatted Dates Output: ", formatted_dates)
