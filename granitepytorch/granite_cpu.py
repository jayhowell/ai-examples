from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model name
model_name = "ibm-granite/granite-3.1-3b-a800m-instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model to CPU (no device_map!)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to("cpu")

# Prompt
prompt = "Explain the benefits of OpenShift for enterprise developers."

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

# Generate output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

