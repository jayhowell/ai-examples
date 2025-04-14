from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "ibm-granite/granite-3.1-3b-a800m-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to("cpu")  # Or "cuda" if you ever move to GPU

# Initialize chat history
chat_history = []

def build_prompt(history, user_input):
    prompt = ""
    for turn in history:
        prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    prompt += f"User: {user_input}\nAssistant:"
    return prompt

print("🧠 Granite Chat is ready. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("👋 Exiting chat.")
        break

    # Build full prompt from chat history
    prompt = build_prompt(chat_history, user_input)

    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and extract just the new assistant reply
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_reply = full_response[len(prompt):].strip()

    # Add to chat history
    chat_history.append({
        "user": user_input,
        "assistant": assistant_reply,
    })

    print(f"Granite: {assistant_reply}\n")

