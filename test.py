from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("./trained_model")
model = GPT2LMHeadModel.from_pretrained("./trained_model")

prompt = " chai ke bina   "

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=50,
    min_new_tokens=10,     # 🔥 FORCE new words
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.5
)


print(tokenizer.decode(outputs[0], skip_special_tokens=True))
