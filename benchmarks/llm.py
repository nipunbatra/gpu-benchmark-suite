from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time

model_id = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

inputs = tokenizer("What is AI?", return_tensors="pt").to("cuda")
start = time.time()
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
print(f"Time taken: {time.time() - start:.2f}s")
