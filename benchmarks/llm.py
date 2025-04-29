from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time

model_id = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

inputs = tokenizer("What is AI?", return_tensors="pt").to("cuda")

start = time.time()
outputs = model.generate(**inputs, max_new_tokens=50)
end = time.time()

generated_tokens = outputs.shape[-1]
time_taken = end - start
tokens_per_sec = generated_tokens / time_taken

print(f"LLM Benchmark Results:")
print(f"Generated Tokens: {generated_tokens}")
print(f"Time Taken (s): {time_taken:.4f}")
print(f"Tokens per Second: {tokens_per_sec:.2f}")
