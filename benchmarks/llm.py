from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time
import yaml

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

result = {
    "LLM": {
        "Generated_Tokens": int(generated_tokens),
        "Time_Taken_Sec": round(time_taken, 4),
        "Tokens_per_Sec": round(tokens_per_sec, 2)
    }
}

print(yaml.dump(result))
