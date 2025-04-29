from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time
from utils import update_results

model_id = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

prompt = "Tell me about climate change. " * 10
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

gen_tokens = 512

# Warm-up
_ = model.generate(**inputs, max_new_tokens=gen_tokens)

# First run
start = time.time()
_ = model.generate(**inputs, max_new_tokens=gen_tokens)
first_time = time.time() - start

# Steady state (3x)
times = []
for _ in range(3):
    start = time.time()
    _ = model.generate(**inputs, max_new_tokens=gen_tokens)
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
tokens_per_sec = gen_tokens / avg_time
tokens_per_sec_first = gen_tokens / first_time

update_results("LLM", {
    "Generated_Tokens": gen_tokens,
    "First_Run_Sec": round(first_time, 4),
    "First_Tokens_per_Sec": round(tokens_per_sec_first, 2),
    "Steady_Avg_Sec": round(avg_time, 4),
    "Steady_Tokens_per_Sec": round(tokens_per_sec, 2)
})
