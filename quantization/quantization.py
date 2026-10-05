import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import matplotlib.pyplot as plt

login(token="hf_")

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, 
    device_map="auto"           
)

print(f"Num layers: {len(model.model.layers)}")
layer_0 = model.model.layers[0]
wq_module = layer_0.self_attn.q_proj
Wq = wq_module.weight.data

print(f"Wq shape: {Wq.shape}")
print(f"Wq dtype: {Wq.dtype}")
print(f"Wq device: {Wq.device}")

'''
Num layers: 16
Wq shape: torch.Size([2048, 2048])
Wq dtype: torch.bfloat16
Wq device: cpu
'''

int8_scale_factor = 127
Wq_max, _ = torch.max(torch.abs(Wq), dim=1, keepdim=True)
Wq_max_scaled = Wq_max / int8_scale_factor
Wq_scaled = Wq / Wq_max_scaled
Wq_quantized = torch.round(Wq_scaled).to(torch.int8)
Wq_quantized_fp16 = Wq_quantized.to(torch.float16)
Wq_unquantized = Wq_quantized_fp16 * Wq_max_scaled
error = Wq - Wq_unquantized
print(torch.linalg.norm(error) / torch.linalg.norm(Wq))
# tensor(0.0101)

u_err, s_err, v_err = torch.linalg.svd(error)
u_wq, s_wq, v_wq = torch.linalg.svd(Wq.float())

s_err_max, _ = torch.max(s_err, dim=0)
s_wq_max, _ = torch.max(s_wq, dim=0)
s_err_scaled = s_err / s_err_max
s_wq_scaled = s_wq / s_wq_max
s_err_npy = s_err_scaled.detach().cpu().numpy()
s_w_q_npy = s_wq_scaled.detach().cpu().numpy()
plt.plot(s_err_npy, label="error sigma", marker='x', color="blue")
plt.plot(s_w_q_npy, label="Wq sigma", marker='x', color="green")
plt.title("singular values of weight vs quantization error")
plt.xlabel("index")
plt.ylabel("singular value")
plt.legend()
plt.savefig("quantization_error_svd.png", dpi=300, bbox_inches="tight")
