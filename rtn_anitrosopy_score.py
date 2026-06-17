import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from datasets import load_dataset
import os
from vllm import LLM

model_id = "meta-llama/Llama-3.2-1B"

def anisotropy_score(x: torch.tensor):
  return torch.linalg.matrix_norm(x.to(torch.float32), ord='fro') / torch.linalg.matrix_norm(x.to(torch.float32), ord=2)

def generate_with_hook_hugging_face():
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if len(t.strip()) > 100])[:5000]
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    activations = {}
    handles = []
    def activation_hook(name):
        def hook(model, input, output):
            activations[name] = input[0].detach()
        return hook
   
    for idx, layer in enumerate(model.model.layers):
        handle = layer.self_attn.q_proj.register_forward_hook(activation_hook(f'layer:{idx}'))
        handles.append(handle)
    
    print(f"Input shape: {tokens['input_ids'].shape}")
    with torch.no_grad():
       outputs = model(**tokens)
    print("Forward pass OK")
    
    for k, v in activations.items():
       print(f'layer: {k}, anisotropy_score: {anisotropy_score(v.reshape(-1, v.size(-1)))}')
       
    for handle in handles:
       handle.remove()
       
    return outputs

def generate_with_hook_vLLM():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" # disable vLLM offloading model execution to multiple threads
    # load model

    # CUDA graphs captures a computation graph to be replayed on GPU bypassing PyTorch. 
    # This would mean the hooks would not fire.
    # enforce_eager=True disables CUDA graph capture.

    # vLLM chunks the input prompt during the prefill phase. 
    # this is done so the entire compute is not spent on 1 prefill.
    # the anitrosopy is to be calculated on the entire input activation
    # enable_chunked_prefill disables it
    llm = LLM(model=model_id, enforce_eager=True, enable_chunked_prefill=False) 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if len(t.strip()) > 100])[:5000]
   
    executor = llm.llm_engine.model_executor
    worker = getattr(executor, "driver_worker", None) or getattr(executor, "worker", None)
    pytorch_model = worker.model_runner.model # get the model
    print(pytorch_model) # print the model to peek into the layers
    
    activations = {}
    handles = []
    def my_forward_activation_hook(name):
        def hook(model, input, output):
            if name not in activations:
                print(f'input shape:{input[0].shape}')
                activations[name] = input[0].detach()
        return hook
    
    for idx, layer in enumerate(pytorch_model.model.layers):
        handle = layer.self_attn.qkv_proj.register_forward_hook(my_forward_activation_hook(f'layer-{idx}'))
        handles.append(handle)
        
    print(f"Starting generation...{len(text)}")
    outputs = llm.generate(text)
    
    for k, v in activations.items():
        print(f'layer: {k}, anisotropy_score: {anisotropy_score(v.reshape(-1, v.size(-1)))}')
    
    for handle in handles:
        handle.remove()
    
    return outputs

login(token="hf_")
generate_with_hook_hugging_face()
generate_with_hook_vLLM()
