import torch
import math
import triton
import triton.language as tl

def softmax(q: torch.tensor) -> torch.tensor:
    q_max, _ = torch.max(q, axis=q.dim()-1, keepdim=True)
    q_shifted = q - q_max
    q_exp = torch.exp(q_shifted)
    return q_exp / torch.sum(q_exp, axis=q.dim()-1, keepdim=True)

def causal_mask(q: torch.tensor) -> torch.tensor:
    _, _, r, c = q.shape
    i, j = torch.meshgrid(torch.arange(r, device="cuda"), torch.arange(c, device="cuda"), indexing="ij")
    return torch.where(j <= i, q, -float("inf"))

'''
Q, K, V: (B x H x N x d_k) -> fp16
'''
def attention(Q: torch.tensor, K: torch.tensor, V: torch.tensor) -> torch.tensor:
    B, _, N, d_k = Q.shape
    q_kt = Q @ K.transpose(K.dim()-2, K.dim()-1) # (B, H, N, N) -> fp16
    q_kt_scaled = q_kt / math.sqrt(d_k) # (B, H, N, N) -> fp16
    q_kt_scaled_masked = causal_mask(q_kt_scaled) # (B, H, N, N) -> fp16
    q_logits = softmax(q_kt_scaled_masked.to(torch.float32)).to(torch.float16) # (B, H, N, N) -> fp32 -> fp16
    attention = q_logits @ V # (B, H, N, d_k) -> fp16
    return attention.transpose(1, 2).reshape(B, N, -1) # (B, N, d)

@triton.jit
def attention_kernel(Q_ptr, K_ptr, V_ptr, output_ptr,
                     HEAD:tl.constexpr, SEQ_LENGTH:tl.constexpr, d_k:tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
  n_pid = tl.program_id(0)
  h_pid = tl.program_id(1)
  b_pid = tl.program_id(2)

  batch_head_offset = (HEAD * b_pid + h_pid) * (SEQ_LENGTH * d_k)

  q_start = (n_pid * BLOCK_SIZE_N)
  q_row_indices = q_start + tl.arange(0, BLOCK_SIZE_N)
  q_row_offsets = (q_row_indices * d_k) + batch_head_offset
  q_row_mask = q_row_indices < SEQ_LENGTH
  q_col_offsets = tl.arange(0, d_k)
  q_col_mask = q_col_offsets < d_k
  q_offsets = q_row_offsets[:, None] + q_col_offsets[None, :]
  q_masks = q_row_mask[:, None] & q_col_mask[None, :]
  q = tl.load(Q_ptr + q_offsets, mask=q_masks, other=0.0)

  acc = tl.zeros((BLOCK_SIZE_N, d_k), dtype=tl.float32)
  global_max = tl.full([BLOCK_SIZE_N], float("-inf"), dtype=tl.float32)
  global_sum = tl.full([BLOCK_SIZE_N], 0.0, dtype=tl.float32)

  for k in range (0, SEQ_LENGTH, BLOCK_SIZE_N):
    row_indices = k + tl.arange(0, BLOCK_SIZE_N)
    row_offsets = (row_indices * d_k) + batch_head_offset
    row_mask = row_indices < SEQ_LENGTH
    col_offsets = tl.arange(0, d_k)
    col_mask = col_offsets < d_k
    offsets = row_offsets[:, None] + col_offsets[None, :]
    masks = row_mask[:, None] & col_mask[None, :]
    k_mat = tl.load(K_ptr + offsets, mask=masks, other=0.0)
    v = tl.load(V_ptr + offsets, mask=masks, other=0.0)
    q_kt = tl.dot(q, tl.trans(k_mat)) / math.sqrt(d_k)
    off_row = tl.arange(0, BLOCK_SIZE_N)[:, None]
    off_col = tl.arange(0, BLOCK_SIZE_N)[None, :]
    condition = (k + off_col) <= (n_pid * BLOCK_SIZE_N + off_row)
    q_kt = tl.where(condition, q_kt, float("-inf"))

    # get new max
    row_max = tl.max(q_kt, axis=1)
    # update changed max across rows,
    # if not updated the global max will remain same
    local_max = tl.maximum(row_max, global_max)
    # calculate correction
    correction = tl.exp(global_max - local_max)
    # apply correction, if max unchanged multiply by e^0 = 1
    global_sum *= correction
    acc *= correction[:, None]
    # update max
    global_max = local_max
    # handle this stride
    q_kt_shifted = q_kt - global_max[:, None]
    q_kt_shifted_exp = tl.exp(q_kt_shifted)
    # update global sum
    global_sum += tl.sum(q_kt_shifted_exp, axis=1)
    # updated accumulated output
    acc += tl.dot(q_kt_shifted_exp.to(tl.float16), v).to(tl.float32)

  acc /= global_sum[:, None]
  op_batch_offset = b_pid * SEQ_LENGTH * d_k * HEAD
  d = d_k * HEAD
  op_row_start = n_pid * BLOCK_SIZE_N
  op_row_indices = op_row_start + tl.arange(0, BLOCK_SIZE_N)
  op_row_offsets = (op_row_indices * d) + (h_pid * d_k) + op_batch_offset
  op_row_mask = op_row_indices < SEQ_LENGTH
  op_col_offset = tl.arange(0, d_k)
  op_col_mask = op_col_offset < d_k
  op_offsets = op_row_offsets[:, None] + op_col_offset[None, :]
  op_masks = op_row_mask[:, None] & op_col_mask[None, :]
  tl.store(output_ptr + op_offsets, acc.to(tl.float16), mask=op_masks)


SEQ_LENGTH = [1024, 2048, 4096]
BLOCK_SIZES = [32, 64, 128]
for N in SEQ_LENGTH:
  for BLOCK_SIZE_N in BLOCK_SIZES:
    print(f'Seq length: {N}, Block size: {BLOCK_SIZE_N}')
    B, d, H = 2, 4096, 32
    d_k = d // H

    Q = torch.randn((B, H, N, d_k), dtype=torch.float16, device="cuda")
    K = torch.randn((B, H, N, d_k), dtype=torch.float16, device="cuda")
    V = torch.randn((B, H, N, d_k), dtype=torch.float16, device="cuda")
    attention_op = attention(Q, K, V)

    attention_kernel_op = torch.empty((B, N, d), dtype=torch.float16, device="cuda")
    n_elements = N * d_k
    grid = (triton.cdiv(N, BLOCK_SIZE_N), H, B)
    benchmark_fn = lambda grid=grid, Q=Q, K=K, V=V, attention_kernel_op=attention_kernel_op, H=H, N=N, d_k=d_k, BLOCK_SIZE_N=BLOCK_SIZE_N: attention_kernel[grid](Q, K, V, attention_kernel_op, H, N, d_k, BLOCK_SIZE_N)
    torch.cuda.synchronize()

    median_time_ms = triton.testing.do_bench(
        benchmark_fn,
        warmup=25,
        rep=100,
        return_mode="median"
    )

    print(f"Median execution time: {median_time_ms:.3f} ms")

    print(torch.allclose(attention_op, attention_kernel_op, atol=5e-3, rtol=1e-3))
    print(torch.allclose(attention_op, attention_kernel_op, atol=1e-4))
    diff = torch.abs(attention_op - attention_kernel_op)
    print(torch.std(diff))
    diff_4d = diff.view(B, N, H, d_k)
    print(torch.max(diff_4d.mean(dim=(1, 2, 3))))
    print(torch.max(diff_4d.mean(dim=(0, 2, 3))))
    print(torch.max(diff_4d.mean(dim=(0, 1, 3))))