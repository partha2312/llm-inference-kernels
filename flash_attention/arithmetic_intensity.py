import sys

B = int(sys.argv[1])
H = int(sys.argv[2])
N = int(sys.argv[3])
d = int(sys.argv[4])
d_k = d // H
median_ms = float(sys.argv[5])
block_size = int(sys.argv[6])
flops = 4 * B * H * N * N * d_k
TFlops_per_sec = flops / (median_ms * 1e9)
total_bytes_moved = (3 * B * H * N * d_k + B * N * d) * 2
arithmetic_intensity = flops / total_bytes_moved

print(f"Seq length: {N} Block_Size: {block_size}")
print(f"flops: {flops}")
print(f"TFlops_per_sec: {TFlops_per_sec}")
print(f"Arithmetic Intensity: {arithmetic_intensity}")
