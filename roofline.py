import torch
import matplotlib.pyplot as plt

tflop_data = torch.tensor([[512, 70.1], [1024, 75.1], [2048, 76.6]], dtype=torch.float32)
x = tflop_data[:, 0]
y = tflop_data[:, 1]
plt.plot(x, y, label="Achieved TFlops/s", marker='x', color="blue")

plt.title("Roofline Analysis — Fused Causal Attention Kernel (A100 PCIe)")
plt.xlabel("Arithmetic Intensity")
plt.ylabel("TFLOP/s (log)")
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.5)
m = 1.935
plt.axline((0, 0), (1, m), color='r', linestyle='-', linewidth=1, label="Memory bandwidth ceiling (1.935 TB/s)")
plt.axhline(y=312, color='y', linestyle='--', linewidth=1, label="Compute bandwidth ceiling (312 TFLOP/s)")
plt.axvline(x=161, color='g', linestyle='--', linewidth=1, label="Ridge point")
plt.legend()
plt.show()