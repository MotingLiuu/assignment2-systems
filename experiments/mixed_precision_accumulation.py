import torch

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(f"Single precision sum float32: {s.item()}")

s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(f"Half precision sum float16: {s.item()}")

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s = s + torch.tensor(0.01, dtype=torch.float16)
print(f"Mixed precision sum float32 + float16: {s.item()}")
    
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s = s + torch.tensor(0.01, dtype=torch.float32)
print(f"Mixed precision sum float16 + float32: {s.item()}")