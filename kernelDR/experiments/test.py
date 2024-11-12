import torch
import torch.nn as nn
import torch.optim as optim

from kernelDR.models.model_kernels import FlatKernelModel

# Define the target diffusion matrix
D_target = torch.tensor([[3.0, 0.0], [0.0, 0.5]])

# Generate synthetic data points
x_train = torch.randn(20, 2)  # 2D points in space
u_train = (x_train[:, 0]**2 + x_train[:, 1]**2).unsqueeze(1)

# Compute y_train using the diffusion matrix D
grad_u = 2 * x_train  # Gradient of u(x) = x1^2 + x2^2
y_train = -(grad_u @ D_target @ grad_u.T).diag().unsqueeze(1)  # Target diffusion effect

# Initialize the modified model
model = FlatKernelModel(layer2=True, in_N=2, out_N=1, str_kernel="gaussian")  # Adjust in_N to 2 for 2D input
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Evaluate learned B
learned_B = model.B.detach()
print("Target Matrix D:\n", D_target)
print("Learned Matrix B:\n", learned_B)
print("Difference:\n", D_target - learned_B)
