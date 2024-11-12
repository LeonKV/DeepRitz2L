import torch
import numpy as np
from kernelDR.problem_definitions.laplace_pacman import LaplaceOnPacmanDomainSingularSolution
from kernelDR.experiments.utilities import OptimizedKernel
from kernelDR.experiments.tkernels import Gaussian

# Initialize the LaplaceOnPacmanDomainSingularSolution with specified parameters
angle = 3 * torch.pi / 2
radius = 1.0
laplace_problem = LaplaceOnPacmanDomainSingularSolution(angle=angle, radius=radius, penalty_parameter=1., device=torch.device('cpu'))

# Generate sample points within the domain as input data X
n_samples = 1000
torch.manual_seed(0)
X = torch.rand((n_samples, 2)) * radius  # Uniform random points within a circle with radius 1

# Calculate the target values using the reference solution in LaplaceOnPacmanDomainSingularSolution
y = laplace_problem.reference_solution(X)

# Define a simple RBF Kernel for the OptimizedKernel model
class RBFKernel:
    def eval(self, X, Z):
        return torch.exp(-torch.cdist(X, Z) ** 2)

# Initialize the OptimizedKernel model with parameters suitable for this problem
dim = 2
kernel = RBFKernel()
model = OptimizedKernel(kernel=Gaussian(), dim=dim, n_epochs=1000, batch_size=20, flag_initialize_diagonal=False, flag_symmetric_A=False)

# Run optimization
model.optimize(X, y, flag_optim_verbose=True)

# Display learned transformation matrix A and objective values during training
print("Final learned matrix A:\n", model.A.detach().numpy())
# print("Objective values during training:", model.list_obj)
