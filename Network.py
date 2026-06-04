import torch              # Core PyTorch tensor operations
import torch.nn as nn     # Neural-network building blocks (layers, activations, etc.)
import math               # Needed for the kaiming_uniform_ initialisation formula


class OWMLayer(nn.Module):
    def __init__(self, input_size, output_size, alpha=0.75):
        super(OWMLayer, self).__init__()         # Initialises nn.Module internal bookkeeping

        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))  # Learnable weight matrix W, shape (out, in); updated by the optimiser
        self.bias   = nn.Parameter(torch.Tensor(output_size))              # Learnable bias vector, one value per output neuron

        # P tracks the input subspace seen across all tasks so far.
        # Starts as identity — meaning no direction has been blocked yet.
        # register_buffer keeps it on the same device as weights but marks it
        # as NOT a learnable parameter, so the optimiser never touches it.
        self.register_buffer('P', torch.eye(input_size))

        self.alpha = alpha          # Regularisation scalar in the RLS denominator; higher = slower P updates
        self.reset_parameters()    # Fill W and b with sensible initial values

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # He/Kaiming init — well-suited for ReLU networks
        nn.init.zeros_(self.bias)                               # Bias starts at zero (standard practice)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)  # Standard affine transform: y = xW^T + b

    def update_projector(self, x):
        """
        RLS update of the OWM projector P (Eq. 2, Zeng et al. 2019):
        P = P - (Pxx^TP) / (alpha + x^TPx)
        """
        x_mean = x.mean(dim=0, keepdim=True).t()   # Average the batch to one representative column vector (input_size, 1)

        Px          = self.P @ x_mean               # How much of x_mean is still reachable through the current projector
        xPx         = x_mean.t() @ Px              # Scalar — measures how much capacity x_mean consumes in the subspace
        denominator = self.alpha + xPx              # alpha prevents division by zero and slows the update down
        numerator   = Px @ Px.t()                  # Outer product — rank-1 matrix that will be subtracted from P

        self.P.sub_(numerator / denominator)        # In-place downdate: shrinks P to block x_mean's direction for future tasks


class OWMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OWMNetwork, self).__init__()

        self.layer1 = OWMLayer(input_dim, hidden_dim)   # First linear layer: 784 → 800
        self.layer2 = OWMLayer(hidden_dim, output_dim)  # Second linear layer: 800 → 10 (one logit per digit class)
        self.relu   = nn.ReLU()                         # Non-linearity applied between the two layers

    def forward(self, x):
        self.input1 = x                                 # Cache raw input to layer1 — needed to update P1 later
        x = self.layer1(x)                              # Apply first linear transformation
        self.h1 = self.relu(x)                          # Apply ReLU; cache result as it's the actual input layer2 sees
        x = self.layer2(self.h1)                        # Apply second linear transformation to produce class logits
        return x                                        # Return raw logits — CrossEntropyLoss handles softmax internally

    def project_gradients(self):
        """
        Projects weight gradients into the null space of P,
        ensuring updates are orthogonal to previously seen inputs.
        """
        with torch.no_grad():                            # We're manually editing .grad; don't let PyTorch track these ops
            if self.layer1.weight.grad is not None:      # Guard: .grad only exists after loss.backward() has run
                self.layer1.weight.grad.copy_(           # Overwrite gradient in-place so the optimiser sees the projected version
                    (self.layer1.P @ self.layer1.weight.grad.t()).t()   # P @ grad^T removes Task-1 directions; second .t() restores shape
                )
            if self.layer2.weight.grad is not None:      # Same projection for layer 2
                self.layer2.weight.grad.copy_(
                    (self.layer2.P @ self.layer2.weight.grad.t()).t()
                )

    def update_all_projectors(self):
        """
        Calls the RLS update for every layer's P matrix using
        the inputs cached during the last forward pass.
        """
        with torch.no_grad():                                    # Projector updates are not part of the computation graph
            self.layer1.update_projector(self.input1)            # Tighten P1 using the raw inputs that just passed through layer1
            self.layer2.update_projector(self.h1)                # Tighten P2 using the hidden states that just passed through layer2


class BaselineNetwork(nn.Module):
    """Plain MLP with no OWM — used as the control to demonstrate catastrophic forgetting."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BaselineNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)   # Standard linear layer: 784 → 800; no projector matrix
        self.layer2 = nn.Linear(hidden_dim, output_dim)  # Standard linear layer: 800 → 10
        self.relu   = nn.ReLU()                          # Same non-linearity as OWMNetwork for a fair comparison

    def forward(self, x):
        x = self.relu(self.layer1(x))   # First layer + ReLU — identical structure to OWMNetwork but no input caching needed
        return self.layer2(x)           # Output logits; no gradient interception, so Task 2 will freely overwrite Task 1 weights
