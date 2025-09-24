import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        
        self.layers = torch.nn.Sequential( # Sequential layer
            # Input num_imputs
            # 1st hidden layer - num_imput to 30
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):   # Forward Method for Neural Network
        logits = self.layers(x) # Call layers in order
        return logits

model = NeuralNetwork(50, 3)

print("--------Model-----------\n",model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)
print("Layer 0",model.layers[0].weight.shape)

# Train step1
torch.manual_seed(123)
X = torch.rand((1, 50))
out = model(X)  # Run model.forward(X)
print(out)

# Run for prediction, no grad calculation for backpropagation
with torch.no_grad():
    out = model(X)
print(out)

# Run prediction with probatility
with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print(out)




