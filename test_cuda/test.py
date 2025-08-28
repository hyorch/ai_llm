import torch

print("Torch Version:",torch.__version__)

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor = torch.tensor([1.,2.,3.])
tensor1 = torch.rand(5, 4, 3)

tensor2 = tensor.to(device)

print(tensor2)