import torch

#Embbeding
inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your (x^1)
[0.55, 0.87, 0.66], # journey (x^2)
[0.57, 0.85, 0.64], # starts (x^3)
[0.22, 0.58, 0.33], # with (x^4)
[0.77, 0.25, 0.10], # one (x^5)
[0.05, 0.80, 0.55]] # step (x^6)
)

# input 2 Attention Scores
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
print("attn_scores init: ",attn_scores_2)
for i, x_i in enumerate(inputs):
    #print("x_i:",x_i," dot query",query)
    attn_scores_2[i] = torch.dot(x_i, query)
print("attn_scores 2",attn_scores_2)

# Normalización Pesos ATTN
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# Normalización Softmax
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights - Softmax:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# Context Vector
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print("Context Vector: ",context_vec_2)

# All Attention Scores
# attn_scores = torch.empty(6, 6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)
# print("attn_scores: ",attn_scores)

# Attention Scores as matrix product
attn_scores = inputs @ inputs.T
print("attn_scores mat dot: ",attn_scores)

# Normalize -> Weights
attn_weights = torch.softmax(attn_scores, dim=-1)
print("attn_weights: ",attn_weights)
print("All row sums:", attn_weights.sum(dim=-1))

# All context vector
all_context_vecs = attn_weights @ inputs
print("All context vector: ",all_context_vecs)