

import torch
import torch.nn.functional as F


def do_normal(input_tensor1, input_layer_weights1, input_layer_bias1, intermediate_weights1, intermediate_biases1):
    # Perform the forward pass through the input layer
    intermediate_output1 = input_tensor1 * input_layer_weights1.reshape(4,5) + input_layer_bias1

    # Apply ReLU activation
    relu_output1 = F.relu(intermediate_output1)

    # Perform the forward pass through the intermediate layer
    final_output1 = torch.matmul(relu_output1, intermediate_weights1) + intermediate_biases1

    # Apply softmax for classification
    softmax_output1 = F.softmax(final_output1, dim=0)

    return softmax_output1


print("changes made on branch main")