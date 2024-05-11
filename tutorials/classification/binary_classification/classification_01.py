import babytorch.nn as b_nn
from babytorch.nn import MSELoss
from babytorch.optim import SGD 
from babytorch import Tensor
from babytorch import Grapher

import torch
import torch.nn as nn
import torch.optim as optim
import cupy as cp

def babytorch_classification(input, target, num_iterations=1000, lr=0.001):
    print("babytorch_classification")
    model = b_nn.Sequential(
            b_nn.Linear(len(input_data[-1]), 4, b_nn.Sigmoid()), 
            b_nn.Linear(4, 1))

    mse = MSELoss()
    optimizer = SGD(model.parameters(), learning_rate=lr)

    losses=[]
    # compute the output of the MLP model
    for i in range(num_iterations):
        model.zero_grad()
        y_pred =  model(input)  
        loss = mse(y_pred, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.data.get()) # get() moves data to CPU
        # print(f"{num_iterations+1} - babytorch loss: {loss.data}")
    return  y_pred, losses

# ----------------PyTorch----------------
def pytorch_classification(input, target, num_iterations=1000, lr=0.001):
    print("pytorch_classification")
    torch_model = nn.Sequential(
        nn.Linear(len(input_data[-1]), 4), # Input layer to hidden layer 1
        nn.Sigmoid(),       # Activation for hidden layer 1
        nn.Linear(4, 1)  # Hidden layer 2 to output layer
    )

    criterion = nn.MSELoss()
    optimizer = optim.SGD(torch_model.parameters(), lr=lr)

    losses = []
    for i in range(num_iterations):
        optimizer.zero_grad()  # Reset gradients
        y_pred = torch_model(input)
        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()  # Update parameters
        losses.append(loss.item())
        # print(f"{i+1} - Pytorch loss: {loss.data}")

    return  y_pred.cpu().detach().numpy() , losses

if __name__ == '__main__':
    input_data = cp.array( [[1, 1,-1], 
                            [1, 1, 1], 
                            [1,-1,-1], 
                            [1,-1, 1], 
                            ])

    # generally the target shape should match the output shape of the model (batch_size, output_size)
    target = cp.array( [ [1], [-1], [-1], [-1] ] ) 

    t_x = Tensor(input_data )
    t_y = Tensor(target )
    p_x = torch.tensor( input_data, dtype=torch.float32 )
    p_y = torch.tensor( target, dtype=torch.float32 )

    pytorch_y_pred, pytorch_losses =  pytorch_classification(p_x, p_y, num_iterations=10000)
    babytorch_y_pred, babytorch_losses = babytorch_classification(input=t_x, target=t_y, num_iterations=10000)

    print("Predictions vs True Labels:")
    print(f"babytorch Prediction |  Pytorch Prediction:| True value ")
    # print(f"{babytorch_y_pred.data=}, {pytorch_y_pred=}, {target=}")
    for t_pred, p_pred, true in zip(babytorch_y_pred.data, pytorch_y_pred ,target ):      
        print(f"  {t_pred[0] :9.4f}         | {p_pred[0]:10.4f}          | {true[0]:2.4f}")

    #  Plot the loss
    G = Grapher()
    # G.show_graph(loss)
    G.plot_loss(pytorch_losses, label='PyTorch')
    G.plot_loss(babytorch_losses, label='babytorch')
    G.show()

