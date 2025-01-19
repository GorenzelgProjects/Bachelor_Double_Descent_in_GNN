import torch.nn as nn
import torch.nn.functional as F
import torch
from resnet18k import make_resnet18k
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    


model = make_resnet18k(1)
#print(model.info())
param = count_parameters(model)
#print(param)
"""
import matplotlib.pyplot as plt
""""""
params_list = []

for i in range(1, 65):
    model = make_resnet18k(i)
    param_count = count_parameters(model)
    params_list.append(param_count)

plt.plot(range(1, 65), params_list)
plt.xlabel('Model Size')
plt.ylabel('Number of Parameters')
plt.title('Number of Parameters vs Model Size')
plt.show()
"""