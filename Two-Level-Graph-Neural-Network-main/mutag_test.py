from torch_geometric.datasets import TUDataset

cur_dataset = TUDataset(root="../dataset/loaded/", 
                               name="MUTAG")
'''
print(cur_dataset.num_classes)
print(cur_dataset.num_node_features)
print(cur_dataset.num_edge_features)
print(cur_dataset.num_features)

print(cur_dataset[0].x)
print(cur_dataset[0].edge_index)
print(cur_dataset[0].edge_attr)
print(cur_dataset[0].y)

print(cur_dataset[0].x.shape)
print(cur_dataset[0].edge_index.shape)
print(cur_dataset[0].edge_attr.shape)

print(cur_dataset[0].x.dtype)
print(cur_dataset[0].edge_index.dtype)
print(cur_dataset[0].edge_attr.dtype)
'''
#print(cur_dataset.y)

print(cur_dataset.data.y.unique())

print(cur_dataset.y)

#for row in cur_dataset.data.x:
    #print(row)
