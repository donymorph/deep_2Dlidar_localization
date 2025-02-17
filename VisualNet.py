import torch
from architectures.architectures import SimpleMLP, Conv1DNet, ConvTransformerNet, Conv1DLSTMNet
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

# Step 1: Instantiate the model
model = SimpleMLP(input_size=360)

# Step 2: Create a dummy input tensor
dummy_input = torch.randn(32, 360)  # Batch size of 1, input size of 360

#"""""""""""""Torchviz Network Architecture VISUALIZER"""""""""""""""""



# Step 3: Forward pass to get the graph
#output = model(dummy_input)

# Step 4: Visualize the computational graph
#dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=True,)
#dot.format = 'png'  # Save as PNG image
#dot.render('ConvTransformerNet')  # This will save the image as 'SimpleMLP_Architecture.png'

# To display the graph directly in a notebook (if applicable)
#dot.view()

# pip install graphviz
# pip install torchviz


#"""""""""""""TENSORBOARD Network Architecture VISUALIZER"""""""""""""""""


# Step 3: Initialize TensorBoard SummaryWriter
writer = SummaryWriter("tensorboard_logs/SimpleMLP_Architecture")

# Step 4: Add the model graph to TensorBoard
writer.add_graph(model, dummy_input)

# Step 5: Close the writer
writer.close()

# pip install tensorboard  //////// for installing tensorboard
# tensorboard --logdir=runs /////// running tensorboard and open graph 