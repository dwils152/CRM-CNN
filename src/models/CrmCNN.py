from torch import nn, permute, Tensor, sigmoid

class CrmCNN(nn.Module):
    def __init__(self, config):
        """
        Initialize the CrmCNN model.

        Parameters:
        - config (dict): Configuration dictionary containing model parameters.
        """
        
        super(CrmCNN, self).__init__()
        
        # Add these as kwargs later ...
        self.batch_size = config['batch_size']
        self.conv_stride = config['conv_stride']
        self.pool_stride = config['pool_stride']
        self.pool_size = config['pool_size']
        self.dropout = config['dropout']
        self.kernels = config['kernels']
        self.layer_n_kernels = config['layer_n_kernels']
        self.kernel_len = config['kernel_len']
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.momentum = config['momentum']
        self.dilation = config['dilation']
        self.num_layers = config['num_layers']
        self.act_fn = config['act_fn']
        self.features = []
        self.data_len = config['training_data_len']

        self.conv_layer_1 = nn.Conv2d(
            in_channels=1, 
            out_channels=self.kernels,
            dilation=(1, self.dilation),
            kernel_size=(4, self.kernel_len),
            stride=self.conv_stride,
            padding='valid'
        )
        
        self.conv_layer_n = nn.Conv2d(
            in_channels=1, 
            out_channels=self.layer_n_kernels,
            dilation=(1, self.dilation),
            kernel_size=(1, self.kernel_len),
            stride=self.conv_stride,
            padding='valid'
        )
        
        if self.act_fn == 'relu':
            self.activation = nn.ReLU()
        elif self.act_fn == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif self.act_fn == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError('Activation function not supported')
            
        self.batch_norm = nn.BatchNorm2d(self.kernels)
        self.batch_norm_n = nn.BatchNorm2d(self.layer_n_kernels)

        self.pooling = nn.MaxPool2d(
            kernel_size=(self.kernels, self.pool_size), 
            stride=(1, self.pool_stride)
        )
        
        self.pooling_n = nn.MaxPool2d(
            kernel_size=(self.layer_n_kernels, self.pool_size), 
            stride=(1, self.pool_stride)
        )
        
        self.dropout = nn.Dropout(p=self.dropout)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.fc_layer_length(), 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

        # Save the weights of the first layers's kernels
        self.first_layer_kernels = self.conv_layer_1.weight.data.cpu().numpy()
        
        
    def feature_hook(self, module, input, output: Tensor):
        """
        Append the output features to the feature list.

        Parameters:
        - output (Tensor): The output tensor to be appended to the features.
        """
        
        self.features.append(output.detach().cpu().numpy())


    def forward(self, x: Tensor) -> Tensor:
        """
        Define the forward pass of the CrmCNN model.

        Parameters:
        - x (Tensor): The input tensor.

        Returns:
        - Tensor: The output tensor after passing through the network.
        """
        
        #print(x.shape)
        x = self.conv_layer_1(x)
        #print(x.shape)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = permute(x, (0, 2, 1, 3))
        #print(x.shape)
        x = self.pooling(x)
        #print(x.shape)
        x = self.dropout(x)

        for layer in range(self.num_layers - 1):
            x = self.conv_layer_n(x) # change to convolutional_layer_n
            #print(x.shape)
            x = self.activation(x)
            x = self.batch_norm_n(x)
            x = permute(x, (0, 2, 1, 3))
            #print(x.shape)
            x = self.pooling_n(x)
            #print(x.shape)
            x = self.dropout(x)
        
        handle = self.fc1.register_forward_hook(self.feature_hook)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        handle.remove() 
        return sigmoid(x)


    def fc_layer_length(self) -> int:
        """
        Calculate the length of the flattened tensor before the fully connected layer.

        Returns:
        - int: The total number of elements in the output tensor before the fully connected layer.
        """
        
        output_length = self.data_len

        for i in range(self.num_layers):
            output_length = (output_length - self.dilation * (self.kernel_len - 1) - 1) // self.conv_stride + 1
            output_length = (output_length - self.pool_size) // self.pool_stride + 1

        return output_length #* self.kernels