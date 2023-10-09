import torch
#from torch import nn
import torch.nn as nn
import torch.nn.functional as F

class CrmCNN(nn.Module):
    def __init__(self, config):
        super(CrmCNN, self).__init__()
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

        self.convolutional_layer_1 = nn.Conv2d(
            in_channels=1, 
            out_channels=self.kernels,
            dilation=(1, self.dilation),
            kernel_size=(4, self.kernel_len),
            stride=self.conv_stride,
            padding='valid'
        )
        
        self.convolutional_layer_n = nn.Conv2d(
            in_channels=1, 
            out_channels=self.layer_n_kernels,
            dilation=(1, self.dilation),
            kernel_size=(1, self.kernel_len),
            stride=self.conv_stride,
            padding='valid'
        )
        
        # Different Kernel Initialization Methods
        #torch.nn.init.ones_(self.convolutional_layer_1.weight)
        #torch.nn.init.uniform_(self.convolutional_layer_1.weight, a=0.0, b=1.0)
        
        if self.act_fn == 'relu':
            self.activation = nn.ReLU()
        elif self.act_fn == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.GELU()
            
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
        self.fc2 = nn.Linear(100, 1)
        self.fc3 = nn.Linear(100, 1)

        #save the weights of the first layers's kernels
        self.first_layer_kernels = self.convolutional_layer_1.weight.data.cpu().numpy()
        
    
    def feature_hook(self, module, input, output):
        self.features.append(output.detach().cpu().numpy())


    def forward(self, x):
        #print(x.shape)
        x = self.convolutional_layer_1(x)
        #print(x.shape)
        x = self.activation(x)
        #x = torch.exp(x)
        x = self.batch_norm(x)
        x = torch.permute(x, (0, 2, 1, 3))
        #print(x.shape)
        x = self.pooling(x)
        #print(x.shape)
        x = self.dropout(x)

        for i in range(self.num_layers - 1):
            x = self.convolutional_layer_n(x) # change to convolutional_layer_n
            #print(x.shape)
            x = self.activation(x)
            x = self.batch_norm_n(x)
            x = torch.permute(x, (0, 2, 1, 3))
            #print(x.shape)
            x = self.pooling_n(x) # change to pooling_n
            #print(x.shape)
            x = self.dropout(x)
        
        # Attention mechanism
        #attention_logit = torch.matmul(x.squeeze(1), self.attention_weights) + self.attention_bias
        #attention_weights = F.softmax(attention_logit, dim=1)
        #x = x * attention_weights.unsqueeze(-1)
        #x = self.multi_head_attention(x) 
    
        handle = self.fc1.register_forward_hook(self.feature_hook)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        handle.remove() 
   
        return torch.sigmoid(x)
        #return x

    def fc_layer_length(self):
        # Initialize the output length to the input size
        output_length = self.data_len

        # Apply the convolutional and pooling layers
        for i in range(self.num_layers):
            # Apply the convolutional layer
            output_length = (output_length - self.dilation * (self.kernel_len - 1) - 1) // self.conv_stride + 1
            # Apply the max pooling layer
            output_length = (output_length - self.pool_size) // self.pool_stride + 1

        # Return the total number of elements in the output tensor
        return output_length #* self.kernels