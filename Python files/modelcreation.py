import pandas as pd
relevant_df=pd.read_csv('D:\\Projects\\ML\\glioma+grading+clinical+and+mutation\\dataset\\relevant_df')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(relevant_df,relevant_df.Grade,test_size=0.25,train_size=0.75, random_state=None, shuffle=True, stratify=None)
X_test=X_test.drop('Grade',axis=1)
X_train=X_train.drop('Grade',axis=1)

#reshaping 1d array to 2d array error: Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
y_train= y_train.values.reshape(1,-1)
y_test= y_test.values.reshape(1,-1)

#scaling the data
from sklearn.preprocessing import MinMaxScaler

mms=MinMaxScaler()
X_train_norm=mms.fit_transform(X_train)
X_test_norm=mms.fit_transform(X_test)
y_train_norm=mms.fit_transform(y_train)
y_test_norm=mms.fit_transform(y_test)

import torch #create tensors to store numerical values(raw data, weights, bias)
import torch.nn as nn #make weight and bias part of NN
import torch.nn.functional as F #gives activation function
import torch.optim as optim

import seaborn as sns

class KNN(nn.Module): #creating a new NN means creating a new class, KNN inherits from pytorch class called module
    def __init__(self, input_size, hidden_size, output_size):
        super(KNN,self).__init__()#iniitialize method for parent class nn.Module
        #Create the input layer
        self.layers=nn.ModuleList([nn.Linear(input_size,hidden_size[0])])
        #create the hidden layer
        for i in range(len(hidden_size)-1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        #create output layer
        self.layers.append(nn.Linear(hidden_size[-1],output_size))

        self.relu=nn.ReLU()

        self.float() #error: RuntimeError: Found dtype Double but expected Float (reduces efficieny because it used double)


    def forward(self,input): 
        for layer in self.layers:
            input=input.to(torch.float32)
            input=layer(input)
            input=self.relu(input)
        return input
    
#converting data set to tensor
X_test_norm_tensor=torch.tensor((X_test_norm),requires_grad=True) #requires_grad=True for optimization
X_test_norm_tensor=X_test_norm_tensor.type(torch.float32)
print("X_test",X_test_norm_tensor.size(),X_test_norm_tensor.dtype)

X_train_norm_tensor=torch.tensor((X_train_norm),requires_grad=True)
X_train_norm_tensor=X_train_norm_tensor.type(torch.float32)
print("X_train",X_train_norm_tensor.size(),X_train_norm_tensor.dtype)

y_test_norm_tensor=torch.tensor((y_test_norm),requires_grad=True)
y_test_norm_tensor=y_test_norm_tensor.type(torch.float32)
print("y_test",y_test_norm_tensor.size(),y_test_norm_tensor.dtype)

y_train_norm_tensor=torch.tensor((y_train_norm),requires_grad=True)
y_train_norm_tensor=y_train_norm_tensor.type(torch.float32)
print("y_train",y_train_norm_tensor.size(),y_train_norm_tensor.dtype)
print("-------------------------------------")
#transposing required tensors
y_test_norm_tensor=y_test_norm_tensor.transpose(0,1)
print("y_test",y_test_norm_tensor.size(),y_test_norm_tensor.dtype)
y_train_norm_tensor=y_train_norm_tensor.transpose(0,1)
print("y_train",y_train_norm_tensor.size(),y_train_norm_tensor.dtype)

input_size=22 # Number of input features
hidden_size=[20,20,20] #Number of neurons in the hidden layer
ouput_size=2 # Number of output classes

#creating a network
net=KNN(input_size,hidden_size,ouput_size)

#defining the loss function and optimizer
criterion= nn.MSELoss() # https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L1423 line no: 3309
optimizer=optim.SGD(net.parameters(),lr=0.01)

X_train=X_train_norm_tensor
y_train=y_train_norm_tensor
#training
for epoch in range(100):
    #forward pass
    outputs=net(X_train_norm_tensor)
    #compute the loss
    loss= criterion(outputs, y_train_norm_tensor)
    #back propagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#save the model
torch.save(net.state_dict(),"Trained_KNN_model.pth")