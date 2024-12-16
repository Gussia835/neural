import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#моделька
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#функция потерь и потимизатор даб
input_size = 28 * 28  
hidden_size = 128     
output_size = 10      

model = MLP(input_size, hidden_size, output_size)
criterion = nn.NLLLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)  

#Обучение
num_epochs = 5

for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
