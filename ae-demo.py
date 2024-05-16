import torch.utils.data
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
mnist_data_train = torchvision.datasets.MNIST("./data", train=True,download=True, transform=transforms.ToTensor())
mnist_data_test = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
x_example, y_example = mnist_data_train[0]
print(x_example.shape)

train_dl = torch.utils.data.DataLoader(mnist_data_train, batch_size=100)

class Encoder(nn.Module):
    def __init__(self, input_size = 28*28, hidden_size1 = 128, hidden_size2 =16, z_dim = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, output_size=28 * 28, hidden_size1=128, hidden_size2=16, z_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

plt.imshow(x_example[0,:], cmap='gray')
plt.show()

enc = Encoder(28*28, 512, 128,5)
dec = Decoder(28*28, 512, 128,5)

loss_fn = nn.MSELoss()
optimizer_enc = torch.optim.Adam(enc.parameters(), lr=0.005)
optimizer_dec = torch.optim.Adam(dec.parameters(), lr=0.005)


train_loss = []
for epoch in range(30):
    train_epoch_loss = 0
    for (img, _) in train_dl:
        img2 = img.view(img.shape[0], -1)
        latents = enc(img2)
        output = dec(latents)
        loss = loss_fn(output, img2)
        train_epoch_loss += loss.cpu().detach().numpy()
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        loss.backward()
        optimizer_enc.step()
        optimizer_dec.step()
    train_loss.append(train_epoch_loss)
    print(epoch, train_epoch_loss)
img3 = img2.view(img2.shape[0], 28,28,1)
cimg = img3.repeat(1,1,1,3)
gimg = output.cpu().detach().numpy()
gimg.resize(100, 28,28)
plt.imshow(gimg[0])
plt.plot(train_loss)
plt.show()
