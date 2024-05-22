#https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L17/1_VAE_mnist_sigmoid_mse.ipynb
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

def show_data(X, n=10, height=28, width=28, title=""):
    plt.figure(figsize=(10, 3))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        plt.imshow(X[i].reshape((height,width)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(title, fontsize = 20)
class Encoder(nn.Module):
    def __init__(self, input_size = 28*28, hidden_size1 = 128, hidden_size2 =16, z_dim = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size1, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, output_size=28 * 28, hidden_size1=128, hidden_size2=16, z_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size2, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class Encoder(nn.Module):
    def __init__(self, input_size = 28*28, hidden_size1 = 128, hidden_size2 =16, z_dim = 3):
        super().__init__()
        # self.fc1 = nn.Linear(input_size, hidden_size1)
        # self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # self.fc3 = nn.Linear(hidden_size1, z_dim)
        # self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1)
        self.relu = nn.LeakyReLU(0.01)


    def forward(self, x):
        # x = self.relu(self.fc1(x))
        # # x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = nn.Flatten(x)
        return x


class Decoder(nn.Module):
    def __init__(self, output_size=28 * 28, hidden_size1=128, hidden_size2=16, z_dim=3):
        super().__init__()
        # self.fc1 = nn.Linear(z_dim, hidden_size1)
        # self.fc2 = nn.Linear(hidden_size2, hidden_size1)
        # self.fc3 = nn.Linear(hidden_size1, output_size)
        # self.relu = nn.ReLU()
        self.ln = nn.Linear(3, 2136)
        self.tconv1 = nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1)
        self.tconv2 = nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1)
        self.tconv3 = nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=1)
        self.tconv4 = nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=1)
        self.relu = nn.LeakyReLU(0.01)


    def forward(self, x):
        # x = self.relu(self.fc1(x))
        # # x = self.relu(self.fc2(x))
        # x = torch.sigmoid(self.fc3(x))
        # return x
        x = self.ln(x)
        x = x.view(-1, 64, 7, 7)
        x = self.tconv1(x)
        x = self.relu(x)
        x = self.tconv2(x)
        x = self.relu(x)
        x = self.tconv3(x)
        x = self.relu(x)
        x = self.tconv4(x)
        x = x[:,:,:28,:28]
        x = nn.Sigmoid(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.mean = nn.Linear(3136,2)
        self.log_var = nn.Linear(3136,2)
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.mean(x), self.log_var(x)
        eps = torch.rand(z_mean.size(0), z_mean.size(1)).to(z_mean.get_device())
        z = z_mean+eps*torch.exp(z_log_var/2.0)
        result = self.decoder(z)
        return x, z_mean, z_log_var, result

plt.imshow(x_example[0,:], cmap='gray')
plt.show()

enc = Encoder(28*28,256,100, 100)
dec = Decoder(28*28, 256, 100, 100)

loss_fn = nn.MSELoss()
loss_fn = nn.BCELoss()
optimizer_enc = torch.optim.Adam(enc.parameters(), lr=0.005)
optimizer_dec = torch.optim.Adam(dec.parameters(), lr=0.005)


train_loss = []
for epoch in range(20):
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


