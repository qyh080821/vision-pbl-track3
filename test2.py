import torch.utils.data
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
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
        # self.fc3 = nn.Linear(hidden_size1, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
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


class VAE0(nn.Module):
    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        super(VAE0, self).__init__()

        self.input_dim = input_dim
        # self.h_dim = h_dim
        # self.z_dim = z_dim

        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)

        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)
        return mu, log_var

    def reparameterization(self, mu, log_var):
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, z):
        h = F.relu(self.fc4(z))
        x_hat = torch.sigmoid(self.fc5(h))
        return x_hat

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.input_dim)

        mu, log_var = self.encode(x)
        sampled_z = self.reparameterization(mu, log_var)
        x_hat = self.decode(sampled_z)
        x_hat = x_hat.view(batch_size, 1, 28, 28)
        return x_hat, mu, log_var

class VAE(nn.Module):
    def __init__(self, input_dim=784, h_dim=400,h_dim2 = 200, z_dim=20):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        # self.h_dim = h_dim
        # self.z_dim = z_dim

        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc11 = nn.Linear(h_dim, h_dim2)
        self.fc2 = nn.Linear(h_dim2, z_dim)
        self.fc3 = nn.Linear(h_dim2, z_dim)

        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc44 = nn.Linear(h_dim2, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc11(h))
        mu = self.fc2(h)
        log_var = self.fc3(h)
        return mu, log_var

    def reparameterization(self, mu, log_var):
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc44(h))
        x_hat = torch.sigmoid(self.fc5(h))
        return x_hat

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.input_dim)

        mu, log_var = self.encode(x)
        sampled_z = self.reparameterization(mu, log_var)
        x_hat = self.decode(sampled_z)
        x_hat = x_hat.view(batch_size, 1, 28, 28)
        return x_hat, mu, log_var


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :28, :28]


class VAE2(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.Flatten(),
        )

        self.z_mean = torch.nn.Linear(3136, 2)
        self.z_log_var = torch.nn.Linear(3136, 2)

        self.decoder = nn.Sequential(
            torch.nn.Linear(2, 3136),
            Reshape(-1, 64, 7, 7),
            nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0),
            Trim(),  # 1x29x29 -> 1x28x28
            nn.Sigmoid()
        )

    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1))
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return decoded, z_mean, z_log_var

def loss_fn(x_hat, x, mu, log_var):
    """
    Calculate the loss. Note that the loss includes two parts.
    :param x_hat:
    :param x:
    :param mu:
    :param log_var:
    :return: total loss, BCE and KLD of our model
    """

    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum') #计算x_hat和x的交叉熵

    KLD = -0.5 * torch.sum(1-torch.exp(log_var) -torch.pow(mu, 2)  + log_var) #tensor 张量：本质是一个多维数组，用来创造高维矩阵、向量

    loss = BCE + KLD
    return loss, BCE, KLD



plt.imshow(x_example[0,:], cmap='gray')
plt.show()

enc = Encoder(28*28,256,100, 100)
dec = Decoder(28*28, 256, 100, 100)
model = VAE2()
# loss_fn = nn.MSELoss()
# loss_fn = nn.BCELoss()

optimizer_enc = torch.optim.Adam(enc.parameters(), lr=0.005)
optimizer_dec = torch.optim.Adam(dec.parameters(), lr=0.005)
opt = torch.optim.Adam(model.parameters(), lr=0.002  )

train_loss = []
for epoch in range(20):
    train_epoch_loss = []
    index = 0
    for (img, _) in train_dl:
        # img2 = img.view(img.shape[0], -1)
        img2 = img
        # latents = enc(img2)
        # output = dec(latents)
        # loss = loss_fn(output, img2)
        output, mu, log_var = model(img2)
        loss, BCE, KLD = loss_fn(output, img2, mu, log_var)
        # train_epoch_loss += loss.cpu().detach().numpy()
        train_epoch_loss.append(loss.item())
        # optimizer_enc.zero_grad()
        # optimizer_dec.zero_grad()
        opt.zero_grad()
        loss.backward()
        opt.step()
        # optimizer_enc.step()
        # optimizer_dec.step()
        if (index + 1) % 100 == 0:
            print('Epoch [{}/{}], Batch [{}/{}] : Total-loss = {:.4f}, BCE-Loss = {:.4f}, KLD-loss = {:.4f}'
                  .format(epoch + 1, 20, index + 1, len(train_dl.dataset) // 100,
                          loss.item() / 100, BCE.item() / 100,
                          KLD.item() / 100))
        index += 1
    # train_loss.append(train_epoch_loss)
    # print(epoch, train_epoch_loss)
    train_loss.append(np.sum(train_epoch_loss) / len(train_dl.dataset))
gimg = output.cpu().detach().numpy()
show_data(gimg)
plt.show()
plt.plot(train_loss)
plt.show()
# img3 = img2.view(img2.shape[0], 28,28,1)
# cimg = img3.repeat(1,1,1,3)
# gimg = output.cpu().detach().numpy()
# gimg.resize(100, 28,28)
# plt.imshow(gimg[0])
# plt.plot(train_loss)
# plt.show()
#
test_dl = torch.utils.data.DataLoader(mnist_data_test, batch_size=10)
for (img, _) in test_dl:
    pass
img2 = img
output, mu, log_var = model(img2)
gimg = output.cpu().detach().numpy()
show_data(gimg)
plt.show()
show_data(img)
plt.show()
# latents = enc(img2)
# output = dec(latents)
# gimg = output.cpu().detach().numpy()
# gimg.resize(10, 28,28)
# show_data(gimg)

