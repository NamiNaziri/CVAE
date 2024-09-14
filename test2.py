class WAE_Encoder(nn.Module):
    def __init__(self, args):
        super(WAE_Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        
        # convolutional filters, work excellent with image data
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
        )
        
        # final layer is fully connected
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

class WAE_Decoder(nn.Module):
    def __init__(self, args):
        super(WAE_Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # first layer is fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU()
        )

        # deconvolutional filters, essentially the inverse of convolutional filters
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.deconv(x)
        return x

# define the descriminator
class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # main body of discriminator, returns [0,1]
        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x
    
# control which parameters are frozen / free for optimization
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False







# instantiate discriminator model, and restart encoder and decoder, for fairness. Set to train mode, etc
wae_encoder, wae_decoder, discriminator = WAE_Encoder(args), WAE_Decoder(args), Discriminator(args)

criterion = nn.MSELoss()

if args['train']:
    enc_optim = torch.optim.Adam(wae_encoder.parameters(), lr = args['lr'])
    dec_optim = torch.optim.Adam(wae_decoder.parameters(), lr = args['lr'])
    dis_optim = torch.optim.Adam(discriminator.parameters(), lr = args['lr'])

    enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, step_size=30, gamma=0.5)
    dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, step_size=30, gamma=0.5)
    dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optim, step_size=30, gamma=0.5)

    # one and -one allow us to control descending / ascending gradient descent
    one = torch.tensor([1], dtype=torch.float)
    
    for epoch in range(args['epochs']):

        # train for one epoch -- set nets to train mode
        wae_encoder.train()
        wae_decoder.train()
        discriminator.train()
        
        # Included are elements similar to the Schelotto (2018) implementation
        # on GitHub. Schelotto's implementation repository is worth looking into, 
        # because the WAE-MMD ("Maximum Mean Discrepancy") implementation, a second 
        # WAE algorithm discussed in the original Wasserstein Auto-Encoders paper,
        # is also implemented there.

        for images, _ in tqdm(train_loader):
            # zero gradients for each batch
            wae_encoder.zero_grad()
            wae_decoder.zero_grad()
            discriminator.zero_grad()

            #### TRAIN DISCRIMINATOR ####

            # freeze auto encoder params
            frozen_params(wae_decoder)
            frozen_params(wae_encoder)

            # free discriminator params
            free_params(discriminator)

            # run discriminator against randn draws
            z = torch.randn(images.size()[0], args['n_z']) * args['sigma']
            d_z = discriminator(z)

            # run discriminator against encoder z's
            z_hat = wae_encoder(images)
            d_z_hat = discriminator(z_hat)

            d_z_loss = args['lambda']*torch.log(d_z).mean()
            d_z_hat_loss = args['lambda']*torch.log(1 - d_z_hat).mean()

            # formula for ascending the descriminator -- -one reverses the direction of the gradient.
            d_z_loss.backward(-one)
            d_z_hat_loss.backward(-one)

            dis_optim.step()

            #### TRAIN GENERATOR ####

            # flip which networks are frozen, which are not
            free_params(wae_decoder)
            free_params(wae_encoder)
            frozen_params(discriminator)

            batch_size = images.size()[0]

            # run images
            z_hat = wae_encoder(images)
            x_hat = wae_decoder(z_hat)

            # discriminate latents
            z_hat2 = wae_encoder(Variable(images.data))
            d_z_hat = discriminator(z_hat2)

            # calculate reconstruction loss
            # WAE is happy with whatever cost function, let's use BCE
            BCE = nn.functional.binary_cross_entropy(
                x_hat.view(-1,784), 
                images.view(-1, 784), 
                reduce=False
            ).mean()
            
            # calculate discriminator loss
            d_loss = args['lambda'] * (torch.log(d_z_hat)).mean()
            
            # we keep the BCE and d_loss on separate graphs to increase efficiency in pytorch
            BCE.backward(one)
            # -one reverse the direction of the gradient, minimizing BCE - d_loss
            d_loss.backward(-one)

            enc_optim.step()
            dec_optim.step()

        # test on test set
        wae_encoder.eval()
        wae_decoder.eval()
        for images, _ in tqdm(test_loader):
            z_hat = wae_encoder(images)
            x_hat = wae_decoder(z_hat)
            test_recon_loss = criterion(x_hat, images)

        
        if args['save']:
            save_path = './save/WAEgan_{}-epoch_{}.pth'
            torch.save(wae_encoder.state_dict(), save_path.format('encoder', epoch))
            torch.save(wae_decoder.state_dict(), save_path.format('decoder', epoch))
            torch.save(discriminator.state_dict(), save_path.format('discriminator', epoch))

        # print stats after each epoch
        print("Epoch: [{}/{}], \tTrain Reconstruction Loss: {} d loss: {}, \n"\
              "\t\t\tTest Reconstruction Loss:{}".format(
            epoch + 1, 
            args['epochs'], 
            BCE.data.item(),
            d_loss.data.item(),
            test_recon_loss.data.item()
        ))
        
else:
    enc_checkpoint = torch.load('save/WAEgan_encoder-best_{}.pth'.format(args['dataset']))
    wae_encoder.load_state_dict(enc_checkpoint)

    dec_checkpoint = torch.load('save/WAEgan_decoder-best_{}.pth'.format(args['dataset']))
    wae_decoder.load_state_dict(dec_checkpoint)
    
    dec_checkpoint = torch.load('save/WAEgan_discriminator-best_{}.pth'.format(args['dataset']))
    discriminator.load_state_dict(dec_checkpoint)