import torch
import config
from torch import nn
from torch import optim
from utils import  gradient_penalty,load_checkpoint, save_checkpoint, plot_examples
from loss import Generator_Loss,Discriminator_Loss,ContentLoss

from torch.utils.data import DataLoader
from model import Generator, Discriminator, initialize_weights
from tqdm import tqdm
from dataset import DIV2k_dataset
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True

def train_fn(
    loader,
    disc,
    gen,
    opt_gen,
    opt_disc,
    l1,
    
    writer,
    tb_step,
):
    print("entered training loop")
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        
        real=high_res
        fake = gen(low_res)
            
         # gp = gradient_penalty(disc, high_res, fake, device=config.DEVICE)
        loss =l1(fake,real)
        
        opt_gen.zero_grad()
        opt_disc.zero_grad()
        loss.backward()
        # opt_disc.step()
        opt_gen.step()
        
        writer.add_scalar("pretrain loss", loss.item(), global_step=tb_step)
        

        
        

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        
            
        

        
        

        tb_step += 1

        if idx % 100 == 0 and idx > 0:
            plot_examples("test_images/", gen)

        loop.set_postfix(
            loss=loss.item(),
            
            
            )

    return tb_step


def main():
    dataset = DIV2k_dataset(root_dir="D:\ESRGAN\dataset\DIV2K_train_HR")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    
    print(f"DATA LOADED length -{len(dataset)}")
    
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    initialize_weights(gen)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.99))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.99),)
    writer = SummaryWriter("logs-pretrain")
    tb_step = 0
    l1 = ContentLoss()
    # gen_loss=Generator_Loss(config.L,config.N)
    # disc_loss=Discriminator_Loss()
    gen.train()
    disc.train()
    

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    if False:
        load_checkpoint(
            './checkpoint/pretrain_gen.pth',
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            './checkpoint/pretrain_disc.pth',
            disc,
            opt_disc,
            config.LEARNING_RATE,
        )

    print(f"about enter training loop")
    for epoch in range(100):
        tb_step = train_fn(
            loader,
            disc,
            gen,
            opt_gen,
            opt_disc,
            l1,
             
            writer,
            tb_step,
        )

        if True:
            save_checkpoint(gen, opt_gen, filename=".\checkpoint\pretrain_gen.pth")
            save_checkpoint(disc, opt_disc, filename='.\checkpoint\pretrain_disc.pth')


if __name__ == "__main__":
    try_model = False

    if try_model:
        # Will just use pretrained weights and run on images
        # in test_images/ and save the ones to SR in saved/
        gen = Generator(in_channels=3).to(config.DEVICE)
        opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
        # load_checkpoint(
        #     config.CHECKPOINT_GEN,
        #     gen,
        #     opt_gen,
        #     config.LEARNING_RATE,
        # )
        plot_examples("test_images/", gen)
    else:
        # This will train from scratch
        main()