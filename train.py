import torch
import config
import os
from torch import nn
from torch import optim
from utils import  gradient_penalty,load_checkpoint, save_checkpoint, plot_examples
from loss import Generator_Loss,Discriminator_Loss

from torch.utils.data import DataLoader
from model import Generator, Discriminator, initialize_weights
from tqdm import tqdm
from dataset import DIV2k_dataset
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True
import torchvision.utils as vutils

def log_generated_and_real_images(writer, gen, low_res, high_res, epoch, tb_step):
    # Generate fake images using the generator
    with torch.no_grad():
        gen.eval()
        fake = gen(low_res)
        gen.train()

    # Create a grid of fake, real, and low_res images
    fake_grid = vutils.make_grid(fake[:4], normalize=True, scale_each=True)  # First 4 generated images
    real_grid = vutils.make_grid(high_res[:4], normalize=True, scale_each=True)  # First 4 real images
    low_res_grid = vutils.make_grid(low_res[:4], normalize=True, scale_each=True)  # First 4 low_res images
    
    # Add the low_res images to TensorBoard
    writer.add_image(f"Low-Res Images at Epoch ", low_res_grid, global_step=tb_step)

    # Add the fake images to TensorBoard
    writer.add_image(f"Fake Images at Epoch ", fake_grid, global_step=tb_step)
    
    # Add the real images to TensorBoard
    writer.add_image(f"Real Images at Epoch ", real_grid, global_step=tb_step)

def train_fn(
    loader,
    disc,
    gen,
    opt_gen,
    opt_disc,
    gen_loss,
    disc_loss,
    g_scaler,
    d_scaler,
    writer,
    tb_step,
):
    print("entered training loop")
    loop = tqdm(loader, leave=True)
    
    last_low_res = None  # To store the last batch of low_res images
    last_high_res = None  # To store the last batch of high_res images


    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            real=high_res
            fake = gen(low_res)
            
            # gp = gradient_penalty(disc, high_res, fake, device=config.DEVICE)
            loss_critic =disc_loss(fake.detach(),real,disc)+config.LAMBDA_GP*gradient_penalty(disc,real,fake,config.DEVICE)

        
        # if idx%5==0:
        opt_disc.zero_grad()
        d_scaler.scale(loss_critic).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        with torch.cuda.amp.autocast():
            
            loss_gen = gen_loss(fake,real,disc)

        opt_gen.zero_grad()
        g_scaler.scale(loss_gen).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        writer.add_scalar("Critic loss", loss_critic.item(), global_step=tb_step)
        writer.add_scalar("Gen loss", loss_gen.item(), global_step=tb_step)
        tb_step += 1

        if idx % 100 == 0 and idx > 0:
            plot_examples("test_images/", gen)

        loop.set_postfix(
            loss_gen=loss_gen.item(),
            loss_critic=loss_critic.item(),
            
            )
        last_low_res = low_res  # Store the last batch of low_res images
        last_high_res = high_res  # Store the last batch of high_res images


    return tb_step, last_low_res, last_high_res  # Return tb_step, low_res, and high_res


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
    writer = SummaryWriter("logs")
    tb_step = 0
    # l1 = nn.L1Loss()
    gen_loss=Generator_Loss(config.L,config.N)
    disc_loss=Discriminator_Loss()
    gen.train()
    disc.train()
    

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(
            "pretrain_checkpoint\pretrain_gen.pth",
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            "pretrain_checkpoint\pretrain_disc.pth",
            disc,
            opt_disc,
            config.LEARNING_RATE,
        )

    print(f"about enter training loop")
    for epoch in range(config.NUM_EPOCHS):
        tb_step, last_low_res, last_high_res = train_fn(
            loader,
            disc,
            gen,
            opt_gen,
            opt_disc,
            gen_loss,
            disc_loss,
            g_scaler,
            d_scaler,  
            writer,
            tb_step,
        )
        # Log generated and real images after each epoch for comparison
        log_generated_and_real_images(writer, gen, last_low_res, last_high_res, epoch, tb_step)

        writer.add_scalar("epoch number",epoch,global_step=tb_step)
        

        if config.SAVE_MODEL and (epoch)%20==0:
            save_checkpoint(gen, opt_gen, filename=os.path.join("training_checkpoint",f"train_gen_{epoch}.pth"))
            save_checkpoint(disc, opt_disc, filename=os.path.join("training_checkpoint",f"train_disc_{epoch}.pth"))


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