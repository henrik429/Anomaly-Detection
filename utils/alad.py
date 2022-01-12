import torch
import os
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.networks_image_data import *
from utils.dataloader import Dataloader

Tensor = torch.FloatTensor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        m.bias.data.zero_()


class ALAD():
    def __init__(self, dataset, batch_size, network_name,
                 device="cpu",
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs=None,
                 max_epochs=100,
                 latent_dim=100,
                 verbose=False,
                 betas=(0.5, 0.999),
                 report_interval=10,
                 checkpoint_interval=500):

        self.writer = None
        self.dataset = dataset
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        self.network_name = network_name
        self.network_dir = f'{os.getcwd()}/network_dir/{network_name}'

        self.report_interval = report_interval
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.device = device

        self.epoch_id = 0
        self.step_id = 0

        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 2e-4}

        self.encoder = Encoder(latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim).to(self.device)
        self.dis_xz = DiscriminatorXZ(latent_dim).to(self.device)
        self.dis_xx = DiscriminatorXX(latent_dim).to(self.device)
        self.dis_zz = DiscriminatorZZ(latent_dim).to(self.device)

        if dataset == "CIFAR10":
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')

        self.encoder_optimizer = optimizer(self.encoder.parameters(), betas=betas, **optimizer_kwargs)
        self.decoder_optimizer = optimizer(self.decoder.parameters(), betas=betas, **optimizer_kwargs)
        self.dis_xz_optimizer = optimizer(self.dis_xz.parameters(), betas=betas, **optimizer_kwargs)
        self.dis_xx_optimizer = optimizer(self.dis_zz.parameters(), betas=betas, **optimizer_kwargs)
        self.dis_zz_optimizer = optimizer(self.dis_zz.parameters(), betas=betas, **optimizer_kwargs)

        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.l1 = nn.L1Loss(reduction='none').to(self.device)

    def save_networks(self, normal_sample):
        path = f'{self.network_dir}/{normal_sample}.pth'
        torch.save({'Generator': self.decoder.state_dict(),
                    'Encoder': self.encoder.state_dict(),
                    'DiscriminatorXZ': self.dis_xz.state_dict(),
                    'DiscriminatorXX': self.dis_xx.state_dict(),
                    'DiscriminatorZZ': self.dis_zz.state_dict()}, path)

    def load_networks(self, normal_sample):
        """Load weights."""
        path = f'{self.network_dir}/{normal_sample}.pth'
        state_dict = torch.load(path)

        self.decoder.load_state_dict(state_dict['Generator'])
        self.encoder.load_state_dict(state_dict['Encoder'])
        self.dis_xz.load_state_dict(state_dict['DiscriminatorXZ'])
        self.dis_xx.load_state_dict(state_dict['DiscriminatorXX'])
        self.dis_zz.load_state_dict(state_dict['DiscriminatorZZ'])

    def weights_init(self):
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.dis_xz.apply(weights_init)
        self.dis_xx.apply(weights_init)
        self.dis_zz.apply(weights_init)

    def forward(self, x):
        """
        One forward step of one batch x through whole ALAD model.
        """
        z_gen = self.encoder(x)

        z = Variable(Tensor(np.random.normal(0, 1, (x.shape[0], self.latent_dim))), requires_grad=False).to(self.device)
        x_gen = self.decoder(z)

        rec_x = self.decoder(z_gen)
        rec_z = self.encoder(x_gen)

        out_dis_x_z_gen = self.dis_xz(x, z_gen)
        out_dis_xgen_z = self.dis_xz(x_gen, z)

        x_logit_real, _ = self.dis_xx(x, x)
        x_logit_fake, _ = self.dis_xx(x, rec_x)

        z_logit_real = self.dis_zz(z, z)
        z_logit_fake = self.dis_zz(z, rec_z)

        return z_gen, x_gen, out_dis_x_z_gen, out_dis_xgen_z, x_logit_real, x_logit_fake, z_logit_real, z_logit_fake

    def step(self, x):
        """Performs a single step of ALAD training.
        Args:
          data: data points used for training."""

        #self.optimizer_d.zero_grad()
        #self.optimizer_g.zero_grad()

        self.dis_xz_optimizer.zero_grad()
        self.dis_xx_optimizer.zero_grad()
        self.dis_zz_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        z_gen, x_gen, out_dis_x_z_gen, out_dis_xgen_z, x_logit_real, x_logit_fake, z_logit_real, z_logit_fake = \
            self.forward(x)

        # Calculate losses

        # discriminator xz
        loss_dis_enc = self.criterion(out_dis_x_z_gen, torch.ones_like(out_dis_x_z_gen))
        loss_dis_gen = self.criterion(out_dis_xgen_z, torch.zeros_like(out_dis_xgen_z))
        dis_loss_xz = loss_dis_gen + loss_dis_enc

        # discriminator xx
        x_real_dis = self.criterion(x_logit_real, torch.ones_like(x_logit_real))
        x_fake_dis = self.criterion(x_logit_fake, torch.zeros_like(x_logit_fake))
        dis_loss_xx = x_real_dis + x_fake_dis

        # discriminator zz
        z_real_dis = self.criterion(z_logit_real, torch.ones_like(z_logit_real))
        z_fake_dis = self.criterion(z_logit_fake, torch.zeros_like(z_logit_fake))
        dis_loss_zz = z_real_dis + z_fake_dis

        #loss_d = dis_loss_xz + dis_loss_zz + dis_loss_xx
        #loss_d.backward(retain_graph=True)
        #self.optimizer_d.step()

        # decoder and encoder
        gen_loss_xz = self.criterion(out_dis_xgen_z, torch.ones_like(out_dis_xgen_z))
        enc_loss_xz = self.criterion(out_dis_x_z_gen, torch.zeros_like(out_dis_x_z_gen))
        x_real_gen = self.criterion(x_logit_real, torch.ones_like(x_logit_real))
        x_fake_gen = self.criterion(x_logit_fake, torch.zeros_like(x_logit_fake))
        z_real_gen = self.criterion(z_logit_real, torch.ones_like(z_logit_real))
        z_fake_gen = self.criterion(z_logit_fake, torch.zeros_like(z_logit_fake))

        cycle_consistency_loss = x_real_gen + x_fake_gen + z_real_gen + z_fake_gen
        loss_generator = gen_loss_xz + cycle_consistency_loss
        loss_encoder = enc_loss_xz + cycle_consistency_loss

        dis_loss_xz.backward(retain_graph=True)
        dis_loss_xx.backward(retain_graph=True)
        dis_loss_zz.backward(retain_graph=True)
        loss_generator.backward(retain_graph=True)
        loss_encoder.backward()

        self.dis_xz_optimizer.step()
        self.dis_xx_optimizer.step()
        self.dis_zz_optimizer.step()
        self.decoder_optimizer.step()
        self.encoder_optimizer.step()

        #loss_g = loss_generator + loss_encoder
        #loss_g.backward(retain_graph=True)
        #self.optimizer_g.step()

        # If verbose is True, then save losses of training.
        if self.verbose:
            self.writer.add_scalar("discriminator_xz loss", dis_loss_xz.item(), self.step_id)
            self.writer.add_scalar("discriminator_xx loss", dis_loss_xx.item(), self.step_id)
            self.writer.add_scalar("discriminator_zz loss", dis_loss_zz.item(), self.step_id)
            self.writer.add_scalar("generator loss", loss_generator.item(), self.step_id)
            self.writer.add_scalar("encoder loss", loss_encoder.item(), self.step_id)
            # self.writer.add_scalar("cycle consistency loss", cycle_consistency_loss, self.step_id)

    def valid_step(self, x):
        """Performs a single step of VAE validation.
        Args:
          data: data points used for validation."""
        with torch.no_grad():
            z_gen, x_gen, out_dis_x_z_gen, out_dis_xgen_z, x_logit_real, x_logit_fake, z_logit_real, z_logit_fake = \
                self.forward(x)

            # Calculate losses

            # discriminator xz
            loss_dis_enc = self.criterion(out_dis_x_z_gen, torch.ones_like(out_dis_x_z_gen))
            loss_dis_gen = self.criterion(out_dis_xgen_z, torch.zeros_like(out_dis_xgen_z))
            dis_loss_xz = loss_dis_gen + loss_dis_enc
            self.writer.add_scalar("valid_discriminator_xz loss", dis_loss_xz.item(), self.step_id)

            # discriminator xx
            x_real_dis = self.criterion(x_logit_real, torch.ones_like(x_logit_real))
            x_fake_dis = self.criterion(x_logit_fake, torch.zeros_like(x_logit_fake))
            dis_loss_xx = x_real_dis + x_fake_dis
            self.writer.add_scalar("valid_discriminator_xx loss", dis_loss_xx.item(), self.step_id)

            # discriminator zz
            z_real_dis = self.criterion(z_logit_real, torch.ones_like(z_logit_real))
            z_fake_dis = self.criterion(z_logit_fake, torch.zeros_like(z_logit_fake))
            dis_loss_zz = z_real_dis + z_fake_dis
            self.writer.add_scalar("valid_discriminator_zz loss", dis_loss_zz.item(), self.step_id)

            # decoder and encoder
            gen_loss_xz = self.criterion(out_dis_xgen_z, torch.ones_like(out_dis_xgen_z))
            enc_loss_xz = self.criterion(out_dis_x_z_gen, torch.zeros_like(out_dis_x_z_gen))
            x_real_gen = self.criterion(x_logit_real, torch.ones_like(x_logit_real))
            x_fake_gen = self.criterion(x_logit_fake, torch.zeros_like(x_logit_fake))
            z_real_gen = self.criterion(z_logit_real, torch.ones_like(z_logit_real))
            z_fake_gen = self.criterion(z_logit_fake, torch.zeros_like(z_logit_fake))

            cycle_consistency_loss = x_real_gen + x_fake_gen + z_real_gen + z_fake_gen
            # self.writer.add_scalar("valid_cycle_consistency_loss", cycle_consistency_loss, self.step_id)

            loss_generator = gen_loss_xz + cycle_consistency_loss
            self.writer.add_scalar("valid_generator_loss", loss_generator.item(), self.step_id)

            loss_encoder = enc_loss_xz + cycle_consistency_loss
            self.writer.add_scalar("valid_encoder_loss", loss_encoder.item(), self.step_id)

    def train(self):
        # each label should be one time the normal sample whereas all other are anomalous
        for anomalous_i, anomalous_class in enumerate(self.classes):
            trainloader, validloader = Dataloader(self.dataset, anomalous_i, batch_size=self.batch_size)

            self.writer = SummaryWriter(f"{self.network_dir}/{anomalous_class}")
            self.epoch_id = 0
            self.step_id = 0
            self.weights_init()

            print("-"*100)
            print('-'*24, f'Start training with class {anomalous_class} as anomalous class', '-'*24)
            print("-"*100)

            for epoch_id in tqdm(range(self.max_epochs)):
                self.epoch_id = epoch_id
                if validloader is not None:
                    valid_iter = iter(validloader)

                for (data, _) in trainloader:
                    self.step(data.to(self.device))
                    if validloader is not None and self.step_id % self.report_interval == 0:
                        try:
                            vdata, _ = next(valid_iter)
                        except StopIteration:
                            valid_iter = iter(validloader)
                            vdata, _ = next(valid_iter)
                        self.valid_step(vdata.to(self.device))
                    self.step_id += 1

            self.save_networks(anomalous_class)

    def score_function(self, x):
        x = x.to(self.device)
        _, real_feature = self.dis_xx(x, x)
        rec_x = self.decoder(self.encoder(x))
        _, fake_feature = self.dis_xx(x, rec_x)
        score = self.l1(real_feature, fake_feature)
        return torch.sum(score, dim=1)

    def evaluate(self):
        class_scores = []
        class_targets = []

        print("Start evaluation...")
        # each label should be one time the anomalous sample
        for anomalous_i, anomalous_class in enumerate(self.classes):
            testloader, num_anomalous_sample = Dataloader(self.dataset, anomalous_i, batch_size=self.batch_size, training=False)

            print("Number of anomalous samples: {}".format(num_anomalous_sample))
            self.load_networks(anomalous_class)
            self.encoder.eval()
            self.decoder.eval()
            self.dis_xx.eval()

            targets = []
            scores = []
            with torch.no_grad():
                for imgs, label in tqdm(testloader):
                    score = self.score_function(imgs)
                    scores.extend(list(score.cpu().detach().numpy()))
                    targets.extend(list(label.cpu().detach().numpy()))

            class_targets.append(targets)
            class_scores.append(scores)

        class_targets = np.array(class_targets)
        class_scores = np.array(class_scores)
        self.plot_roc(class_scores, class_targets)

        return class_targets, class_scores

    def plot_roc(self, class_scores, class_targets):
        # Initialize plot of ROC curves for each class
        lw = 1
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        ax.set_prop_cycle(color=
            ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
             "#17becf"]
        )
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver operating characteristic example")
        ax.legend(loc='lower right')
        ax.grid()
        lines = []

        # Initialize bar plot for each class
        roc_auc = np.zeros(len(self.classes), dtype=np.float)

        for anomalous_i, anomalous_class in enumerate(self.classes):
            scores = class_scores[anomalous_i].astype(np.float)
            targets = class_targets[anomalous_i].astype(np.uint8)

            roc_auc[anomalous_i] = roc_auc_score(targets, scores)
            print('ROC AUC score of class {}: {:.2f}'.format(anomalous_class, roc_auc[anomalous_i] * 100))
            fpr, tpr, _ = roc_curve(targets, scores, pos_label=1)

            line, = ax.plot(
                fpr,
                tpr,
                lw=lw,
                label="%s (AUC = %0.2f)" % (anomalous_class, roc_auc[anomalous_i])
            )
            lines.append(line)

        ax.legend(lines, self.classes)
        plt.show()
        plt.close()

        # AUC bar plot
        fig, ax1 = plt.subplots()
        y_pos = np.arange(len(self.classes))
        plt.xticks(y_pos, self.classes)

        # Create names on the x-axis
        plt.xlabel("AUC")
        plt.ylabel("Classes")

        ax1.set_prop_cycle(color=
                           ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                            "#bcbd22",
                            "#17becf"]
                           )

        # Create bars
        plt.bar(y_pos, roc_auc, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                            "#bcbd22",
                            "#17becf"])

        # Show graphic
        plt.show()

