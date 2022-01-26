import torch
import os
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, PrecisionRecallDisplay, precision_recall_fscore_support

import utils.networks_image_data as img
import utils.networks_tabular_data as tab
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
        m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        m.bias.data.zero_()


def xavier_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data)
        m.bias.data.zero_()


class ALAD():
    def __init__(self, dataset, batch_size, network_name,
                 device="cpu",
                 optimizer=torch.optim.Adam,
                 max_epochs=100,
                 verbose=False,
                 betas=(0.5, 0.999),
                 report_interval=10,
                 checkpoint_interval=500):

        self.writer = None
        self.dataset = dataset
        self.batch_size = batch_size

        self.network_name = network_name
        self.network_dir = f'{os.getcwd()}/network_dir/{network_name}'

        self.report_interval = report_interval
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.device = device

        self.epoch_id = 0
        self.step_id = 0

        if dataset == "CIFAR10":
            optimizer_kwargs = {"lr": 2e-4}

            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog') # , 'horse', 'ship', 'truck')
            latent_dim = 100

            self.encoder = img.Encoder(latent_dim).to(self.device)
            self.decoder = img.Decoder(latent_dim).to(self.device)
            self.dis_xz = img.DiscriminatorXZ(latent_dim).to(self.device)
            self.dis_xx = img.DiscriminatorXX(latent_dim).to(self.device)
            self.dis_zz = img.DiscriminatorZZ(latent_dim).to(self.device)

            self.train_loader, self.valid_loader, self.test_loader, self.number_anomalous_samples = {}, {}, {}, {}

            for i, c in enumerate(self.classes):
                self.train_loader[c],  self.valid_loader[c],  self.test_loader[c], self.number_anomalous_samples[c] =\
                    Dataloader(self.dataset, normal_class=i, batch_size=self.batch_size)

        elif dataset == "KDDCup":
            optimizer_kwargs = {"lr": 1e-5}
            latent_dim = 32

            self.encoder = tab.Encoder(latent_dim).to(self.device)
            self.decoder = tab.Decoder(latent_dim).to(self.device)
            self.dis_xz = tab.DiscriminatorXZ(latent_dim).to(self.device)
            self.dis_xx = tab.DiscriminatorXX(latent_dim).to(self.device)
            self.dis_zz = tab.DiscriminatorZZ(latent_dim).to(self.device)

            self.train_loader, self.valid_loader, self.test_loader, self.number_anomalous_samples = \
                Dataloader(self.dataset, batch_size=self.batch_size)

        else:
            raise NameError("No valid dataset.")

        self.latent_dim = latent_dim
        self.encoder_optimizer = optimizer(self.encoder.parameters(), betas=betas, **optimizer_kwargs)
        self.decoder_optimizer = optimizer(self.decoder.parameters(), betas=betas, **optimizer_kwargs)
        self.dis_xz_optimizer = optimizer(self.dis_xz.parameters(), betas=betas, **optimizer_kwargs)
        self.dis_xx_optimizer = optimizer(self.dis_zz.parameters(), betas=betas, **optimizer_kwargs)
        self.dis_zz_optimizer = optimizer(self.dis_zz.parameters(), betas=betas, **optimizer_kwargs)

        self.BCEWithLogits = nn.BCEWithLogitsLoss().to(self.device)
        self.BCEWithLogitsReductionNone = nn.BCEWithLogitsLoss(reduction='none').to(self.device)
        self.l1 = nn.L1Loss(reduction='none').to(self.device)

    def save_networks(self, normal_sample=None):
        if self.dataset == "CIFAR10":
            path = f'{self.network_dir}/{normal_sample}.pth'
        elif self.dataset == "KDDCup":
            path = f'{self.network_dir}/kdd.pth'
        else:
            raise NameError("No valid dataset.")
        torch.save({'Generator': self.decoder.state_dict(),
                    'Encoder': self.encoder.state_dict(),
                    'DiscriminatorXZ': self.dis_xz.state_dict(),
                    'DiscriminatorXX': self.dis_xx.state_dict(),
                    'DiscriminatorZZ': self.dis_zz.state_dict()}, path)

    def load_networks(self, normal_sample=None):
        """Load weights."""
        if self.dataset == "CIFAR10":
            path = f'{self.network_dir}/{normal_sample}.pth'
        elif self.dataset == "KDDCup":
            path = f'{self.network_dir}/kdd.pth'
        else:
            raise NameError("No valid dataset.")
        state_dict = torch.load(path)

        self.decoder.load_state_dict(state_dict['Generator'])
        self.encoder.load_state_dict(state_dict['Encoder'])
        self.dis_xz.load_state_dict(state_dict['DiscriminatorXZ'])
        self.dis_xx.load_state_dict(state_dict['DiscriminatorXX'])
        self.dis_zz.load_state_dict(state_dict['DiscriminatorZZ'])

    def weights_init(self):
        if self.dataset == "CIFAR10":
            self.encoder.apply(weights_init)
            self.decoder.apply(weights_init)
            self.dis_xz.apply(weights_init)
            self.dis_xx.apply(weights_init)
            self.dis_zz.apply(weights_init)
        elif self.dataset == "KDDCup":
            self.encoder.apply(xavier_init)
            self.decoder.apply(xavier_init)
            self.dis_xz.apply(xavier_init)
            self.dis_xx.apply(xavier_init)
            self.dis_zz.apply(xavier_init)

    def _forward(self, x):
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

        # Calculate losses

        # discriminator xz
        loss_dis_enc = self.BCEWithLogits(out_dis_x_z_gen, torch.ones_like(out_dis_x_z_gen))
        loss_dis_gen = self.BCEWithLogits(out_dis_xgen_z, torch.zeros_like(out_dis_xgen_z))

        dis_loss_xz = loss_dis_gen + loss_dis_enc

        # discriminator xx
        x_real_dis = self.BCEWithLogits(x_logit_real, torch.ones_like(x_logit_real))
        x_fake_dis = self.BCEWithLogits(x_logit_fake, torch.zeros_like(x_logit_fake))
        dis_loss_xx = x_real_dis + x_fake_dis

        # discriminator zz
        z_real_dis = self.BCEWithLogits(z_logit_real, torch.ones_like(z_logit_real))
        z_fake_dis = self.BCEWithLogits(z_logit_fake, torch.zeros_like(z_logit_fake))
        dis_loss_zz = z_real_dis + z_fake_dis

        # decoder and encoder
        gen_loss_xz = self.BCEWithLogits(out_dis_xgen_z, torch.ones_like(out_dis_xgen_z))
        enc_loss_xz = self.BCEWithLogits(out_dis_x_z_gen, torch.zeros_like(out_dis_x_z_gen))

        x_real_gen = self.BCEWithLogitsReductionNone(x_logit_real, torch.zeros_like(x_logit_real))
        x_fake_gen = self.BCEWithLogitsReductionNone(x_logit_fake, torch.ones_like(x_logit_fake))
        z_real_gen = self.BCEWithLogitsReductionNone(z_logit_real, torch.zeros_like(z_logit_real))
        z_fake_gen = self.BCEWithLogitsReductionNone(z_logit_fake, torch.ones_like(z_logit_fake))

        cost_x = torch.mean(x_real_gen + x_fake_gen)
        cost_z = torch.mean(z_real_gen + z_fake_gen)

        cycle_consistency_loss = cost_x + cost_z
        loss_generator = gen_loss_xz + cycle_consistency_loss
        loss_encoder = enc_loss_xz + cycle_consistency_loss

        return dis_loss_xz, dis_loss_xx, dis_loss_zz, loss_generator, loss_encoder, cycle_consistency_loss

    def step(self, x):
        """Performs a single step of ALAD training.
        Args:
          x: data points used for training."""

        self.dis_xz_optimizer.zero_grad()
        self.dis_xx_optimizer.zero_grad()
        self.dis_zz_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        dis_loss_xz, dis_loss_xx, dis_loss_zz, loss_generator, loss_encoder, cycle_consistency_loss = self._forward(x)

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

        # If verbose is True, then save losses of training.
        if self.verbose:
            self.writer.add_scalar("discriminator_xz loss", dis_loss_xz.item(), self.step_id)
            self.writer.add_scalar("discriminator_xx loss", dis_loss_xx.item(), self.step_id)
            self.writer.add_scalar("discriminator_zz loss", dis_loss_zz.item(), self.step_id)
            self.writer.add_scalar("generator loss", loss_generator.item(), self.step_id)
            self.writer.add_scalar("encoder loss", loss_encoder.item(), self.step_id)
            self.writer.add_scalar("cycle consistency loss", cycle_consistency_loss, self.step_id)

    def valid_step(self, x):
        """Performs a single step of ALAD validation.
        Args:
          data: data points used for validation."""
        with torch.no_grad():
            dis_loss_xz, dis_loss_xx, dis_loss_zz, loss_generator, loss_encoder, cycle_consistency_loss = \
                self._forward(x)

            self.writer.add_scalar("valid_discriminator_xz loss", dis_loss_xz.item(), self.step_id)
            self.writer.add_scalar("valid_discriminator_xx loss", dis_loss_xx.item(), self.step_id)
            self.writer.add_scalar("valid_discriminator_zz loss", dis_loss_zz.item(), self.step_id)
            self.writer.add_scalar("valid_cycle_consistency_loss", cycle_consistency_loss, self.step_id)
            self.writer.add_scalar("valid_generator_loss", loss_generator.item(), self.step_id)
            self.writer.add_scalar("valid_encoder_loss", loss_encoder.item(), self.step_id)

    def train(self):
        if self.dataset == "CIFAR10":
            # each label should be one time the normal sample whereas all other are anomalous
            for normal_i, normal_class in enumerate(self.classes):
                self.writer = SummaryWriter(f"{self.network_dir}/{normal_class}")
                self.epoch_id = 0
                self.step_id = 0
                self.weights_init()

                print("-"*100)
                print('-'*24, f'Start training with class {normal_class} as normal class', '-'*24)
                print("-"*100)

                for epoch_id in tqdm(range(self.max_epochs)):
                    self.epoch_id = epoch_id
                    if self.valid_loader[normal_class] is not None:
                        valid_iter = iter(self.valid_loader[normal_class])

                    for (data, _) in self.train_loader[normal_class]:
                        self.step(data.to(self.device))
                        if self.valid_loader[normal_class] is not None and self.step_id % self.report_interval == 0:
                            try:
                                vdata, _ = next(valid_iter)
                            except StopIteration:
                                valid_iter = iter(self.valid_loader[normal_class])
                                vdata, _ = next(valid_iter)
                            self.valid_step(vdata.to(self.device))
                        self.step_id += 1

                self.save_networks(normal_class)

        elif self.dataset == "KDDCup":
            self.writer = SummaryWriter(f"{self.network_dir}/")
            self.epoch_id = 0
            self.step_id = 0
            self.weights_init()

            print("-" * 100)
            print('-' * 29, f'Start training with dataset KDDCup99', '-' * 29)
            print("-" * 100)
            for epoch_id in tqdm(range(self.max_epochs)):
                self.epoch_id = epoch_id
                if self.valid_loader is not None:
                    valid_iter = iter(self.valid_loader)

                for (data, _) in self.train_loader:
                    self.step(data.to(self.device))
                    if self.valid_loader is not None and self.step_id % self.report_interval == 0:
                        try:
                            vdata, _ = next(valid_iter)
                        except StopIteration:
                            valid_iter = iter(self.valid_loader)
                            vdata, _ = next(valid_iter)
                        self.valid_step(vdata.to(self.device))
                    self.step_id += 1

            self.save_networks()

    def score_function(self, x):
        x = x.to(self.device)
        _, real_feature = self.dis_xx(x, x)
        rec_x = self.decoder(self.encoder(x))
        _, fake_feature = self.dis_xx(x, rec_x)
        score = self.l1(real_feature, fake_feature)
        return torch.sum(score, dim=1)

    def evaluate(self):
        if self.dataset == "CIFAR10":
            scores = []
            targets = []

            print("Start evaluation...")
            # each label should be one time the anomalous sample
            for normal_i, normal_class in enumerate(self.classes):
                print("Number of anomalous samples: {}".format(self.number_anomalous_samples[normal_class]))
                self.load_networks(normal_class)
                self.encoder.eval()
                self.decoder.eval()
                self.dis_xx.eval()

                batch_targets = []
                batch_scores = []
                with torch.no_grad():
                    for imgs, label in tqdm(self.test_loader[normal_class]):
                        score = self.score_function(imgs)
                        batch_scores.extend(list(score.cpu().detach().numpy()))
                        batch_targets.extend(list(label.cpu().detach().numpy()))

                targets.append(batch_targets)
                scores.append(batch_scores)

            targets = np.array(targets)
            scores = np.array(scores)
            self.plot_roc(scores, targets)

        elif self.dataset == "KDDCup":
            print("Number of anomalous samples: {}".format(self.number_anomalous_samples))
            self.load_networks()
            self.encoder.eval()
            self.decoder.eval()
            self.dis_xx.eval()

            targets = []
            scores = []
            with torch.no_grad():
                for imgs, label in tqdm(self.test_loader):
                    score = self.score_function(imgs)
                    scores.extend(list(score.cpu().detach().numpy()))
                    targets.extend(list(label.cpu().detach().numpy()))

            self.plot_pr(scores, targets)

    def plot_roc(self, y_scores, y_labels):
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

        for normal_i, normal_class in enumerate(self.classes):
            scores = y_scores[normal_i].astype(np.float)
            targets = y_labels[normal_i].astype(np.uint8)

            roc_auc[normal_i] = roc_auc_score(targets, scores)
            print('ROC AUC score of class {}: {:.2f}'.format(normal_class, roc_auc[normal_i] * 100))
            fpr, tpr, _ = roc_curve(targets, scores, pos_label=1)

            line, = ax.plot(
                fpr,
                tpr,
                lw=lw,
                label="%s (AUC = %0.2f)" % (normal_class, roc_auc[normal_i])
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

    def plot_pr(self, y_score, y_true):
        display = PrecisionRecallDisplay.from_predictions(y_true, y_score, name="Features KDDCup99")
        _ = display.ax_.set_title("Precision-Recall Curve")
        plt.show()

        # Transform contiguous score into binary score.
        # Search for threshold to distinguish between normal and anomalous. Note: The closer the
        # score to zero, the more probable the sample stems from the learned normal distribution.
        # Vice versa for anomalous samples.

        threshold = np.sort(y_score)[-self.number_anomalous_samples]
        y_score = np.where(y_score <= threshold, 0.0, 1.0)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_score)
        print("Precision: {:.4f} \t Recall: {:.4f} \t F1-Score: {:.4f}".format(
            np.mean(precision), np.mean(recall), np.mean(f1)))

        display = PrecisionRecallDisplay.from_predictions(y_true, y_score, name="Features KDDCup99")
        _ = display.ax_.set_title("Precision-Recall Curve (Binary scores)")
        plt.show()



