######################
###     IMPORTS    ###
######################
import os
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

########################
###  PLOT FUNCTIONS  ###
########################
def TrainingSnapShot(generator, epoch, test_input, save_path):
    predictions = generator(test_input, training=False)
    predictions = np.squeeze(predictions)
    fig = plt.figure(figsize=(5, 5))

    for i in range(predictions.shape[0]):
        plt.subplot(3, 3, i+1)
        plt.imshow(predictions[i], cmap="gnuplot2", vmin=0, aspect='auto')
        #plt.imshow(predictions[i][0], cmap="gnuplot2", vmin=0)
        plt.axis('off')

    fig.suptitle(f"Epoch {epoch}")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(os.path.join(save_path, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.close()


def NonAdversarialLossComponentPlot(losshist, save_path):
    # Plot Style
    #plt.style.use('fivethirtyeight')
    colors = ["#fc4f30","#30a2da", "#e5ae38", "#6d904f", "#8b8b8b"]
    titles = [r"Average Training/Validation Image Non-Adversarial Loss (Not $\alpha$ corrected)", r"Huber Validation loss (Corrected for $\alpha$)",
                r"Statistics Validation loss (Corrected for $\alpha$)", r"Aperture/Peak Flux Validation loss (Corrected for $\alpha$)"]
    labels = [r"Log10(Loss) ($SNR\geq 5, \ \delta = 5\sigma$)", r"LogCosh Loss", r"Loss", r"Aperture/Peak Flux Loss ($SNR\geq 5$)"]
    c_idx = 0
    # Plot Body
    fig = plt.figure(figsize=(14,12))
    fig.suptitle("Unfiltered Loss")

    gs = GridSpec(2, 2, hspace=.3, wspace=0.2)

    # Compute alpha corrected loss components
    Lh_alph = losshist["Lh"]/losshist['alpha']
    Lflux_alph = losshist["Lflux"]/(1 - losshist['alpha'])
    Ltot = losshist["Lh"] + losshist["Lstats"] + losshist["Lflux"]
    losses = [Ltot, Lh_alph, losshist["Lstats"], Lflux_alph]

    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(gs[i, j])
            if i == 0 and j == 0:
                ax.set_title(titles[c_idx], fontsize=15)

                ax.plot(losshist["train_epochs"], np.log10(Ltot/losshist['Nvalid']), lw=2, label=rf"Validation Loss ($\alpha = ${losshist['alpha']})", color=colors[c_idx], alpha=1)
                ax.plot(losshist["train_epochs"], np.log10(losshist['train_loss']/losshist['Ntrain']), lw=2, label=rf"Training Loss ($\alpha = ${losshist['alpha']})", color=colors[c_idx], alpha=0.5)

                ax.hlines(y = np.min(np.log10(Ltot/losshist['Nvalid'])), xmin = losshist["train_epochs"][0], xmax = losshist["train_epochs"][-1], linestyle="dashed", label = f"Minimum loss: {np.min(np.log10(Ltot/losshist['Nvalid'])):.2f}", color=colors[c_idx], lw=2, alpha=1)
                ax.hlines(y = np.min(np.log10(losshist['train_loss']/losshist['Ntrain'])), xmin = losshist["train_epochs"][0], xmax = losshist["train_epochs"][-1], linestyle="dashed", label = f"Minimum loss: {np.min(np.log10(losshist['train_loss']/losshist['Ntrain'])):.2f}", color=colors[c_idx], lw=2, alpha=0.5)

            else:
                ax.set_title(titles[c_idx], fontsize=15)
                ax.plot(losshist["train_epochs"], losses[c_idx], lw=2, label=rf"$\alpha = ${losshist['alpha']}", color=colors[c_idx], alpha=1)
                ax.hlines(y = losses[c_idx][losshist["epoch_best_model_save"]], xmin = losshist["train_epochs"][0], xmax = losshist["train_epochs"][-1], linestyle="dashed", label = f"loss (Saved Model): {losses[c_idx][losshist['epoch_best_model_save']]:.2f}", color=colors[c_idx], lw=2, alpha=1)

            ax.set_xlabel(r"Epoch number")
            ax.set_ylabel(labels[c_idx])
            ax.legend()

            c_idx += 1

    plt.savefig(os.path.join(save_path, "NonAdversarialUnfilteredLossHistory.png"), dpi=450)
    plt.close()



def NonAdversarialFilteredLossComponentPlot(losshist, save_path):
    # Plot Style
    #plt.style.use('fivethirtyeight')
    colors = ["#fc4f30","#30a2da", "#e5ae38", "#6d904f", "#8b8b8b"]
    titles = [r"Average Training/Validation Image Loss (Not $\alpha$ corrected)", r"Huber Validation loss (Corrected for $\alpha$)",
                r"Statistics Validation loss (Corrected for $\alpha$)", r"Aperture Flux Validation loss (Corrected for $\alpha$)"]
    labels = [r"Loss ($SNR\geq 5, \ \delta = 5\sigma$)", r"Huber Loss ($\delta = 5\sigma$)", r"Loss", r"Aperture Flux Loss ($SNR\geq 5$)"]
    c_idx = 0
    # Plot Body
    fig = plt.figure(figsize=(14,12))
    fig.suptitle("Filtered Loss")
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.2)

    # Compute alpha corrected loss components
    Lh_alph = losshist["Lh"]/losshist['alpha']
    Lflux_alph = losshist["Lflux"]/(1 - losshist['alpha'])
    Ltot_alph = Lh_alph + losshist["Lstats"] + Lflux_alph
    Ltot = losshist["Lh"] + losshist["Lstats"] + losshist["Lflux"]
    losses = [Ltot, Lh_alph, losshist["Lstats"], Lflux_alph]

    N = 10
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(gs[i, j])
            if i == 0 and j == 0:
                ax.set_title(titles[c_idx], fontsize=15)
                ax.plot(losshist["train_epochs"][N-1:],  np.convolve(Ltot/losshist['Nvalid'], np.ones(N)/N, mode='valid'), lw=2, label=rf"$\alpha = ${losshist['alpha']}", color=colors[c_idx], alpha=1)
                ax.plot(losshist["train_epochs"][N-1:], np.convolve(losshist['train_loss']/losshist['Ntrain'], np.ones(N)/N, mode='valid'), lw=2, label=rf"$\alpha = ${losshist['alpha']}", color=colors[c_idx], alpha=0.5)

                ax.hlines(y = np.min(np.convolve(Ltot/losshist['Nvalid'], np.ones(N)/N, mode='valid')), xmin = losshist["train_epochs"][N-1:][0], xmax = losshist["train_epochs"][N-1:][-1], linestyle="dashed", label = f"Minimum loss: {np.min(Ltot/losshist['Nvalid']):.2f}", color=colors[c_idx], lw=2, alpha=1)
                ax.hlines(y = np.min(np.convolve(losshist['train_loss']/losshist['Ntrain'], np.ones(N)/N, mode='valid')), xmin = losshist["train_epochs"][N-1:][0], xmax = losshist["train_epochs"][N-1:][-1], linestyle="dashed", label = f"Minimum loss: {np.min(losshist['train_loss']/losshist['Ntrain']):.2f}", color=colors[c_idx], lw=2, alpha=0.5)

            else:
                ax.set_title(titles[c_idx], fontsize=15)
                ax.plot(losshist["train_epochs"][N-1:], np.convolve(losses[c_idx], np.ones(N)/N, mode='valid'), lw=2, label=rf"$\alpha = ${losshist['alpha']}", color=colors[c_idx], alpha=1)
                ax.hlines(y = losses[c_idx][losshist["epoch_best_model_save"]], xmin = losshist["train_epochs"][N-1:][0], xmax = losshist["train_epochs"][N-1:][-1], linestyle="dashed", label = f"loss (Saved Model): {losses[c_idx][losshist['epoch_best_model_save']]:.2f}", color=colors[c_idx], lw=2, alpha=1)

            ax.set_xlabel(r"Epoch number")
            ax.set_ylabel(labels[c_idx])
            ax.legend()

            c_idx += 1

    plt.savefig(os.path.join(save_path, "NonAdversarialFilteredLossHistory.png"), dpi=450)
    plt.close()


def AdversarialLossComponentPlot(losshist, save_path):
    # Plot Style
    #plt.style.use('fivethirtyeight')
    colors = ["#fc4f30","#30a2da", "#e5ae38", "#6d904f", "#8b8b8b"]
    titles = [r"Generator Validation Loss", r"Critic Validation Losses"]
    c_idx = 0
    # Plot Body
    fig = plt.figure(figsize=(14,12))
    fig.suptitle("Unfiltered Loss")

    gs = GridSpec(1, 2, hspace=.3, wspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax1.set_title(titles[0], fontsize=15)
    ax2.set_title(titles[1], fontsize=15)

    ax1.plot(losshist["train_epochs"], losshist["LwG"], lw=2, label=rf"Validation Generated Loss", color=colors[0], alpha=1)
    ax1.hlines(y = losshist["LwG"][losshist["epoch_best_model_save"]], xmin = losshist["train_epochs"][0], xmax = losshist["train_epochs"][-1], linestyle="dashed", label = f"loss (Saved Model): {losshist['LwG'][losshist['epoch_best_model_save']]:.2f}", color=colors[0], lw=2, alpha=1)

    ax2.plot(losshist["train_epochs"], losshist["LwD_fake_score"]+losshist["LwD_real_score"], lw=2, label=rf"Validation Critic Loss", color=colors[1], alpha=1)
    ax2.hlines(y = (losshist["LwD_fake_score"]+losshist["LwD_real_score"])[losshist["epoch_best_model_save"]], xmin = losshist["train_epochs"][0], xmax = losshist["train_epochs"][-1], linestyle="dashed", label = f"loss (Saved Model): {(losshist['LwD_fake_score']+losshist['LwD_real_score'])[losshist['epoch_best_model_save']]:.2f}", color=colors[1], lw=2, alpha=1)
    
    
    ax1.set_xlabel(r"Epoch number")
    ax1.set_ylabel(r"Score")
    ax1.legend()
    ax2.set_xlabel(r"Epoch number")
    ax2.set_ylabel(r"Score")
    ax2.legend()

    plt.savefig(os.path.join(save_path, "AdversarialUnfilteredLossHistory.png"), dpi=450)
    plt.close()


