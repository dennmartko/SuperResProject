import os
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

def confusion_plot(flux_bins, C, R, save):
    plt.clf()
    C_color = "#1e1e1e"
    R_color = "#3399FF"

    fig = plt.figure(figsize=(9, 5), frameon=False, facecolor='white')
    ax1 = fig.add_subplot(111)
    
    # Gridline Styling
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, color='0.65', ls='-', lw=1.5, zorder=0)

    # Completeness and Reliability line plots
    flux = np.array([((bin[0] + bin[1])/2) for bin in flux_bins])
    ax1.plot(flux*1000, C, color=C_color, alpha=0.9, label='Completeness', marker='o', lw=2, fillstyle='none')
    ax1.plot(flux*1000, R, color=R_color, alpha=0.9, label='Reliability', marker='o', lw=2, fillstyle='none')

    ax1.set_xlabel("Source Flux (mJy/beam)")

    # Plot Styling
    ax1.set_yticks(np.arange(0, 1.2, 0.2))
    ax1.set_xticks(np.arange(0, flux[-1]*1000, 2))
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')

    ax1.tick_params(axis='y', length=0)
    for tick_label in ax1.yaxis.get_ticklabels():
        tick_label.set_fontsize(12)
    for tick_label in ax1.xaxis.get_ticklabels():
        tick_label.set_fontsize(12)
        tick_label.set_horizontalalignment('center')
    ax1.tick_params(direction='in', axis='x', length=7, color='0.1')

    # Display analogy completeness and reliability
    ax1.annotate(u'Completeness: Accuracy', xy=(0.09, 0.9),
                xycoords='figure fraction', size=12, color=C_color,
                fontstyle='italic')
    ax1.annotate(u'Reliability: Precision', xy=(0.75, 0.9),
                xycoords='figure fraction', size=12, color=R_color,
                fontstyle='italic')
    ax1.text(x=-1.5, y=1.15, s="Confusion Plot", ha='left', fontsize=13, weight='bold', alpha=.8)
    # Save the plot
    fig.savefig(save, edgecolor='none', dpi=350, transparent=True)

# def completeness_plot(aper_flux_bins, c, α, save):
#     color = "#30a2da"
#     plt.figure(figsize=(7,7))
#     plt.plot(np.array([((bin[0] + bin[1])/2) for bin in aper_flux_bins]), c, color=color, alpha=0.9, label=rf"$\alpha = ${α}", marker="s", lw=2)
#     plt.title("Completeness (SNR>=5) - Matched generated sources", fontsize=15)
#     plt.xlabel("Peak Flux (Jy/beam)", fontsize=15)
#     plt.ylabel("Completeness", fontsize=15)
#     plt.legend()
#     plt.savefig(save)
#     plt.close()

# def reliability_plot(aper_flux_bins, r, α, save):
#     color = "#30a2da"
#     plt.figure(figsize=(7,7))
#     plt.plot(np.array([((bin[0] + bin[1])/2) for bin in aper_flux_bins]), r, color=color, alpha=0.9, label=rf"$\alpha = ${α}", marker="s", lw=2)
#     plt.title("Reliability (SNR >= 1) - Matched real sources", fontsize=15)
#     plt.xlabel("Peak Flux (Jy/beam)", fontsize=15)
#     plt.ylabel("Reliability", fontsize=15)
#     plt.legend()
#     plt.savefig(save)
#     plt.close()


def comparison_plot(gen_output, target, save):
    plt.clf()
    fig = plt.figure(figsize=(14,14))
    fig.suptitle("Comparison generated-validation images (@500microns, 1'', 7.9'' FWHM)")
    gs = GridSpec(4, 2, hspace=.4, wspace=0.1)
    for i in range(2):
        for j in range(4):
            ax = fig.add_subplot(gs[j, i])
            if i == 0:
                ax.imshow(gen_output[j], aspect="auto", cmap="gnuplot2", vmin = 0)
                ax.set_title("Generated")
            else:
                ax.imshow(target[j], aspect="auto", cmap="gnuplot2", vmin = 0)
                ax.set_title("Validation images")


    plt.savefig(save)
    plt.close()


def hexplot(target_galx_fluxes, gen_galx_fluxes, target_galx_aperfluxes, gen_galx_aperfluxes, save1, save2):
    plt.clf()
    # The peakflux hexplot
    fig = plt.figure(figsize=(7,7))
    gs = GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[95,5])
    
    ax1 = plt.subplot(gs[1,0])
    ax2 = plt.subplot(gs[0,0], sharex = ax1)
    ax3 = plt.subplot(gs[1, 1])

    hb1 = ax1.hexbin(np.array(target_galx_fluxes)*1000, np.array(gen_galx_fluxes)*1000, gridsize=(250, 250), cmap='turbo', mincnt=1)
    cb = fig.colorbar(hb1, cax = ax3, label="Number of galaxies") # Check logarithm
    lim = [np.round(np.min(np.array(gen_galx_fluxes)*1000, 0)),20]
    ax1.set_ylim(lim)
    ax1.set_xlim(lim)

    # Set straight line = true correlation
    ytrue = np.linspace(lim[0], lim[1], 100)
    ax1.plot(ytrue, ytrue, color='red', alpha=0.7, label='True correlation')
    ax1.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, loc='lower right')

    # Set-up sample points along straight line
    samples = np.unique(hb1.get_offsets()[:,0])
    counts = hb1.get_array()
    positions = hb1.get_offsets()
    mae_x = []
    mae_y = []
    Sdiff_rel = []
    
    for idx, sample in enumerate(samples):
        if idx % 5 == 0:
            tmp_l = []
            mask = np.where(positions[:,0] == sample)[0]
            for cnt_idx, cnt in enumerate(counts[mask]):
                tmp_l += [positions[mask,1][cnt_idx] if len(counts[mask]) > 1 else positions[mask,1] for i in range(int(cnt))]
            if len(tmp_l) >= 3:
                mae_x.append(sample)
                mae_y.append(1/len(tmp_l) * np.sum(abs(sample - np.array(tmp_l))))
                Sdiff_rel.append(np.mean(np.abs(sample - np.array(tmp_l))/sample)) # <(S_in - S_out)/S_in>

    ax2.set_ylabel(r"$\left<\left(S_{in} - S_{out}\right)/S_{in}\right>$", labelpad=15)
    #ax2.scatter(mae_x, mae_y, marker='^', s=10, color='red', alpha=0.7)
    ax2.plot(mae_x, Sdiff_rel, alpha=0.9, marker='o', lw=2, fillstyle='none', color='red')
    ax2.yaxis.grid(True, color='0.65', ls='-', lw=1., zorder=0)
    ax2.xaxis.grid(False)

    ax2.set_ylim([0, None])

    ax1.errorbar(mae_x, mae_x, mae_y, lw=2, color='red', capsize=3, capthick=2, alpha=0.35)
    ax1.set_xlabel(r"$S_{in}$ (mJy/beam)")
    ax1.set_ylabel(r"$S_{out}$ (mJy/beam)")

    ax2.text(x=0.05, y=0.95, s="Correlation Plot*", ha='left', fontsize=13, weight='bold', alpha=.8, transform=fig.transFigure)
    ax1.text(x=0.05, y=0.01, s="""*Errorbars signify the MAE""", ha='left', fontsize=9, alpha=.7, transform=fig.transFigure)

    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels()[-1], visible=False)
    plt.subplots_adjust(hspace=0.)
    plt.savefig(save1, dpi=350)
    plt.close()


    # The aperture flux hexplot
    fig = plt.figure(figsize=(7,7))
    gs = GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[95,5])
    
    ax1 = plt.subplot(gs[1,0])
    ax2 = plt.subplot(gs[0,0], sharex = ax1)
    ax3 = plt.subplot(gs[1, 1])

    hb1 = ax1.hexbin(np.array(target_galx_aperfluxes)*1000, np.array(gen_galx_aperfluxes)*1000, gridsize=(200, 200), cmap='turbo', mincnt=1)
    cb = fig.colorbar(hb1, cax = ax3, label="Number of galaxies") # Check logarithm


    # Set straight line = true correlation
    ytrue = np.linspace(lim[0], lim[1], 100)
    ax1.plot(ytrue, ytrue, color='red', alpha=0.8, label='True correlation')
    ax1.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, loc='lower right')

    # Set-up sample points along straight line
    samples = np.unique(hb1.get_offsets()[:,0])
    counts = hb1.get_array()
    positions = hb1.get_offsets()
    mae_x = []
    mae_y = []
    Sdiff_rel = []
    
    for idx, sample in enumerate(samples):
        if idx % 6 == 0:
            tmp_l = []
            mask = np.where(positions[:,0] == sample)[0]
            for cnt_idx, cnt in enumerate(counts[mask]):
                tmp_l += [positions[mask,1][cnt_idx] if len(counts[mask]) > 1 else positions[mask,1] for i in range(int(cnt))]
            if len(tmp_l) >= 3:
                mae_x.append(sample)
                mae_y.append(1/len(tmp_l) * np.sum(abs(sample - np.array(tmp_l))))
                Sdiff_rel.append(np.mean(np.abs(sample - np.array(tmp_l))/sample)) # <(S_in - S_out)/S_in>

    lim = [np.round(np.min(np.array(gen_galx_aperfluxes)*1000, 0)),np.round(positions[-25,1], 0)]
    ax1.set_ylim(lim)
    ax1.set_xlim(lim)

    ax2.set_ylabel(r"$\left<\left(S_{aper, in} - S_{aper, out}\right)/S_{aper, in}\right>$", labelpad=15)
    #ax2.scatter(mae_x, mae_y, marker='^', s=10, color='red', alpha=0.7)
    ax2.plot(mae_x, Sdiff_rel, alpha=0.9, marker='o', lw=2, fillstyle='none', color='red')
    ax2.yaxis.grid(True, color='0.65', ls='-', lw=1., zorder=0)
    ax2.xaxis.grid(False)
    ax2.set_ylim([0, 3])

    ax1.errorbar(mae_x, mae_x, mae_y, lw=2, color='red', capsize=2, capthick=2, alpha=0.3)
    ax1.set_xlabel(r"$S_{aper, in}$ (mJy/beam)")
    ax1.set_ylabel(r"$S_{aper, out}$ (mJy/beam)")

    ax2.text(x=0.05, y=0.95, s="Correlation Plot*", ha='left', fontsize=13, weight='bold', alpha=.8, transform=fig.transFigure)
    ax1.text(x=0.05, y=0.01, s="""*Errorbars signify the MAE""", ha='left', fontsize=9, alpha=.7, transform=fig.transFigure)

    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels()[-1], visible=False)
    plt.subplots_adjust(hspace=0.)
    plt.savefig(save2, dpi=350)
    plt.close()


def insetplot(gen, target, Y_cat, save):
    gen = np.squeeze(gen)
    target = np.squeeze(target)

    # The aperture flux hexplot
    fig = plt.figure(figsize=(11,11))
    gs = GridSpec(2, 2 , height_ratios=[1.5, 1], hspace=0.005)
    
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[0,1])

    ax1.imshow(gen, vmin=0, origin='lower', cmap='gnuplot2')
    ax2.imshow(target, vmin=0, origin='lower', cmap='gnuplot2')

    ax1.set_title("Generated Image", fontsize=13, weight='bold', alpha=.8)
    ax2.set_title("True Image", fontsize=13, weight='bold', alpha=.8)

    # Make a zoom
    y0 = 100
    ysize = 40
    x0 = 60
    xsize = 40
    axins1 = ax1.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins2 = ax2.inset_axes([0.5, 0.5, 0.47, 0.47])
    #sources = Y_cat[np.logical_and(y0 < Y_cat[:,1] < y0+ysize, x0 < Y_cat[:,0] < x0+xsize)]
    sources = Y_cat[np.where((Y_cat[:,-2] < y0+ysize) & (Y_cat[:,-3] < x0+xsize))]
    sources = sources[np.where((sources[:,-2] > y0) & (sources[:,-3] > x0))][:,-4:-1]
    if len(sources) == 0:
        return
    sprof_idx = np.argmax(sources[:,0])

    source_finder = DAOStarFinder(fwhm=7.9, threshold=1/1000)
    gen_sources = source_finder(gen)
    tr_gen = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
    tr_gen = tr_gen[np.where((tr_gen[:,-1] < y0+ysize) & (tr_gen[:,-2] < x0+xsize))]
    tr_gen = tr_gen[np.where((tr_gen[:,-1] > y0) & (tr_gen[:,-2] > x0))]
    if len(tr_gen) == 0:
        return
    # sources[:,0] -= xsize
    # sources[:,1] -= ysize

    axins1.imshow(gen, vmin=0, origin='lower', cmap='gnuplot2')
    axins2.imshow(target, vmin=0, origin='lower', cmap='gnuplot2')
    for idx, s in enumerate(sources):
        if idx == sprof_idx:
            axins1.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='green', linewidths=1, facecolors='none')
            axins2.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='green', linewidths=1, facecolors='none')

            r = np.sqrt((s[1] - tr_gen[:,0])**2 + (s[2] - tr_gen[:,1])**2)
            rmin_idx = np.argmin(r)
            if r[rmin_idx] > 10:
                return
        else:
            axins1.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='red', linewidths=1, facecolors='none')
            axins2.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='red', linewidths=1, facecolors='none')
        

    for idx, s in enumerate(tr_gen):
        peak = gen_sources['peak'][np.where((gen_sources['xcentroid'] == s[0]) & (gen_sources['ycentroid'] == s[1]))]
        if idx == rmin_idx:
            axins1.scatter(s[0], s[1], marker="s", s=peak*10000, color='green', linewidths=1, facecolors='none')
            sgenprof_peak = gen_sources['peak'][np.where((gen_sources['xcentroid'] == s[0]) & (gen_sources['ycentroid'] == s[1]))]
        else:
            axins1.scatter(s[0], s[1], marker="s", s=peak*10000, color='red', linewidths=1, facecolors='none')

    axins1.set_xlim(x0, x0+xsize)
    axins1.set_ylim(y0, y0+ysize)
    axins2.set_xlim(x0, x0+xsize)
    axins2.set_ylim(y0, y0+ysize)
    axins1.set_xticklabels('')
    axins1.set_yticklabels('')
    axins2.set_xticklabels('')
    axins2.set_yticklabels('')

    axins1.spines['bottom'].set_color('gray')
    axins1.spines['top'].set_color('gray') 
    axins1.spines['right'].set_color('gray')
    axins1.spines['left'].set_color('gray')

    axins2.spines['bottom'].set_color('gray')
    axins2.spines['top'].set_color('gray')
    axins2.spines['right'].set_color('gray')
    axins2.spines['left'].set_color('gray')

    ax1.indicate_inset_zoom(axins1, edgecolor='gray')
    ax2.indicate_inset_zoom(axins2, edgecolor='gray')


    # Profile flux of a source
    ## estimate noise level
    _, __, std_noise = sigma_clipped_stats(target, sigma=3.0)

    strueprof_x = sources[sprof_idx][1]
    strueprof_y = sources[sprof_idx][2]
    strueprof_peak = sources[sprof_idx][0]

    if std_noise > strueprof_peak or  std_noise > sgenprof_peak:
        return

    sgenprof_x = tr_gen[rmin_idx][0]
    sgenprof_y = tr_gen[rmin_idx][1]
    
    h_slicetrue = np.arange(int(strueprof_x) - 10, int(strueprof_x) + 10, 1)
    h_slicegen = np.arange(int(sgenprof_x) - 10, int(sgenprof_x) + 10, 1)

    ax3 = plt.subplot(gs[1,:])
    ax3.plot((h_slicegen - int(sgenprof_x))*1, gen[int(sgenprof_y), h_slicegen]*1000, color='green', alpha=0.7, ls='dashed', marker='o', fillstyle='none', label="Generated Profile")
    ax3.plot((h_slicetrue - int(strueprof_x))*1, target[int(strueprof_y), h_slicetrue]*1000, color='green', alpha=0.9, ls='-', marker='o', fillstyle='none', label="True Profile")
    ax3.yaxis.grid(True, color='0.65', ls='-', lw=1., zorder=0)
    ax3.xaxis.grid(True, color='0.65', ls='-', lw=1., zorder=0)
    ax3.hlines(y=std_noise*1000, xmin=(h_slicetrue[0]- int(strueprof_x))*1, xmax=(h_slicetrue[-1]- int(strueprof_x))*1, color='black', ls='dotted', label=rf'Noise $\sigma={std_noise*1000:.1f}$ mJy')
    #ax3.vlines(x = sprof_x, ymin=np.min(target[int(np.round(sprof_y)), h_slice]), ymax=np.max(target[int(np.round(sprof_y)), h_slice]), color='black', ls='dotted')
    ax3.scatter(0, strueprof_peak*1000, marker='x', color='red', label='True Peak Flux', s=15)
    ax3.scatter(0, sgenprof_peak*1000, marker='x', color='green', label='Generated Peak Flux', s=15)
    ax3.set_xlabel("Offset ('')")
    ax3.set_ylabel(r'$S$ (mJy\beam)')

    ax3.set_xlim([(h_slicetrue[0]- int(strueprof_x))*1, (h_slicetrue[-1]- int(strueprof_x))*1])

    ax3.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, loc='upper right')
    ax1.text(x=0.05, y=0.9, s="Comparison Plot", ha='left', fontsize=13, weight='bold', alpha=.8, transform=fig.transFigure)
    ax1.text(x=0.05, y=0.88, s="""Source detection threshold is set at 1mJy""", ha='left', fontsize=10, alpha=.7, transform=fig.transFigure)
    plt.savefig(save, dpi=350)
    plt.close('all')
