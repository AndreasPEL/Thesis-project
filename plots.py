import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from seaborn import heatmap
from pykalman import KalmanFilter
from numpy import ma
from matplotlib import pyplot
import scipy



hba_dot = pd.read_excel('hba-dot-after snout- 0 = dot appearance.xlsx')
hba_dot.set_index('Duration', inplace=True)
# hba_no_dot = pd.read_excel('hba-no-dot-after snout.xlsx')
# hba_no_dot.set_index('Duration', inplace=True)
# hba_dot_abs = pd.read_excel('abs-hba-dot-after snout- 0 = dot appearance.xlsx')
# hba_dot_abs.set_index('Duration', inplace=True)
# hba_no_dot_abs = pd.read_excel('abs-hba-no-dot-after snout.xlsx')
# hba_no_dot_abs.set_index('Duration', inplace=True)
#
# hba_opto = pd.read_excel('hba-task+opto- after snout- 0 = sti start.xlsx')
# hba_opto.set_index('Duration', inplace=True, drop=True)
# hba_opto.drop(['Unnamed: 0'], axis=1, inplace=True)
# hba_no_opto = pd.read_excel('hba-no-stimulation-after snout.xlsx')
# hba_no_opto.set_index('Unnamed: 0', inplace=True)
# hba_opto_abs = pd.read_excel('abs-hba-task+opto-after snout- 0 = sti start.xlsx')
# hba_opto_abs.set_index('Duration', inplace=True)
# hba_no_opto_abs = pd.read_excel('abs-hba-no-stimulation-after snout.xlsx')
# hba_no_opto_abs.set_index('Duration', inplace=True)
#
hba = pd.read_excel('hba.xlsx', index_col='Duration')
hba_no_stim = pd.read_excel('hba-no-stim.xlsx', index_col='Duration')

#
hba_sti = pd.read_excel('hba-sti.xlsx')
hba_sti.set_index('Duration', inplace=True)
hba_no_sti = pd.read_excel('hba-no-sti.xlsx')
hba_no_sti.set_index('Duration', inplace=True)
#
hba_dot = pd.read_excel('hba-dot.xlsx')
hba_dot.set_index('Duration', inplace=True)
hba_no_dot = pd.read_excel('hba-no-dot.xlsx')
hba_no_dot.set_index('Duration', inplace=True)
#
hba_dot_al = pd.read_excel('hba-dot_al.xlsx')
hba_dot_al.set_index('Duration', inplace=True)

hba_dot_all = pd.read_excel('hba-dot_all.xlsx')
hba_dot_all.set_index('Duration', inplace=True)
# space_hba_dot = pd.read_excel('space-hba-dot.xlsx')
# space_hba_dot.set_index('snouty', inplace=True)
# space_hba_no_dot = pd.read_excel('space-hba-no-dot.xlsx')
# space_hba_no_dot.set_index('snouty', inplace=True)
#
# space_hba_sti = pd.read_excel('space-hba-sti.xlsx')
# space_hba_sti.set_index('snouty', inplace=True)
# space_hba_no_sti = pd.read_excel('space-hba-no-sti.xlsx')
# space_hba_no_sti.set_index('snouty', inplace=True)

hbamax = pd.read_excel('max_response_hba.xlsx')

space_hba_dis = pd.read_excel('distance-dot-hba.xlsx')
space_hba_dis.set_index('dis', inplace=True)

space_vel_dis = pd.read_excel('distance-dot-vel.xlsx')
space_vel_dis.set_index('dis', inplace=True)

space_bdlen_dis = pd.read_excel('distance-dot-bdlen.xlsx')
space_bdlen_dis.set_index('dis', inplace=True)


htmp = pd.read_excel('heatmap-hd.xlsx')
htmp['anghd'] = htmp['anghd']*(-1)

htmpalpha = pd.read_excel('heatmap-acc.xlsx')
htmpalpha['anghd'] = htmpalpha['anghd']*(-1)

htmpbdlen = pd.read_excel('heatmap-bdlen.xlsx')
htmpbdlen['anghd'] = htmpbdlen['anghd']*(-1)

def filter(dfname, napercent):

    def filterloners(dfname):
        dfname = dfname.drop(dfname[dfname.std(axis=1).isna()].index)
        return dfname

    dfname = filterloners(dfname)

    def outlier(dfname):
        meancol = dfname.mean(axis=0)
        stdcol = dfname.std(axis=0)
        # identify outliers
        cut_off = stdcol * 3
        lower = meancol - cut_off
        upper = meancol + cut_off
        # identify ouliers
        for i in dfname.columns:
            dfname.loc[dfname[i] < lower[i], i] = None
        for i in dfname.columns:
            dfname.loc[dfname[i] > upper[i], i] = None
        return dfname

    dfname = outlier(dfname)

    def remna(dfname, napercent):
        na = dfname.isna().mean().round(4) * 100
        na = na.loc[na < napercent]
        naf = na.index.tolist()
        dfname = dfname[naf]
        return dfname

    dfname = remna(dfname, napercent)

    return dfname

def transfilter(dfname):
    dfname = dfname.T
    meancol = dfname.mean(axis=0)
    stdcol = dfname.std(axis=0)
    # identify outliers
    cut_off = stdcol * 3
    lower = meancol - cut_off
    upper = meancol + cut_off
    # identify ouliers
    for i in dfname.columns:
        dfname.loc[dfname[i] < lower[i], i] = None
    for i in dfname.columns:
        dfname.loc[dfname[i] > upper[i], i] = None
    return dfname

def outlierder(dfname,colname):
    # calculate summary statistics
    meancol = dfname[colname].mean(axis=0)
    stdcol = dfname[colname].std(axis=0)
    # identify outliers
    upper = 0.2
    # remove outliers
    dfname = dfname.loc[(dfname[colname] < upper)]
    return dfname

def outliercol(dfname,colname):
    # calculate summary statistics
    meancol = dfname[colname].mean(axis=0)
    stdcol = dfname[colname].std(axis=0)
    # identify outliers
    cut_off = stdcol * 3
    lower = meancol - cut_off
    upper = meancol + cut_off
    # remove outliers
    dfname = dfname.loc[(dfname[colname] > lower) & (dfname[colname] < upper)]
    return dfname


def htmp_find_max_responses(df):
    htmp2 = df.loc[df['dermax'] > df.dermax.quantile(.96)]
    return htmp2


def hba_find_max_responses(df):
    hbam = df.loc[df['head_body_angle'] > 1]
    return hbam

hbamaxf = hba_find_max_responses(hbamax)
trials = hbamaxf.trial.tolist()

htmp2 = htmp_find_max_responses(hbamax)
trials2 = htmp2.trial.tolist()

hba_dot_all = hba_dot_all[trials]
hba_dot_all = hba_dot_all*(-1)
trials.remove(164)
hba_dot_all = hba_dot_all[trials]

hba_dot_alf1 = hba_dot_alf.interpolate()

hba_sti2 = hba_sti.interpolate(method='quadratic', limit_direction="both")
hba2 = hba.interpolate(method='quadratic', limit_direction='both')
hba_no_sti = filter(hba_no_sti, 50)
hba_dot_all = filter(hba_dot_all, 50)
hba_sti = filter(hba_sti, 50)
hba = filter(hba, 50)
hba_dotf = filter(hba_dot, 50)
htmpf = outlierder(htmp, 'dermax')
hba_dot_all = hba_dot_all*(-1)
# mean = hba_dot_2.mean(axis=1)
# der = mean.diff(1).div(2)
# der = der.to_frame()
# derf = outliercol(der, 0)
# derf = derf.abs()
# x = derf.max()
# a = derf.loc[derf[0].isin(x)]


def plotmestd1(dfnamed, title, value, ytitle, xtitle):
    plt.style.use('fivethirtyeight')
    plt.plot(dfnamed.index, dfnamed.mean(axis=1), label=value)
    plt.fill_between(dfnamed.index, dfnamed.mean(axis=1) + abs(dfnamed.std(axis=1)),
                     dfnamed.mean(axis=1) - abs(dfnamed.std(axis=1)),
                     color='blue', alpha=0.3)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plotmestd2(dfnamed, dfnamend,title, label1, label2,ytitle,xtitle):
    plt.style.use('fivethirtyeight')
    plt.plot(dfnamed.index, dfnamed.mean(axis=1), label=label1)
    plt.fill_between(dfnamed.index, dfnamed.mean(axis=1) + abs(dfnamed.std(axis=1)),
                     dfnamed.mean(axis=1) - abs(dfnamed.std(axis=1)),
                     color='blue', alpha=0.3)
    plt.axvline(x=0, label='laser start', c='k', linewidth=1)
    plt.axvline(x=1000, label='laser end', c='k', linewidth=1)
    plt.plot(dfnamend.index, dfnamend.mean(axis=1), label=label2)
    plt.fill_between(dfnamend.index, dfnamend.mean(axis=1) + abs(dfnamend.std(axis=1)),
                     dfnamend.mean(axis=1) - abs(dfnamend.std(axis=1)),
                     color='red', alpha=0.3)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.ylim(-2, 1)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('plot2.eps')
    plt.show()


def plotmestd3(dfnamed, dfnamend, dfnamens, title, label1, label2, label3, ytitle,xtitle):
    plt.style.use('fivethirtyeight')
    plt.plot(dfnamed.index, dfnamed.mean(axis=1), label=label1)
    plt.fill_between(dfnamed.index, dfnamed.mean(axis=1) + abs(dfnamed.std(axis=1)),
                     dfnamed.mean(axis=1) - abs(dfnamed.std(axis=1)),
                     color='blue', alpha=0.3)
    plt.axvline(x=0, label='laser start', c='k', linewidth=1)
    plt.axvline(x=1000, label='laser end', c='k', linewidth=1)
    plt.plot(dfnamend.index, dfnamend.mean(axis=1), label=label2)
    plt.fill_between(dfnamend.index, dfnamend.mean(axis=1) + abs(dfnamend.std(axis=1)),
                     dfnamend.mean(axis=1) - abs(dfnamend.std(axis=1)),
                     color='red', alpha=0.3)
    plt.plot(dfnamens.index, dfnamens.mean(axis=1), label=label3, color='green' )
    plt.fill_between(dfnamens.index, dfnamens.mean(axis=1) + abs(dfnamens.std(axis=1)),
                     dfnamens.mean(axis=1) - abs(dfnamens.std(axis=1)),
                     color='green', alpha=0.3)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.ylim(-2, 1)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    #plt.savefig('plot1.eps')

def plotmestd3(dfnamed, dfnamend, dfnamens, title, label1, label2, label3, ytitle,xtitle):
    plt.style.use('fivethirtyeight')
    plt.plot(dfnamed.index, dfnamed.mean(axis=1), label=label1)
    plt.fill_between(dfnamed.index, dfnamed.mean(axis=1) + abs(dfnamed.std(axis=1)),
                     dfnamed.mean(axis=1) - abs(dfnamed.std(axis=1)),
                     color='blue', alpha=0.3)
    plt.axvline(x=0, label='laser start', c='k', linewidth=1)
    plt.plot(dfnamend.index, dfnamend.mean(axis=1), label=label2)
    plt.fill_between(dfnamend.index, dfnamend.mean(axis=1) + abs(dfnamend.std(axis=1)),
                     dfnamend.mean(axis=1) - abs(dfnamend.std(axis=1)),
                     color='red', alpha=0.3)
    plt.plot(dfnamens.index, dfnamens.mean(axis=1), label=label3, color='green' )
    plt.fill_between(dfnamens.index, dfnamens.mean(axis=1) + abs(dfnamens.std(axis=1)),
                     dfnamens.mean(axis=1) - abs(dfnamens.std(axis=1)),
                     color='green', alpha=0.3)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.ylim(-2, 1)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('plot2.svg')
    plt.show()

def plotmerange1(dfnamed, title, value, ytitle, xtitle):
    plt.style.use('fivethirtyeight')
    plt.plot(dfnamed.index, dfnamed.mean(axis=1), label=value)
    plt.fill_between(dfnamed.index, dfnamed.mean(axis=1) + abs(dfnamed.max(axis=1)),
                     dfnamed.mean(axis=1) - abs(dfnamed.min(axis=1)),
                    color ='blue', alpha=0.3)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plotmerange2(dfnamed, dfnamend, title, label1, label2, ytitle, xtitle):
    plt.style.use('fivethirtyeight')
    plt.plot(dfnamed.index, dfnamed.mean(axis=1), label=label1)
    plt.fill_between(dfnamed.index, dfnamed.mean(axis=1) + abs(dfnamed.max(axis=1)),
                     dfnamed.mean(axis=1) - abs(dfnamed.min(axis=1)),
                    color ='blue', alpha=0.3)
    plt.plot(dfnamend.index, dfnamend.mean(axis=1), label=label2)
    plt.fill_between(dfnamend.index, dfnamend.mean(axis=1) + abs(dfnamend.max(axis=1)),
                     dfnamend.mean(axis=1) - abs(dfnamend.min(axis=1)),
                    color ='red', alpha=0.3)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend()
    plt.tight_layout()
    plt.show()



plotmerange1(hba_dot2, 'head-body-angle', 'mean', 'head-body-angle(rad)', 'time(ms)')
plotmestd1(hba_dot_al, 'head-body-angle', 'mean', 'head-body-angle(rad)', 'time(ms)')
plotmerange2(hba, hba_sti, 'head-body-angle', 'laser(no task)', 'laser(task)', 'head-body-angle(rad)', 'time(ms)')
plotmestd2(hba, hba_sti, 'head-body-angle', 'laser(no task)', 'laser(task)', 'head-body-angle(rad)', 'time(ms)')
plotmerange2(hba_sti, hba_dot2, 'head-body-angle', 'stim', 'dot', 'head-body-angle(rad)', 'time(ms)')
plotmestd2(hba_sti, hba_dot_alf2, 'head-body-angle', 'laser', 'dot', 'head-body-angle(rad)', 'time(ms)')
plotmestd1(hba_dotf, 'head-body-angle', 'mean', 'head-body-angle(rad)', 'time(ms)')
plotmestd3(hba_sti2, hba_dot_all, hba_no_sti, 'head-body-angle', 'laser', 'dot','no laser','head-body-angle(rad)', 'time(ms)')



# space_hba_no_dot = space_hba_no_dot*(-1)
# plotmerange(hba_dot, hba_no_dot, 'head-body-angle', 'head-body-angle(rad)', 'time(ms)')
# plotmestd(hba_dot, hba_sti, 'Head-body angle', 'Head-body-angle', 'time(ms)')
# plotmerange(space_hba_dot, space_hba_no_dot, 'head-body-angle', 'head-body-angle', 'space')
# plotmerange(hba_sti, hba_no_sti, 'head-body-angle', 'head-body-angle(rad)', 'time(ms)')

def cropped(dfname,htmpname):
    a = dfname.tail(1).index.tolist()
    distance = np.arange(-int(htmpname['dis'].mean()), a[0]).tolist()
    dfname = dfname.apply(pd.to_numeric)
    space = dfname.loc[dfname.index.isin(distance)]
    return space


space_vel_crop = cropped(space_vel_dis, htmpalpha)
space_hba_crop = cropped(space_hba_dis, htmp)
space_bdlen_crop = cropped(space_bdlen_dis, htmpbdlen)

def plotmerange1d(originaldfname,dfnamed,title,ytitle,xtitle):
    plt.style.use('fivethirtyeight')
    plt.plot(dfnamed.index, dfnamed.mean(axis=1), label='mean')
    plt.fill_between(dfnamed.index, dfnamed.mean(axis=1) + abs(dfnamed.max(axis=1)),
                     dfnamed.mean(axis=1) - abs(dfnamed.min(axis=1)),
                     color='blue', alpha=0.3)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    a = originaldfname.tail(1).index.tolist()
    plt.xticks([-int(htmp['dis'].mean()), 0, a[0]], ['max', 'min', 'max'])
    plt.legend()
    plt.tight_layout()
    plt.show()

plotmerange1d(space_vel_dis,space_vel_crop, 'head-body-angle', 'velocity', 'distance from dot')
plotmerange1d(space_hba_dis,space_hba_crop, 'head-body-angle', 'head_body_angle(rad)', 'distance from dot')
plotmerange1d(space_bdlen_dis,space_bdlen_crop, 'head-body-angle', 'body_length', 'distance from dot')
#plotmerange1(space_balen_dis, 'body-length', 'body-length', 'distance from dot')
#plotall(space_hba_dis)

htmpalpha = outliercol(htmpalpha, 'dermax')
htmpbdlen = outliercol(htmpbdlen, 'dermax')
val = htmpf.index.tolist()
sizevalues = [i*1.5+10 for i in val]
sizevalues.reverse()

def polscat(coldet,htmpname,title,colorname):
    htmpname[coldet] = np.log(htmpname[coldet])
    plt.subplot(projection="polar")
    plt.scatter(htmpname.anghd, htmpname.dis, c=htmpname.dermax, cmap='bwr', s=sizevalues, ec='k')
    plt.yscale('linear')
    plt.colorbar(orientation="vertical", pad=0.1, label=colorname)
    plt.title(title)
    ax = plt.gca()
    ax.set_theta_zero_location('N')
    ax.set_rlabel_position(77.5)
    plt.grid
    plt.show()

polscat('dermax', htmpf, 'Stimulus effect vs position', 'Max(log(Δ(head-body-angle)/Δx)) per trial')
polscat('dermax', htmpalpha, 'Stimulus effect vs position', 'Max(Log(Acceleration/Δx) per trial')
polscat('dermax', htmpbdlen, 'Stimulus effect vs position', 'Max(log(Δ(body-len)/Δx)) per trial')
polscat('dermax', htmp2, 'Stimulus effect vs position', 'Max(log(Δ(head-body-angle)/Δx)) per trial')

htmp['dermax'] = np.log(htmp['dermax'])
htmp2 = htmp.loc[htmp['dermax'] > -3]

val2 = htmp2.index.tolist()
sizevalues2 = [i*1.5+10 for i in val2]
sizevalues2.reverse()

def polscat2(htmpname):
    plt.subplot(projection="polar")
    plt.scatter(htmpname.anghd, htmpname.dis, s=50, c='red', ec='k')
    plt.yscale('linear')
    plt.title('Position of the more effective stimuli')
    #plt.colorbar(orientation="vertical", pad=0.1)
    ax = plt.gca()
    ax.set_theta_zero_location('N')
    ax.set_rlabel_position(77.5)
    plt.grid
    plt.show()

def polscat3():
    plt.subplot(projection="polar")
    plt.scatter(m, s, s=90, c='red', ec='k', alpha=0.75)
    plt.yscale('linear')
    plt.ylim(30)
    plt.title('Position of the more effective stimuli')
    #plt.colorbar(orientation="vertical", pad=0.1)
    ax = plt.gca()
    ax.set_theta_zero_location('N')
    ax.set_rlabel_position(77.5)
    plt.grid
    plt.show()

htmp2 = htmp.loc[htmp['trial'].isin(trials)]
htmp2 = htmp2.loc[htmp2['trial'] < 60]
htmp2['dis'] = htmp2['dis']*0.15
polscat2(htmpalpha)
polscat3()
#sns.swarmplot(x="groups", y="vals", data=df)
#plt.show()
htmp2.drop(htmp2['trial'] == 164, inplace=True)
s = htmp2.dis.mean()
m = htmp2.anghd.mean()

x = htmp.index.tolist()
s_linear = [1.5*n + 10 for n in range(len(x))]
s_linear.reverse()
plt.scatter(x, [0]*len(x), s=s_linear, label='trial')
plt.ylim(0.1, -0.1)
plt.show()