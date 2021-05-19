import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
import scipy

hba3 = pd.read_excel('hba(1276).xlsx')
hba3.set_index('Duration', inplace=True)
hba4 = pd.read_excel('hba(1300).xlsx')
hba4.set_index('Duration', inplace=True)
hba5 = pd.read_excel('hba(1301).xlsx')
hba5.set_index('Duration', inplace=True)
hba6 = pd.read_excel('hba(3909).xlsx')
hba6.set_index('Duration', inplace=True)
hba7 = pd.read_excel('hba(3913).xlsx')
hba7.set_index('Duration', inplace=True)
hba8 = pd.read_excel('hba(1280).xlsx')
hba8.set_index('Duration', inplace=True)


hba3_no = pd.read_excel('hba-no-stim(1276).xlsx')
hba3_no.set_index('Duration', inplace=True)
hba4_no = pd.read_excel('hba-no-stim(1300).xlsx')
hba4_no.set_index('Duration', inplace=True)
hba5_no = pd.read_excel('hba-no-stim(1301).xlsx')
hba5_no.set_index('Duration', inplace=True)
hba6_no = pd.read_excel('hba-no-stim(3909).xlsx')
hba6_no.set_index('Duration', inplace=True)
hba7_no = pd.read_excel('hba-no-stim(3913).xlsx')
hba7_no.set_index('Duration', inplace=True)
hba8_no = pd.read_excel('hba-no-stim(1280).xlsx')
hba8_no.set_index('Duration', inplace=True)

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

hbaall = pd.DataFrame()
list1 = []
for i in range(1, 7):
    list1.append('mean'+str(i))
list2 = [hba3, hba4, hba5, hba6, hba7, hba8]

for i in range(0, 6):
    hbaall[list1[i]] = list2[i].mean(axis=1)


hbaall_no = pd.DataFrame()
list1 = []
for i in range(1, 7):
    list1.append('mean'+str(i))
list2 = [hba3_no, hba4_no, hba5_no, hba6_no, hba7_no, hba8_no]

for i in range(0, 6):
    hbaall_no[list1[i]] = list2[i].mean(axis=1)

hbaall = filter(hbaall, 50)
hbaall_no = filter(hbaall_no, 50)
hbaall_int = hbaall.interpolate(method='quadratic', limit_direction="both")
hbaall_no_int = hbaall_no.interpolate(method='quadratic', limit_direction='both')

def cropped(dfname,htmpname):
    a = dfname.tail(1).index.tolist()
    distance = np.arange(-int(htmpname['dis'].mean()), a[0]).tolist()
    dfname = dfname.apply(pd.to_numeric)
    space = dfname.loc[dfname.index.isin(distance)]
    return space

def plotmestd2(dfnamed,dfnamend, label1, label2,title, ytitle, xtitle):
    plt.style.use('fivethirtyeight')
    plt.plot(dfnamed.index, dfnamed.mean(axis=1), label=label1)
    plt.fill_between(dfnamed.index, dfnamed.mean(axis=1) + abs(dfnamed.std(axis=1)),
                     dfnamed.mean(axis=1) - abs(dfnamed.std(axis=1)),
                     color='blue', alpha=0.3)
    plt.axvline(x=500, c='k', linewidth=1)
    plt.axvline(x=1500, c='k', linewidth=1)
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
    #plt.savefig('hba-sti-6animals.svg')
    plt.show()

plotmestd2(hbaall_int,hbaall_no_int, 'Optogenetic stimulation', 'laser','no laser', 'head-body angle(rad)', 'time(ms)')

def plotmerange2(dfnamed,dfnamend, title, label1, label2, ytitle,xtitle):
    plt.figure()
    plt.style.use('fivethirtyeight')
    plt.plot(dfnamed.index, dfnamed.mean(axis=1), label=label1)
    plt.fill_between(dfnamed.index, dfnamed.mean(axis=1) + abs(dfnamed.max(axis=1)),
                     dfnamed.mean(axis=1) - abs(dfnamed.min(axis=1)),
                     color='blue', alpha=0.3)
    plt.axvline(x=500, c='k', linewidth=1)
    plt.axvline(x=1500, c='k', linewidth=1)
    plt.plot(dfnamend.index, dfnamend.mean(axis=1), label=label2)
    plt.fill_between(dfnamend.index, dfnamend.mean(axis=1) + abs(dfnamend.std(axis=1)),
                     dfnamend.mean(axis=1) - abs(dfnamend.std(axis=1)),
                     color='red', alpha=0.3)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.ylim(-3, 1)
    plt.legend(fontsize=10)
    plt.tight_layout()
    #plt.savefig('hba-sti-no-sti-6animals-range.svg')
    plt.show()


plotmerange2(hbaall_int,hbaall_no_int, 'Optogenetic stimulation', 'laser', 'no laser','head-body angle(rad)', 'time(ms)')

hba1d = pd.read_excel('hba-dot_al.xlsx')
hba1d.set_index('Duration', inplace=True)
hba2d = pd.read_excel('hba-dot_al(1276).xlsx')
hba2d.set_index('Duration', inplace=True)
hba3d = pd.read_excel('hba-dot_al(1280).xlsx')
hba3d.set_index('Duration', inplace=True)
hba4d = pd.read_excel('hba-dot_al(3909).xlsx')
hba4d.set_index('Duration', inplace=True)
hba5d = pd.read_excel('hba-dot_al(3913).xlsx')
hba5d.set_index('Duration', inplace=True)


hba1d_no = pd.read_excel('hba-no-dot.xlsx')
hba1d_no.set_index('Duration', inplace=True)
hba2d_no = pd.read_excel('hba-no-dot(1276).xlsx')
hba2d_no.set_index('Duration', inplace=True)
hba3d_no = pd.read_excel('hba-no-dot(1280).xlsx')
hba3d_no.set_index('Duration', inplace=True)
hba4d_no = pd.read_excel('hba-no-dot(3909).xlsx')
hba4d_no.set_index('Duration', inplace=True)
hba5d_no = pd.read_excel('hba-no-dot(3913).xlsx')
hba5d_no.set_index('Duration', inplace=True)

hbaall_dot = pd.DataFrame()
list1 = []
for i in range(1, 6):
    list1.append('mean' + str(i))
list2 = [hba2d, hba1d, hba3d, hba4d, hba5d]

for i in range(0, 5):
    hbaall_dot[list1[i]] = list2[i].mean(axis=1)

hbaall_dot = hbaall_dot[:751]

hbaall_no_dot = pd.DataFrame()
list1 = []
for i in range(1, 6):
    list1.append('mean' + str(i))
list2 = [hba2d_no, hba1d_no, hba3d_no, hba4d_no, hba5d_no]

for i in range(0, 5):
    hbaall_no_dot[list1[i]] = list2[i].mean(axis=1)

hbaall_no_dot = hbaall_no_dot[100:850]
hbaall_no_dot.index = hbaall_no_dot.index - 200
hbaall_dot = hbaall_dot.interpolate(method='quadratic', limit_direction="both")
hbaall_no_dot = hbaall_no_dot.interpolate(method='quadratic', limit_direction='both')
hbaall_dot = filter(hbaall_dot, 50)
hbaall_no_dot = filter(hbaall_no_dot, 50)
hbaall_dot = hbaall_dot.interpolate(method='quadratic', limit_direction="both")
hbaall_no_dot = hbaall_no_dot.interpolate(method='quadratic', limit_direction='both')

def plotmestd2(dfnamed, dfnamend,title, label1, label2,ytitle,xtitle):
    plt.style.use('fivethirtyeight')
    plt.plot(dfnamed.index, dfnamed.mean(axis=1), label=label1)
    plt.fill_between(dfnamed.index, dfnamed.mean(axis=1) + abs(dfnamed.std(axis=1)),
                     dfnamed.mean(axis=1) - abs(dfnamed.std(axis=1)),
                     color='blue', alpha=0.3)
    plt.plot(dfnamend.index, dfnamend.mean(axis=1), label=label2)
    plt.fill_between(dfnamend.index, dfnamend.mean(axis=1) + abs(dfnamend.std(axis=1)),
                     dfnamend.mean(axis=1) - abs(dfnamend.std(axis=1)),
                     color='red', alpha=0.3)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.ylim(-1, 1)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('hba-dot-no dot-5animals(std).svg')
    plt.show()


def plotmerange2(dfnamed, dfnamend,title, label1, label2,ytitle,xtitle):
    plt.style.use('fivethirtyeight')
    plt.plot(dfnamed.index, dfnamed.mean(axis=1), label=label1)
    plt.fill_between(dfnamed.index, dfnamed.mean(axis=1) + abs(dfnamed.max(axis=1)),
                     dfnamed.mean(axis=1) - abs(dfnamed.min(axis=1)),
                     color='blue', alpha=0.3)
    plt.plot(dfnamend.index, dfnamend.mean(axis=1), label=label2)
    plt.fill_between(dfnamend.index, dfnamend.mean(axis=1) + abs(dfnamend.max(axis=1)),
                     dfnamend.mean(axis=1) - abs(dfnamend.min(axis=1)),
                     color='red', alpha=0.3)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.ylim(-1, 1)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('hba-dot-no dot-5animals(range).svg')
    plt.show()

plotmestd2(hbaall_dot, hbaall_no_dot, 'Visual stimulation', 'dot', 'no dot', 'head-body angle(rad)', 'time(ms)')
plotmerange2(hbaall_dot, hbaall_no_dot, 'Visual stimulation', 'dot', 'no dot', 'head-body angle(rad)', 'time(ms)')

def cropped(dfname):
    distance = np.arange(-150, 150).tolist()
    dfname = dfname.apply(pd.to_numeric)
    space = dfname.loc[dfname.index.isin(distance)]
    return space
hbaall_dot_cr = cropped(hbaall_dot)


def plotmerange1d(dfnamed,title,ytitle,xtitle):
    plt.style.use('fivethirtyeight')
    plt.plot(dfnamed.index, dfnamed.mean(axis=1), label='mean')
    plt.fill_between(dfnamed.index, dfnamed.mean(axis=1) + abs(dfnamed.max(axis=1)),
                     dfnamed.mean(axis=1) - abs(dfnamed.min(axis=1)),
                     color='blue', alpha=0.3)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.xticks([-150, 0, 150], ['max', 'min', 'max'])
    plt.legend()
    plt.tight_layout()
    plt.savefig('dis from dot 5 animals.svg')
    plt.show()


plotmestd2(hbaall_dot, hbaall_no_dot, 'Visual stimulation', 'dot', 'no dot', 'head-body angle(rad)', 'time(ms)')
plotmerange1d(hbaall_dot, 'Visual stimulation', 'body-length', 'distance from dot')

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
    plt.savefig('dot-time aligned with reaction(std) 5 animals.svg')
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
    plt.savefig('dot-time aligned with reaction(range) 5 animals.svg')
    plt.show()

plotmestd1(hbaall_dot, 'Visual stimulation', 'mean', 'head_body_angle(rad)', 'time(ms)')
plotmerange1(hbaall_dot, 'Visual stimulation', 'mean', 'head_body_angle(rad)', 'time(ms)')

fig, axs = plt.subplots(3, 2)

axs[0, 0].plot(hba3.index, hba3.mean(axis=1), 'blue')
axs[0, 0].set_title('1276', fontsize=10)
axs[0, 0].axvline(x=500, c='k', linewidth=1)
axs[0, 0].axvline(x=1500, c='k', linewidth=1)
axs[0, 1].plot(hba4.index, hba4.mean(axis=1), 'red')
axs[0, 1].set_title('1300', fontsize=10)
axs[0, 1].axvline(x=500, c='k', linewidth=1)
axs[0, 1].axvline(x=1500, c='k', linewidth=1)
axs[1, 0].plot(hba5.index, hba5.mean(axis=1), 'blue')
axs[1, 0].set_title('1301', fontsize=10)
axs[1, 0].axvline(x=500, c='k', linewidth=1)
axs[1, 0].axvline(x=1500, c='k', linewidth=1)
axs[1, 1].plot(hba6.index, hba6.mean(axis=1), 'blue')
axs[1, 1].set_title('3909', fontsize=10)
axs[1, 1].axvline(x=500, c='k', linewidth=1)
axs[1, 1].axvline(x=1500, c='k', linewidth=1)
axs[2, 0].plot(hba7.index, hba7.mean(axis=1), 'blue')
axs[2, 0].set_title('3913', fontsize=10)
axs[2, 0].axvline(x=500, c='k', linewidth=1)
axs[2, 0].axvline(x=1500, c='k', linewidth=1)
axs[2, 1].plot(hba8.index, hba8.mean(axis=1), 'red')
axs[2, 1].set_title('1280', fontsize=10)
axs[2, 1].axvline(x=500, c='k', linewidth=1)
axs[2, 1].axvline(x=1500, c='k', linewidth=1)

fig.text(0.5, 0.005, 'time(ms)', ha='center')
fig.text(0.005, 0.5, 'head-body angle(rad)', va='center', rotation='vertical')
fig.suptitle('Optogenetic stimulation (6 animals)', fontsize=14)
fig.tight_layout()
plt.savefig('optogenetic stimulation 6 animals.svg')
plt.show()


for i in list2:
    i.loc[:, :] = i[:751]
for i in list2:
    i.loc[:, :] = i.interpolate(method='quadratic', limit_direction="both")
    i.loc[:, :] = filter(i, 50)
    i.loc[:, :] = i.interpolate(method='quadratic', limit_direction="both")

fig, axs = plt.subplots(3, 2)

axs[0, 0].plot(hba1d.index, hba1d.mean(axis=1), 'blue')
axs[0, 0].set_title('0957', fontsize=10)
axs[1, 0].plot(hba2d.index, hba2d.mean(axis=1), 'blue')
axs[1, 0].set_title('1276', fontsize=10)
axs[2, 0].plot(hba3d.index, hba3d.mean(axis=1), 'blue')
axs[2, 0].set_title('1280', fontsize=10)
axs[0, 1].plot(hba4d.index, hba4d.mean(axis=1), 'red')
axs[0, 1].set_title('3909', fontsize=10)
axs[1, 1].plot(hba5d.index, hba5d.mean(axis=1), 'blue')
axs[1, 1].set_title('3913', fontsize=10)
axs[2, 1].set_visible(False)
fig.text(0.5, 0.005, 'time(ms)', ha='center')
fig.text(0.005, 0.5, 'head-body angle(rad)', va='center', rotation='vertical')
fig.suptitle('Visual stimulation', fontsize=14)
fig.tight_layout()
plt.savefig('Visual stimulation 5 animals.svg')
plt.show()

hba3sti = pd.read_excel('hba-sti.xlsx')
hba3sti.set_index('Duration', inplace=True)
hba6sti = pd.read_excel('hba-sti(1301).xlsx')
hba6sti.set_index('Duration', inplace=True)
hba7sti = pd.read_excel('hba-sti(3913).xlsx')
hba7sti.set_index('Duration', inplace=True)


hba3_no_sti = pd.read_excel('hba-no-sti.xlsx')
hba3_no_sti.set_index('Duration', inplace=True)
hba6_no_sti = pd.read_excel('hba-no-sti(1301).xlsx')
hba6_no_sti.set_index('Duration', inplace=True)
hba7_no_sti = pd.read_excel('hba-no-stim(3913).xlsx')
hba7_no_sti.set_index('Duration', inplace=True)

hbaall_sti = pd.DataFrame()
list1 = []
for i in range(1, 4):
    list1.append('mean' + str(i))
list2 = [hba3sti, hba6sti, hba7sti]

for i in range(0, 3):
    hbaall_sti[list1[i]] = list2[i].mean(axis=1)

hbaall_no_sti = pd.DataFrame()
list1 = []
for i in range(1, 4):
    list1.append('mean' + str(i))
list2 = [hba6_no_sti, hba3_no_sti, hba7_no_sti]

for i in range(0, 3):
    hbaall_no_sti[list1[i]] = list2[i].mean(axis=1)

hbaall_sti = hbaall_sti.interpolate(method='quadratic', limit_direction="both")
hbaall_no_sti = hbaall_no_sti.interpolate(method='quadratic', limit_direction='both')
hbaall_sti = filter(hbaall_sti, 50)
hbaall_no_sti = filter(hbaall_no_sti, 50)
hbaall_sti = hbaall_sti.interpolate(method='quadratic', limit_direction="both")
hbaall_no_sti = hbaall_no_sti.interpolate(method='quadratic', limit_direction='both')


def plotmestd2(dfnamed,dfnamend,title,label1,label2, ytitle, xtitle):
    plt.style.use('fivethirtyeight')
    plt.plot(dfnamed.index, dfnamed.mean(axis=1), label=label1)
    plt.fill_between(dfnamed.index, dfnamed.mean(axis=1) + abs(dfnamed.std(axis=1)),
                     dfnamed.mean(axis=1) - abs(dfnamed.std(axis=1)),
                     color='blue', alpha=0.3)
    plt.axvline(x=0, c='k', linewidth=1)
    plt.axvline(x=1000, c='k', linewidth=1)
    plt.plot(dfnamend.index, dfnamend.mean(axis=1), label=label2)
    plt.fill_between(dfnamend.index, dfnamend.mean(axis=1) + abs(dfnamend.std(axis=1)),
                     dfnamend.mean(axis=1) - abs(dfnamend.std(axis=1)),
                     color='red', alpha=0.3)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(fontsize=10)
    plt.tight_layout()
    #plt.savefig('hba-sti-6animals.svg')
    plt.show()


def plotmerange2(dfnamed,dfnamend, title, label1, label2, ytitle,xtitle):
    plt.figure()
    plt.style.use('fivethirtyeight')
    plt.plot(dfnamed.index, dfnamed.mean(axis=1), label=label1)
    plt.fill_between(dfnamed.index, dfnamed.mean(axis=1) + abs(dfnamed.max(axis=1)),
                     dfnamed.mean(axis=1) - abs(dfnamed.min(axis=1)),
                     color='blue', alpha=0.3)
    plt.axvline(x=0, c='k', linewidth=1)
    plt.axvline(x=1000, c='k', linewidth=1)
    plt.plot(dfnamend.index, dfnamend.mean(axis=1), label=label2)
    plt.fill_between(dfnamend.index, dfnamend.mean(axis=1) + abs(dfnamend.std(axis=1)),
                     dfnamend.mean(axis=1) - abs(dfnamend.std(axis=1)),
                     color='red', alpha=0.3)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(fontsize=10)
    plt.tight_layout()
    #plt.savefig('hba-sti-no-sti-6animals-range.svg')
    plt.show()

plotmestd2(hba3sti,hba3_no_sti, 'Optogenetic stimulation during task', 'laser','no laser', 'head_body_angle(rad)', 'time(ms)')
plotmestd2(hbaall_sti,hbaall_no_sti, 'Optogenetic stimulation during task', 'laser','no laser', 'head_body_angle(rad)', 'time(ms)')
plotmerange2(hbaall_sti, hbaall_no_sti, 'Optogenetic stimulation during task', 'laser', 'no laser', 'head_body_angle(rad)', 'time(ms)')

list2 = [hba3sti, hba6sti, hba7sti]
for i in list2:
    #i.loc[:, :] = filter(i, 50)
    i.loc[:, :] = i.interpolate(method='quadratic', limit_direction='both')


list2 = [ hba3_no_sti, hba6_no_sti, hba7_no_sti]
for i in list2:
    #i.loc[:, :] = filter(i, 50)
    i.loc[:, :] = i.interpolate(method='quadratic', limit_direction='both')


fig, axs = plt.subplots(3, 2)

axs[0, 0].plot(hba3sti.index, hba3sti.mean(axis=1), 'blue')
axs[0, 0].set_title('3909', fontsize=10)
axs[0, 0].axvline(x=0, c='k', linewidth=1)
axs[0, 0].axvline(x=1000, c='k', linewidth=1)
axs[0, 0].fill_between(hba3sti.index, hba3sti.mean(axis=1) + abs(hba3sti.std(axis=1)),
                     hba3sti.mean(axis=1) - abs(hba3sti.std(axis=1)),
                     color='blue', alpha=0.3)
axs[1, 0].plot(hba4sti.index, hba4sti.mean(axis=1), 'blue')
axs[1, 0].set_title('1276', fontsize=10)
axs[1, 0].axvline(x=0, c='k', linewidth=1)
axs[1, 0].axvline(x=1000, c='k', linewidth=1)
axs[1, 0].fill_between(hba4sti.index, hba4sti.mean(axis=1) + abs(hba4sti.std(axis=1)),
                     hba4sti.mean(axis=1) - abs(hba4sti.std(axis=1)),
                     color='blue', alpha=0.3)
axs[2, 0].plot(hba5sti.index, hba5sti.mean(axis=1), 'blue')
axs[2, 0].set_title('1300', fontsize=10)
axs[2, 0].axvline(x=0, c='k', linewidth=1)
axs[2, 0].axvline(x=1000, c='k', linewidth=1)
axs[2, 0].fill_between(hba5sti.index, hba5sti.mean(axis=1) + abs(hba5sti.std(axis=1)),
                     hba5sti.mean(axis=1) - abs(hba5sti.std(axis=1)),
                     color='blue', alpha=0.3)
axs[0, 1].plot(hba6sti.index, hba6sti.mean(axis=1), 'blue')
axs[0, 1].set_title('1301', fontsize=10)
axs[0, 1].axvline(x=0, c='k', linewidth=1)
axs[0, 1].axvline(x=1000, c='k', linewidth=1)
axs[0, 1].fill_between(hba6sti.index, hba6sti.mean(axis=1) + abs(hba6sti.std(axis=1)),
                     hba6sti.mean(axis=1) - abs(hba6sti.std(axis=1)),
                     color='blue', alpha=0.3)
axs[1, 1].plot(hba7sti.index, hba7sti.mean(axis=1), 'blue')
axs[1, 1].set_title('3913', fontsize=10)
axs[1, 1].axvline(x=0, c='k', linewidth=1)
axs[1, 1].axvline(x=1000, c='k', linewidth=1)
axs[1, 1].fill_between(hba7sti.index, hba7sti.mean(axis=1) + abs(hba7sti.std(axis=1)),
                     hba7sti.mean(axis=1) - abs(hba7sti.std(axis=1)),
                     color='blue', alpha=0.3)
axs[2, 1].set_visible(False)
fig.text(0.5, 0.005, 'time(ms)', ha='center')
fig.text(0.005, 0.5, 'head-body angle(rad)', va='center', rotation='vertical')
fig.suptitle('Optogenetic stimulation during task', fontsize=14)
fig.tight_layout()
#plt.savefig('Visual stimulation 5 animals.svg')
plt.show()
