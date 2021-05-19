"""#importing packages"""

import numpy as np
import pandas as pd
import math

"""#loading the data"""

pose = pd.read_csv('pose_data.csv')
stimuli = pd.read_csv('data.csv')
col_list = ['Est time', 'Interval min', 'Interval max', 'stim_type', 'size', 'start_pos', 'end_pos', 'speed', 'intensity', 'relative_to_mouse', 'distance', 'distance_unit', 'angle_intervals', 'random', 'trigger_pos', 'trigger_pos_unit']
dot = pd.read_csv('stimuli.csv', usecols=col_list, index_col=False)
col_list = ['DLC_resnet50_KI2020_TrainingSep19shuffle1_200000', 'DLC_resnet50_KI2020_TrainingSep19shuffle1_200000.1']
raw = pd.read_csv('raw_data.csv', low_memory=False, usecols=col_list)

"""copy dot for using later"""
dot_copy = dot.copy()

"""fixing the data-frames"""

stimuli.rename(columns={'Time (ms)': 'time'}, inplace=True)
pose.rename(columns={'frame': 'time'}, inplace=True)
stimuli.loc[stimuli['time'] % 2 != 0, 'time'] = stimuli.loc[:]['time']-1
df = pd.DataFrame({'time': range(0, int(pose.tail(1)['time']), 2)})
pose = pd.merge(pose, df, on='time', how="outer")
pose = pose.sort_values('time', ascending=True)
raw = raw.iloc[2:]
raw = raw.reset_index(drop=True)
raw.columns = ['snoutx', 'snouty']
raw.insert(0, 'time', raw.index*2)
raw = raw.apply(pd.to_numeric)
dot = dot.iloc[1:]
dot.rename(columns={'Est time': 'time'}, inplace=True)
dot.loc[dot['time'] % 2 != 0, 'time'] = dot.loc[:]['time']-1

'#filter dataframe for outliers'
pose.reset_index(drop=True, inplace=True)
pose['dhba'] = pose["head_body_angle"].diff(1)
condition = (pose['dhba'] > 0.25) | (pose['dhba'] < (-0.25))
condition.sum()
for i in range(0, len(pose.index)):
   if condition[i]:
       pose.iloc[i-15:i+15, 1:11] = None

""" find the appearance position of the dot"""
dfla = dot.start_pos
dfla = dfla.to_frame()
dfla[['x', 'y']] = dfla.start_pos.str.split(',', expand=True)
dfla['x'] = dfla['x'].str[1:]
dfla['y'] = dfla['y'].str[:-1]
dfla['time'] = dot['time']
dfla = dfla.drop(['start_pos'], axis=1)
dfla = dfla.apply(pd.to_numeric)
angles = dfla.copy()

"""find the angle of the dot from x axis and invert head angle in pose so the subtraction with dot angle is correct"""

dangle = []
for i in angles.index:
    dangle.append(math.atan2(angles['y'][i], angles['x'][i]))
angles['dangle'] = dangle
pose['head_angle'] = pose['head_angle']*(-1)


"""adjust the numbers of dot appearance to match snout values"""

dfla['x'] = dfla['x'] + 0.08
dfla['y'] = dfla['y'] - 0.16
dfla['x'] = dfla['x'] * 195
dfla['y'] = dfla['y'] * 165
dfla.loc[(dfla['x']/195)-0.08 >= 0, 'x'] = dfla.loc[:]['x']+225
dfla.loc[(dfla['x']/195)-0.08 < 0, 'x'] = 225 - abs(dfla.loc[:]['x'])
dfla.loc[(dfla['y']/165)+0.16 >= 0, 'y'] = 175 - abs(dfla.loc[:]['y'])
dfla.loc[(dfla['y']/165)+0.16 < 0, 'y'] = dfla.loc[:]['y'] + 175
dfla['dangle'] = dangle


"""calculate velocity"""

spacevel = pose[['cog_x', 'cog_y']]
velx = pose['cog_x'].diff(1).div(2)
vely = pose['cog_y'].diff(1).div(2)
velxsq = velx**2
velysq = vely**2
velsq = velxsq + velysq
pose['velsq'] = velsq
pose['vel'] = np.sqrt(pose['velsq'])


'#merging the data'

dataframe = pd.merge(pose, stimuli, on='time', how="outer")

'#find the start, end and the number of trials'

dataframe.insert(14, 'trial', 0)
condition1 = (dataframe['ID'] == 'RewardLED') & (dataframe['State'] == 2)
n_start = condition1.sum()
dataframe.loc[condition1, ['trial']] = 1
condition2 = (dataframe['ID'] == 'RewardLED') & (dataframe['State'] == 0)
n_end = condition2.sum()
dataframe.loc[condition2, ['trial']] = 1

'#isolate only the values with (start,end) trials'

pose_1 = dataframe.loc[dataframe['trial'] == 1]
pose_1 = pose_1.sort_values('time', ascending=True)
pose_1 = pose_1.reset_index(drop=True)
if (len(pose_1.index)) % 2 == 1:
    pose_1.drop(pose_1.tail(1).index, inplace=True)
if (len(pose_1.index)) % 2 == 0:
    pose_1.drop(pose_1.tail(2).index, inplace=True)

'#add numbers to the trials'

n = 0
for i in range(1, len(pose_1.index)+1):
    pose_1.loc[n, 'trial'] = i
    n = n+1

'#merging dataframes with trial time-points and pose'

pose_1 = pose_1[['time', 'trial']]
dataframe = dataframe.drop(['trial'], axis=1)
dataframe = pd.merge(dataframe, pose_1, on='time', how="outer")
dataframe.drop_duplicates(subset='time', keep='first', inplace=True)

'#find duration of each trial in dataframe'

dataframe = dataframe.reset_index(drop=True)
a = 1
for i in range(1, n_start):
    dataframe.loc[int(dataframe.loc[dataframe['trial'] == a].index.values): int(dataframe.loc[dataframe['trial'] == a+1].index.values), 'trial'] = i
    a = a+2

'#isolate only the trials'

dataframe.dropna(subset=['trial'], inplace=True)
dataframe.drop(columns={'ID', 'State', 'Value'}, inplace=True)

'#merge dataframe with dot to see which trials contain dot and create a list with dot starts'

dot = dot[['time']]
dot.insert(1, 'dot', 'dot')
start = dot['time'].tolist()
dataframe = pd.merge(dataframe, dot, on='time', how="outer")
dataframe.dropna(subset=['trial'], inplace=True)

'#find all trials with and without dot appearance'
aldots = dataframe.pivot(columns='trial', values='dot')
aldots = aldots.apply(lambda a: a.dropna().reset_index(drop=True))
alltrials = aldots.columns.tolist()
aldots = aldots.dropna(axis=1, how='all', inplace=False)
aldots = aldots.columns.tolist()
without = np.setdiff1d(alltrials, aldots).tolist()

"#duration of whole trials(in observations)"

dur = dataframe.trial.value_counts()
dur = dur.sort_index(ascending=True)

'#merge dataframe with raw to see position of snout after trial start'

snout = pd.merge(dataframe, raw, on='time', how='inner')
snout = snout.loc[(snout['snouty'] < 400)]
snout = snout.reset_index(drop=True)
tc = snout.trial.value_counts()
tc = tc.sort_index(ascending=True)
ser = pd.Series(1, index=tc.index)
items = []
for i in range(1, len(tc.index) + 1):
    for j in range(ser[i], tc[i] + 1):
        s = str(j)
        items.append(s)
duration = pd.DataFrame(items, columns=['Duration'])
duration = duration.apply(pd.to_numeric)
copy = snout.copy()

'#merge snout and copy with duration'

snout = pd.merge(snout, duration, left_index=True, right_index=True)
copy = pd.merge(copy, duration, left_index=True, right_index=True)
copy.set_index('Duration', inplace=True)

'#filter trials without dot appearance'

for i in range(0, len(without)):
    indextrials = snout[snout['trial'] == without[i]].index
    snout.drop(indextrials, inplace=True)

'#filter'
dot = snout.pivot(columns='trial', values='dot')
dot = dot.apply(lambda a: a.dropna().reset_index(drop=True))
dot = dot[aldots]
dot = dot.apply(lambda a: a.dropna().reset_index(drop=True))

'#choose'
dot = dot.dropna(axis=1, how='all', inplace=False)
dots = dot.apply(pd.value_counts)
dots = (dots > 1).sum()
dotkalo = dots[dots == 0]
dotkako = dots[dots != 0]
dotkalo = dotkalo.index.to_list()
dotkako = dotkako.index.to_list()
dot = dot[dotkalo]
wd = dot.columns.tolist()
without2 = np.setdiff1d(aldots, wd).tolist()

"""find trials with a lot of observations"""
tcdot = tc[wd]
tcdot = tcdot[tcdot < 2500]
tcnodot = tc[without]
tcnodot = tcnodot[tcnodot < 2500]
stddot = tcdot.std()
stdnodot = tcnodot.std()
meandot = tcdot.mean()
meannodot = tcnodot.mean()
dotf = tcdot[tcdot < (1.5*stddot) + meandot]
nodotf = tcnodot[tcnodot < (1.5*stdnodot) + meannodot]
dotf = dotf.index.to_list()
nodotf = nodotf.index.to_list()

'#filter snout from trials that there are either too long or have more than one dots'
for i in range(0, len(without2)):
    indextrials = snout[snout['trial'] == without2[i]].index
    snout.drop(indextrials, inplace=True)
for i in range(0, len(nodotf)):
    indextrials = snout[snout['trial'] == nodotf[i]].index
    snout.drop(indextrials, inplace=True)

'#filter snout from trials with not wanted dots'
loca = snout.loc[snout['dot'] == 'dot']
loca = loca.loc[loca['snouty'] < 95]
dottf = loca['trial'].tolist()
for i in range(0, len(dottf)):
    indextrials = snout[snout['trial'] == dottf[i]].index
    snout.drop(indextrials, inplace=True)

without3 = np.setdiff1d(wd, dotf).tolist()
for i in range(0, len(without3)):
    indextrials = snout[snout['trial'] == without3[i]].index
    snout.drop(indextrials, inplace=True)

dotf = [i for i in dotf if i not in dottf]

'#locate the time when every found dot appears after all the filtering'

loce = snout.loc[snout['dot'] == 'dot']
m = loce['Duration'].mean(axis=0)
m = m*2
start = loce['time'].tolist()

'#reset index of snout and then locate the duration of the trial when dot appears'
snout.set_index('Duration', inplace=True)
snout_al = snout.copy()
durati = []
for i in range(0, len(start)):
    indexstart = snout[snout['time'] == start[i]].index
    durati.append(indexstart)

"# find number of observations before dot appearance"
before = durati.copy()
before[:] = [number - 1 for number in before]
before[:] = [number*(-1) for number in before]

"#find number of observations after dot appearance"
tc2 = tc[dotf]
tc2 = tc2 - 1
du = before.copy()
tc2 = tc2.tolist()
after = []
zip_object = zip(tc2, du)
for tc2_i, du_i in zip_object:
    after.append(tc2_i+du_i)

'#convert lists to series'
aft = []
bef = []
for i in range(0, len(after)):
    aft.append(after[i].tolist())
    bef.append(before[i].tolist())

saft = pd.Series((i[0] for i in aft))
sbef = pd.Series((i[0] for i in bef))
items = []
for i in range(0, len(sbef.index)):
    for j in range(sbef[i], saft[i] + 1):
        s = str(j)
        items.append(s)

duration3 = pd.DataFrame(items, columns=['Duration'])
duration3 = duration3.apply(pd.to_numeric)
snout.reset_index(drop=True, inplace=True)
snout = pd.merge(snout, duration3, left_index=True, right_index=True)
snout.set_index('Duration', inplace=True)


'#isolate only the trials(with a certain value its time)'

def find_velocity(dfname):
    cogx = dfname.pivot(columns='trial', values='cog_x')
    cogy = dfname.pivot(columns='trial', values='cog_y')
    cogx = cogx.pow(2)
    cogy = cogy.pow(2)
    cogs = cogx + cogy
    cogs = cogs.transform('sqrt')
    return cogs

def find_change_in_cogy(dfname):
    cogy = dfname.pivot(columns='trial', values='cog_y')
    cogy = cogy.diff(1)
    return cogy


#hbaala = find_change_in_cogy(snout)
#hba = find_velocity(copy)

hbaala = snout.pivot(columns='trial', values='head_body_angle')
hba = copy.pivot(columns='trial', values='head_body_angle')
hba = hba.dropna(how='all', axis=1)
hba = hba.dropna(how='all', axis=0)

hba_without_dot = hba[nodotf]
hba_without_dot = hba_without_dot.dropna(how='all', axis=1)
hba_without_dot = hba_without_dot.dropna(how='all', axis=0)

'#find the percentage of na in each trial'
'#dot'
nad = hbaala.isnull().sum()
nad = len(hbaala.index) - nad
nad = tc - nad
nad = nad*100
nad = nad/tc
nad = nad[nad > 20]
nad = nad.index.tolist()
'#no dot'
nand = hba_without_dot.isnull().sum()
nand = len(hba_without_dot.index) - nand
nand = tc - nand
nand = nand*100
nand = nand/tc
nand = nand[nand > 20]
nand = nand.index.tolist()

hba_dot = hba[aldots]
hba_dot = hba_dot.dropna(how='all', axis=1)
hba_dot = hba_dot.dropna(how='all', axis=0)

'#remove trials with a lot of na from dataframes'

hbaala = hbaala.drop(nad, axis=1)
hbaala = hbaala.dropna(how='all', axis=0)
hba_without_dot = hba_without_dot.drop(nand, axis=1)
hba_without_dot = hba_without_dot.dropna(how='all', axis=0)

'#shape index'
hba_without_dot.index = hba_without_dot.index + hbaala.index[0]+(-1)
hbaala.index = hbaala.index*2
hba_without_dot.index = hba_without_dot.index*2
hba_without_dotc = hba_without_dot.copy()

hbaala.reset_index(inplace=True)
hba_without_dotc.reset_index(drop=True, inplace=True)
a = int(hbaala.loc[hbaala['Duration'] == 0].index.values)
n_hbaala = hbaala.iloc[a-250:a+751, :]
n_hbaala.set_index('Duration', inplace=True)

b = int(hba_without_dotc.tail(1).index.values/3)
if b > 250:
    n_hba_without_dot = hba_without_dotc.iloc[b-250:b+751, :]
    n_hba_without_dot.reset_index(drop=True, inplace=True)
    n_hba_without_dot.index = n_hba_without_dot.index - 250
    n_hba_without_dot.index = n_hba_without_dot.index * 2
    n_hba_without_dot.index.rename('Duration', inplace=True)

else:
    n_hba_without_dot = hba_without_dotc.copy()
    n_hba_without_dot.index = hba_without_dotc.index - 100
    n_hba_without_dot.index.rename('Duration', inplace=True)

hbaala.set_index('Duration', inplace=True)

"""duration of each trial"""
dot = tc[aldots]
dot = dot.to_frame()
nodot = tc[without]
nodot = nodot.to_frame()
dot.reset_index(drop=True, inplace=True)
nodot.reset_index(drop=True, inplace=True)
dot.rename(columns={'trial': 'dot'}, inplace=True)
nodot.rename(columns={'trial': 'nodot'}, inplace=True)
times = pd.merge(dot, nodot, right_index=True, left_index=True, how='outer')
df = times.melt(var_name='groups', value_name='vals')

def space(dfdot,dfnodot):
    space = dfdot.copy()
    space.reset_index(drop=True, inplace=True)
    space['snouty'] = space['snouty'].round(decimals=0)
    space.set_index('snouty', inplace=True)
    space = space[['head_body_angle', 'trial', 'dot']]
    table = space.pivot_table(columns='trial', index='snouty', values='head_body_angle', aggfunc='mean')

    space = dfnodot.copy()
    space.reset_index(drop=True, inplace=True)
    space['snouty'] = space['snouty'].round(decimals=0)
    space.set_index('snouty', inplace=True)
    space = space[['head_body_angle', 'trial', 'dot']]
    table2 = space.pivot_table(columns='trial', index='snouty', values='head_body_angle', aggfunc='mean')
    table2 = table2[nodotf]

    table = table.drop(nad, axis=1)
    table = table.dropna(how='all', axis=0)
    table2 = table2.drop(nand, axis=1)
    table2 = table2.dropna(how='all', axis=0)

    table.iloc[:] = table.iloc[::-1].values
    table2.iloc[:] = table2.iloc[::-1].values

    table.to_excel('space-hba-dot.xlsx')
    table2.to_excel('space-hba-no-dot.xlsx')

space(snout,copy)

"""Export to xlsx"""

hbaala.to_excel('hba-dot-after snout- 0 = dot appearance.xlsx')
hba_without_dot.to_excel('hba-no-dot-after snout.xlsx')

hbaala_abs = hbaala.abs()
hba_without_dot_abs = hba_without_dot.abs()
hbaala_abs.to_excel('abs-hba-dot-after snout- 0 = dot appearance.xlsx')
hba_without_dot_abs.to_excel('abs-hba-no-dot-after snout.xlsx')

n_hbaala.to_excel('hba-dot.xlsx')
n_hba_without_dot.to_excel('hba-no-dot.xlsx')



dot_copy = pd.merge(snout, dfla, on='time', how="inner")
dot_copy = dot_copy[['trial', 'x', 'y']]

def help(dfname,snoutname,dotname,columnname):
    space = dfname.copy()
    space.reset_index(drop=True, inplace=True)
    space2 = space.pivot(columns='trial', values=snoutname)
    space2 = space2.apply(lambda a: a.dropna().reset_index(drop=True))
    dot_copy2 = dot_copy.pivot(columns='trial', values=dotname)
    dot_copy2 = dot_copy2.apply(lambda a: a.dropna().reset_index(drop=True))
    con = pd.concat([dot_copy2]*len(space2.index), ignore_index=True)
    dif = con - space2
    dfd = pd.Series(dif.values.ravel('F'))
    dfd = dfd.dropna()
    pd.Series(dfd).reset_index(drop=True, inplace=True)
    space[columnname] = dfd
    return space
space = help(snout, 'snoutx', 'x', 'dx')
space = help(space, 'snouty', 'y', 'dy')

space['dxsq'] = space['dx']**2
space['dysq'] = space['dy']**2
space['dis'] = space['dxsq'] + space['dysq']
space['dis'] = np.sqrt(space['dis'])
htmp = pd.merge(space, dfla, on='time', how='inner')

def mindis(dfname):
    mindis=dfname.pivot(columns='trial', values='dis')
    mindis = mindis.apply(lambda a: a.dropna().reset_index(drop=True))
    mindist = mindis.min(axis=0)
    ind = []
    for i in mindis.columns:
        ind.append(mindis.loc[mindis[i] == mindist[i]].index)
    inde = []
    for i in range(0, len(ind)):
        inde.append(ind[i].tolist())
    sinde = pd.Series((i[0] for i in inde))
    sinde.index = mindist.index
    for j in mindis.columns:
        mindis.loc[0:sinde[j], j] = (mindis.loc[0:sinde[j], j])*(-1)
    dis = pd.Series(mindis.values.ravel('F'))
    dis = dis.dropna()
    pd.Series(dis).reset_index(drop=True, inplace=True)
    space['dis'] = dis
    return space
space = mindis(space)

"""create dataframe that for each trial contains only values from dot appearance to first min dis from dot"""
align = space.copy()
align['durat'] = snout.index
align = align.loc[align['durat'] >= 0]
align['dis'] = abs(align.dis)
grouped = align.groupby('trial')
md = grouped.apply(lambda g: g[g['dis'] < g['dis'].quantile(.05)])
md.time.sort_values(ascending=True)
md.drop_duplicates(subset=['trial'], keep='first', inplace=True)
timemd = md.time.tolist()

align['new'] = 0
align.loc[align['durat'] == 0, 'new'] = 1
align.loc[align['time'].isin(timemd), 'new'] = 1
align2 = align[align['new'].between(1, 1)]
align2.reset_index(inplace=True, drop=True)
mylist=[]

for i in range(0, len(align2.index), 2):
    mylist.append(list(np.arange(align2['time'][i], align2['time'][i+1], 2)))
flat_list = [item for sublist in mylist for item in sublist]

align3 = align.loc[align.time.isin(flat_list)]
def find_max_value_time(value):
    align4 = align3.loc[align3[value] == align3[value].abs()].copy()
    dhbamax = align4.groupby('trial')[value].max()
    altimes = align4.loc[align4[value].isin(dhbamax), 'time'].tolist()
    return altimes
altimes = find_max_value_time('head_body_angle')

def find_max_response(value,filename):
    align5 = align3.loc[align3[value] == align3[value].abs()].copy()
    resmax = align5.groupby('trial')[value].max()
    resmax.to_excel(filename)
maxres = find_max_response('head_body_angle','max_response_hba.xlsx')

'#reset index of snout_al and then locate the duration of the trial when dot appears'
durati = []
for i in range(0, len(altimes)):
    indexstart = snout_al[snout_al['time'] == altimes[i]].index
    durati.append(indexstart)

"# find number of observations before dot appearance"
before = durati.copy()
before[:] = [number - 1 for number in before]
before[:] = [number*(-1) for number in before]

"#find number of observations after dot appearance"
du = before.copy()
after = []
zip_object = zip(tc2, du)
for tc2_i, du_i in zip_object:
    after.append(tc2_i+du_i)

'#convert lists to series'
aft = []
bef = []
for i in range(0, len(after)):
    aft.append(after[i].tolist())
    bef.append(before[i].tolist())

saft = pd.Series((i[0] for i in aft))
sbef = pd.Series((i[0] for i in bef))
items = []
for i in range(0, len(sbef.index)):
    for j in range(sbef[i], saft[i] + 1):
        s = str(j)
        items.append(s)

duration4 = pd.DataFrame(items, columns=['Duration'])
duration4 = duration4.apply(pd.to_numeric)
snout_al.reset_index(drop=True, inplace=True)
snout_al = pd.merge(snout_al, duration4, left_index=True, right_index=True)
snout_al.set_index('Duration', inplace=True)

hbaala_al = snout_al.pivot(columns='trial', values='head_body_angle')

'#find the percentage of na in each trial'
'#dot'
nad = hbaala_al.isnull().sum()
nad = len(hbaala_al.index) - nad
nad = tc - nad
nad = nad*100
nad = nad/tc
nad = nad[nad > 20]
nad = nad.index.tolist()

'#remove trials with a lot of na from dataframes'

hbaala_al = hbaala_al.drop(nad, axis=1)
hbaala_al = hbaala_al.dropna(how='all', axis=0)

'#shape index'
hbaala_al.index = hbaala_al.index*2

hbaala_al.reset_index(inplace=True)
a = int(hbaala_al.loc[hbaala_al['Duration'] == 0].index.values)
n_hbaala_al = hbaala_al.iloc[a-250:a+751, :]
n_hbaala_al.set_index('Duration', inplace=True)
n_hbaala_al.to_excel('hba-dot_al.xlsx')
hbaala_al.to_excel('hba-dot_all.xlsx')
def outlier(dfname):
    meancol = dfname.mean(axis=0)
    stdcol = dfname.std(axis=0)
    #identify outliers
    cut_off = stdcol * 3
    lower = meancol - cut_off
    upper = meancol + cut_off
    #identify ouliers
    for i in dfname.columns:
        dfname.loc[dfname[i] < lower[i], i] = None
    for i in dfname.columns:
        dfname.loc[dfname[i] > upper[i], i] = None
    return dfname


def wantedtable(value, filename):
    space2 = space.loc[:, (value, 'trial', 'dis')]
    space2.apply(pd.to_numeric)
    space2['dis'] = space2['dis'].round(decimals=0)
    space2.set_index('dis', inplace=True)
    table = space2.pivot_table(columns='trial', index='dis', values=value, aggfunc='mean')
    table.to_excel(filename)
    return table

tablevel = wantedtable('vel','distance-dot-vel.xlsx')
tablehd = wantedtable('head_body_angle', 'distance-dot-hba.xlsx')
tablebdlen = wantedtable('ba_len', 'distance-dot-bdlen.xlsx')

htmp['anghd'] = htmp['head_angle'] - htmp['dangle']
fhtmp = htmp[['trial', 'dis', 'anghd']]

def der(tablename,heatmapname):
    space_dis = outlier(tablename)
    fact = space_dis[-100:+100]
    deriv = fact.diff(1)
    derivfilt = outlier(deriv)
    derivfiltabs = abs(derivfilt)
    dermean = derivfiltabs.mean(axis=0)
    dermax = derivfiltabs.max(axis=0)
    dermean.reset_index(inplace=True, drop=True)
    dermax.reset_index(inplace=True, drop=True)
    dermean = dermean.to_frame()
    dermax = dermax.to_frame()
    dermean.columns = ['dermean']
    dermax.columns = ['dermax']
    fhtmp2 = pd.merge(fhtmp, dermean, right_index=True, left_index=True, how='outer')
    fhtmp2 = pd.merge(fhtmp2, dermax, right_index=True, left_index=True, how='outer')
    fhtmp2.to_excel(heatmapname)

heatacc = der(tablevel,'heatmap-acc.xlsx')
heathd = der(tablehd, 'heatmap-hd.xlsx')
heatbdlen = der(tablebdlen, 'heatmap-bdlen.xlsx')