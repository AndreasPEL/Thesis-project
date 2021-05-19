"""#importing packages"""

import numpy as np
import pandas as pd

"""#loading the data"""

pose = pd.read_csv('pose_data.csv')
stimuli = pd.read_csv('data.csv')
col_list = ['DLC_resnet50_KI2020_TrainingSep19shuffle1_200000', 'DLC_resnet50_KI2020_TrainingSep19shuffle1_200000.1']
raw = pd.read_csv('raw_data.csv', low_memory=False, usecols=col_list)

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

'#find start of stimulation'
start = []
stimuli1 = stimuli.loc[stimuli['ID'] == 'Laser2']
stimuli1.reset_index(drop=True, inplace=True)
start.append(stimuli1['time'][0])
for i in stimuli1.index-1:
    if (stimuli1.iloc[i+1]['time']-600) > (stimuli1.iloc[i]['time']):
        start.append(stimuli1.iloc[i+1]['time'])

'#filter dataframes for outliers'
pose.reset_index(drop=True, inplace=True)
copy = pose.copy()
pose['dhba'] = pose["head_body_angle"].diff(-1)
condition = (pose['dhba'] > 0.2) | (pose['dhba'] < (-0.2))
condition.sum()
for i in range(0, len(pose.index)):
    if condition[i]:
        pose.iloc[i-15:i+15, 1:11] = None


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

'#merging the dataframes with trial time-points and pose'

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

'#merge dataframe with raw to see position of snout after trial start'

snout = pd.merge(dataframe, raw, on='time', how='inner')
snout = snout.loc[(snout['snouty'] < 400)]
snout = snout.reset_index(drop=True)
tc = snout.trial.value_counts()
tc = tc.sort_index(ascending=True)
ser = pd.Series(1, index=tc.index)
items = []
for i in range(1, len(tc.index)+1):
    for j in range(ser[i], tc[i]+1):
        s = str(j)
        items.append(s)
duration = pd.DataFrame(items, columns=['Duration'])
duration = duration.apply(pd.to_numeric)
copy = snout.copy()

'#merge snout with duration'

copy = pd.merge(copy, duration, left_index=True, right_index=True)
copy.set_index('Duration', inplace=True)
snout = pd.merge(snout, duration, left_index=True, right_index=True)

'#choosing between trials with and without stimulation'
ID = dataframe.pivot(columns='trial', values='ID')
ID = ID.apply(lambda a: a.dropna().reset_index(drop=True))
trials = ID.columns.tolist()
ID1 = ID[ID == 'Laser2']
ID1 = ID1.dropna(axis=1, how='all', inplace=False)
sti = ID1.columns.tolist()
without = np.setdiff1d(trials, sti).tolist()

'#choosing trials with optical-genetic stimulation'
ID2 = snout.pivot(columns='trial', values='ID')
ID2 = ID2.apply(lambda a: a.dropna().reset_index(drop=True))
ID3 = ID[ID == 'Laser2']
ID3 = ID3.dropna(axis=1, how='all', inplace=False)
sti2 = ID3.columns.tolist()

"#duration of each trial(in observations)"

dur = dataframe.trial.value_counts()
dur = dur.sort_index(ascending=True)

"""find trials with a lot of observations"""

tcsti = tc[sti2]
tcsti = tcsti[tcsti < 2500]
tcnosti = tc[without]
tcnosti = tcnosti[tcnosti < 2500]
stdsti = tcsti.std()
stdnosti = tcnosti.std()
meansti = tcsti.mean()
meannosti = tcnosti.mean()
alot = tcsti[tcsti >= (1 * stdsti) + meansti]
alot2 = tcnosti[tcnosti >= (1 * stdnosti) + meannosti]
stif = tcsti[tcsti < (1 * stdsti) + meansti]
nostif = tcnosti[tcnosti < (1 * stdnosti) + meannosti]
alot = alot.index.to_list()
alot2 = alot2.index.to_list()
stif = stif.index.to_list()
nostif = nostif.index.to_list()

'#filter snout dataframe'

for i in range(0, len(without)):
    indextrials = snout[snout['trial'] == without[i]].index
    snout.drop(indextrials, inplace=True)
for i in range(0, len(nostif)):
    indextrials = snout[snout['trial'] == nostif[i]].index
    snout.drop(indextrials, inplace=True)
for i in range(0, len(alot2)):
    indextrials = snout[snout['trial'] == alot2[i]].index
    snout.drop(indextrials, inplace=True)
for i in range(0, len(alot)):
    indextrials = snout[snout['trial'] == alot[i]].index
    snout.drop(indextrials, inplace=True)

'#locate trials with more than one stimulation and remove them from snout'

stima = snout.loc[snout['time'].isin(start)]
stim = stima[['time']]
merged = pd.merge(stim, snout, how='inner', on='time')
merge = merged.pivot(columns='ID', values='trial')
cnt = merge.value_counts('Laser2')
kako = cnt[cnt > 1]
kako = kako.index.tolist()
for i in range(0, len(kako)):
    indextrials = snout[snout['trial'] == kako[i]].index
    snout.drop(indextrials, inplace=True)

'#choose the actually found stimuli'
x = kako+alot
stitf = np.setdiff1d(sti2, x).tolist()

'#locate the duration of the trial when stimulation occurs'
merged.drop_duplicates(subset='trial', keep=False, inplace=True)
tla = snout.trial.value_counts()
tla = tla.sort_index(ascending=True)
start = merged['time'].tolist()
snout.set_index('Duration', inplace=True)
durati = []
for i in range(0, len(start)):
    indexstart = snout[snout['time'] == start[i]].index
    durati.append(indexstart)

"# find number of observations before stimulation start"
before = durati.copy()
before[:] = [number - 1 for number in before]
before[:] = [number*(-1) for number in before]

"#find number of observations after stimulation start"
tc2 = tc[stitf]
tc2 = tc2 - 1
du = before.copy()
tc2 = tc2.tolist()
tla = tla - 1
tla = tla.tolist()
after = []
zip_object = zip(tla, du)
for tla_i, du_i in zip_object:
    after.append(tla_i+du_i)

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

hbaala = snout.pivot(columns='trial', values='head_body_angle')

hba = copy.pivot(columns='trial', values='head_body_angle')
hba = hba.dropna(how='all', axis=1)
hba = hba.dropna(how='all', axis=0)

hba_without_sti = hba[nostif]
hba_without_sti = hba_without_sti.dropna(how='all', axis=1)
hba_without_sti = hba_without_sti.dropna(how='all', axis=0)

'#find the amount of na in each trial'
'#dot'
nad = hbaala.isnull().sum()
nad = len(hbaala.index) - nad
nad = tc - nad
nad = nad[nad > 300]
nad = nad.index.tolist()
'#no dot'
nand = hba_without_sti.isnull().sum()
nand = len(hba_without_sti.index) - nand
nand = tc - nand
nand = nand[nand > 300]
nand = nand.index.tolist()

'#remove trials with a lot of na from dataframes'

hbaala = hbaala.drop(nad, axis=1)
hbaala = hbaala.dropna(how='all', axis=0)
hba_without_sti = hba_without_sti.drop(nand, axis=1)
hba_without_sti = hba_without_sti.dropna(how='all', axis=0)

'#shape index'
hba_without_sti.index = hba_without_sti.index + hbaala.index[0]-1
hbaala.index = hbaala.index*2
hba_without_sti.index = hba_without_sti.index*2

hbaala.reset_index(inplace=True)
hba_without_sti.reset_index(drop=True, inplace=True)
a = int(hbaala.loc[hbaala['Duration'] == 0].index.values)
n_hbaala = hbaala.iloc[a-250:a+751, :]
n_hbaala.set_index('Duration', inplace=True)

b = int(hba_without_sti.tail(1).index.values/3)
if b > 250:
    n_hba_without_sti = hba_without_sti.iloc[b-250:b+751, :]
    n_hba_without_sti.reset_index(drop=True, inplace=True)
    n_hba_without_sti.index = n_hba_without_sti.index - 250
    n_hba_without_sti.index = n_hba_without_sti.index * 2
    n_hba_without_sti.index.rename('Duration', inplace=True)

else:
    n_hba_without_sti = hba_without_sti.copy()
    n_hba_without_sti.index = hba_without_sti.index - 100
    n_hba_without_sti.index.rename('Duration', inplace=True)

def space(dfsti, dfnosti):
    space = dfsti.copy()
    space.reset_index(drop=True, inplace=True)
    space['snouty'] = space['snouty'].round(decimals=0)
    space.set_index('snouty', inplace=True)
    space = space[['head_body_angle', 'trial']]
    table = space.pivot_table(columns='trial', index='snouty', values='head_body_angle', aggfunc='mean')

    space = dfnosti.copy()
    space.reset_index(drop=True, inplace=True)
    space['snouty'] = space['snouty'].round(decimals=0)
    space.set_index('snouty', inplace=True)
    space = space[['head_body_angle', 'trial']]
    table2 = space.pivot_table(columns='trial', index='snouty', values='head_body_angle', aggfunc='mean')
    table2 = table2[nostif]

    table = table.drop(nad, axis=1)
    table = table.dropna(how='all', axis=0)
    table2 = table2.drop(nand, axis=1)
    table2 = table2.dropna(how='all', axis=0)

    table.iloc[:] = table.iloc[::-1].values
    table2.iloc[:] = table2.iloc[::-1].values

    table.to_excel('space-hba-sti.xlsx')
    table2.to_excel('space-hba-no-sti.xlsx')
space(snout,copy)


"""Export to xlsx"""

hbaala.to_excel('hba-task+opto- after snout- 0 = sti start.xlsx')
hba_without_sti.to_excel('hba-no-stimulation-after snout.xlsx')

hbaala_abs = hbaala.abs()
hba_without_sti_abs = hba_without_sti.abs()
hbaala_abs.to_excel('abs-hba-task+opto-after snout- 0 = sti start.xlsx')
hba_without_sti_abs.to_excel('abs-hba-no-stimulation-after snout.xlsx')

n_hbaala.to_excel('hba-sti.xlsx')
n_hba_without_sti.to_excel('hba-no-sti.xlsx')

