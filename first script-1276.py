"""#importing packages"""

import numpy as np
import pandas as pd

"""#loading the data"""

pose = pd.read_csv('pose_data.csv')
stimuli = pd.read_csv('data.csv')

"""different parameters"""

before = 251
during = 500
after = 250

"""#fixing the dataframes"""

stimuli.rename(columns={'Time (ms)': 'time'}, inplace=True)
pose.rename(columns={'frame': 'time'}, inplace=True)
stimuli.loc[stimuli['time'] % 2 != 0, 'time'] = stimuli.loc[:]['time']-1
df = pd.DataFrame({'time': range(0, int(pose.tail(1)['time']), 2)})
pose = pd.merge(pose, df, on='time', how="outer")
pose = pose.sort_values('time', ascending=True)

'#filter dataframes for outliers'
pose.reset_index(drop=True, inplace=True)
copy = pose.copy()
pose['dhba'] = pose["head_body_angle"].diff(-1)
condition = (pose['dhba'] > 0.2) | (pose['dhba'] < (-0.2))
condition.sum()
for i in range(0, len(pose.index)):
    if condition[i]:
        pose.iloc[i-15:i+15, 1:11] = None

'#merging the data (final2 is created to be without stimulation)'

final = pd.merge(pose, stimuli, on="time", how="outer")
final2 = pd.merge(pose, stimuli, on="time", how="outer")

'#find the start of the stimulation'

start = []

for i in stimuli.index-1:
    if (stimuli.iloc[i+1]['time']-600) > (stimuli.iloc[i]['time']):
        start.append(stimuli.iloc[i+1]['time'])
start = start[:-1]

start2 = [number - 3000 for number in start]

'#choosing specific values before and after stimulation start'

final.insert(14, 'sti', 0)
condition = final['time'].isin(start)
n_start = condition.sum()
for i in final.index:
    if condition[i]:
        final.iloc[i - before: i + during + after, 14] = 1

final2.insert(14, 'sti', 0)
condition = final2['time'].isin(start2)
n_start2 = condition.sum()
for i in final2.index:
    if condition[i]:
        final2.iloc[i - before: i + during + after, 14] = 1


'#creating new data frame containing only values before and after stimulation'

pose_1 = final.drop(['ID', 'State', 'Value'], axis=1)
pose_1 = pose_1.loc[(pose_1['sti'] == 1)]
pose_1 = pose_1.reset_index(drop=True)
n = before + after + during
x = 1
for i in range(1, n_start+1):
    if x == 1:
        pose_1.loc[1:(n-1), 'sti'] = i
        x = x+1
    else:
        pose_1.loc[n:n+n, 'sti'] = i
        n = n + (before + after + during)


'#creating new data frame containing only values before and after time set as stimulation for control'

pose_2 = final2.drop(['ID', 'State', 'Value'], axis=1)
pose_2 = pose_2.loc[(pose_2['sti'] == 1)]
pose_2 = pose_2.reset_index(drop=True)
n = before + after + during
x = 1
for i in range(1, n_start2+1):
    if x == 1:
        pose_2.loc[1:(n-1), 'sti'] = i
        x = x+1
    else:
        pose_2.loc[n:n+n, 'sti'] = i
        n = n + (before + after + during)

'#merge pose_1 with duration and set index as duration so the pivot occurs properly'

sc = pose_1.sti.value_counts()
sc = sc.sort_index(ascending=True)

ser = pd.Series(1, index=sc.index)
items = []
for i in range(1, len(sc.index)+1):
    for j in range(ser[i], sc[i]+1):
        s = str(j)
        items.append(s)
duration = pd.DataFrame(items, columns=['Duration'])
duration = duration.apply(pd.to_numeric)
pose_1 = pd.merge(pose_1, duration, left_index=True, right_index=True)
pose_1.set_index('Duration', inplace=True)

'#merge pose_2 with duration and set index as duration so the pivot occurs properly'

sc2 = pose_2.sti.value_counts()
sc2 = sc2.sort_index(ascending=True)

ser2 = pd.Series(1, index=sc2.index)
items = []
for i in range(1, len(sc2.index)+1):
    for j in range(ser2[i], sc2[i]+1):
        s = str(j)
        items.append(s)
duration2 = pd.DataFrame(items, columns=['Duration'])
duration2 = duration2.apply(pd.to_numeric)
pose_2 = pd.merge(pose_2, duration2, left_index=True, right_index=True)
pose_2.set_index('Duration', inplace=True)

'#create data frames with columns indicating number of trial'

hba = pose_1.pivot(columns='sti', values='head_body_angle')
hba_no_stim = pose_2.pivot(columns='sti', values='head_body_angle')

"""shape index to much time(ms)"""

hba.index = hba.index-2
hba_no_stim.index = hba_no_stim.index-2
hba.index = hba.index*2
hba_no_stim.index = hba_no_stim.index*2

"""Export to xlsx"""

hba.to_excel('hba.xlsx')
hba_no_stim.to_excel('hba-no-stim.xlsx')
