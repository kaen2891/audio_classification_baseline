# -*- coding: cp949 -*-
import pandas as pd
import os
from sklearn.model_selection import train_test_split


annotation = '/data2/wkuData/자살문단_20250422.csv'

df = pd.read_csv(annotation)
# PHQ9_B 없는 값 제거
df = df.dropna(subset=['PHQ9_B'])

# annotation 과 음성 파일 mapping 작업
files = df['연구번호'].values.tolist()

#data_dir = 'E:\wonkwang\Data\가을문단all'
data_dir = '/data2/wkuData/가을문단all'
prefix = '가을문단all/'

data_folder = '/data2/wkuData/'
issues = []

new_files = []
for i, data in enumerate(files):
    
    new_file = os.path.join(data_dir, data+'_가을문단all.wav')
        
    save_file = os.path.join(prefix, data+'_가을문단all.wav')


    if not os.path.isfile(new_file):
        issues.append(data)
    else:
        new_files.append(save_file)
print('# of no mapping data: ', len(issues))


remove_indices = df[df['연구번호'].isin(issues)].index.tolist()
print(remove_indices)

df_cleaned = df.drop(index=remove_indices).reset_index(drop=True)
print(len(df_cleaned))

# 최종 mapping 된 데이터 요소들
files = df['연구번호'].values.tolist()
df_cleaned['연구번호'] = new_files
#print(len(new_files))
#print(len(files))

genders = df_cleaned['Sex'].values.tolist()
ages = df_cleaned['Age'].values.tolist()
mss = df_cleaned['MS'].values.tolist()
phq_bs = df_cleaned['PHQ9_B'].values.tolist()
phq_totals = df_cleaned['PHQ총점_B'].values.tolist()
phq_severities = df_cleaned['PHQ중증도번호_B'].values.tolist()

columns = ['연구번호', 'Sex', 'Age', 'MS', 'PHQ9_B', 'PHQ총점_B', 'PHQ중증도번호_B']
df = df_cleaned[columns]
print(len(df))

print('파일', len(df_cleaned['연구번호'].values.tolist()))

# Train/Test Split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['PHQ9_B']
)

# 저장
train_df.to_csv('./train.csv', index=False)
test_df.to_csv('./test.csv', index=False)

print('# of training {} test {}'.format(len(train_df), len(test_df)))

train_zero = train_df[train_df['PHQ9_B'] == 0]
train_one = train_df[train_df['PHQ9_B'] == 1]
train_two = train_df[train_df['PHQ9_B'] == 2]
train_three = train_df[train_df['PHQ9_B'] == 3]

test_zero = test_df[test_df['PHQ9_B'] == 0]
test_one = test_df[test_df['PHQ9_B'] == 1]
test_two = test_df[test_df['PHQ9_B'] == 2]
test_three = test_df[test_df['PHQ9_B'] == 3]

print('# of label 0 in training {} {} test {} {}'.format(len(train_zero), len(train_zero) / len(train_df), len(train_df), len(test_zero) / len(test_df)))
print('# of label 1 in training {} {} test {} {}'.format(len(train_zero), len(train_one) / len(train_df), len(train_df), len(test_one) / len(test_df)))
print('# of label 2 in training {} {} test {} {}'.format(len(train_zero), len(train_two) / len(train_df), len(train_df), len(test_two) / len(test_df)))
print('# of label 3 in training {} {} test {} {}'.format(len(train_zero), len(train_three) / len(train_df), len(train_df), len(test_three) / len(test_df)))
