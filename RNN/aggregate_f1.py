import glob
import pandas as pd

# acc = []
# for fn in glob.glob('data/**/**/f1_score.csv'):
#     df = pd.read_csv(fn)
#     acc.append(df)
# df = pd.concat(acc,0)
# print(df['f1_score'].mean())
# print('Mean')
# print(df.groupby("cluster_id").apply(lambda x: x["f1_score"].mean()))
# print('Std')
# print(df.groupby("cluster_id").apply(lambda x: x["f1_score"].std()))
# df.to_csv('f1_score_aggregated.csv', index=False)

acc = []
for fn in glob.glob('data/f1_score*.csv'):
    df = pd.read_csv(fn)
    lr = float(fn.split('_')[-1][:-4])
    df['lr'] = lr
    acc.append(df)
df = pd.concat(acc,0)
print(df.groupby("lr").mean().values)

print(df['f1_score'].mean())
print('Mean')
print(df.groupby("cluster_id").apply(lambda x: x["f1_score"].mean()))
print('Std')
print(df.groupby("cluster_id").apply(lambda x: x["f1_score"].std()))
df.to_csv('f1_score_aggregated.csv', index=False)