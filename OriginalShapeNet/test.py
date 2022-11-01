import pandas as pd
from pathlib import Path

dataset = 'ItalyPowerDemand'
ratio = 0.1
randome_state = 0
save_path = './shapenet_result'
if not Path(save_path).exists():
    Path(save_path).mkdir(parents=True)
csv_file = Path(str(save_path), str(dataset) + '.csv')
if csv_file.exists():
    df = pd.read_csv(csv_file)
else:
    df = pd.DataFrame(columns=['ratio', 'randome_state', 'accuracy'])
    df.to_csv(csv_file, index=False)

if df[(df['ratio'] == ratio) & (df['randome_state'] == randome_state)].empty:
    print('empty')