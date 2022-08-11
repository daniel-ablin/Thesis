from tqdm import tqdm
import time

a = tqdm(range(10), leave=True, position=0, desc='a')
for i in a:
    for j in tqdm(range(10), leave=False, position=1, desc='b'):
        time.sleep(0.05)
