import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('out/leave_k_out.out', delimiter=' ')
df.groupby("k")["distance"].mean().plot()
plt.xlabel("k")
plt.ylabel("Average distance")
plt.show()

