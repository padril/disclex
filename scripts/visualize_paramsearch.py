import matplotlib.pyplot as plt
from collections import defaultdict

# read file
with open("out/paramsearch.out") as f:
    lines = f.read().strip().split("\n")

# parse
data = []
for line in lines:
    parts = [x.strip() for x in line.split(",")]
    d = {}
    for p in parts:
        k, v = p.split("=")
        d[k.strip()] = float(v)
    data.append(d)

# --- slice 1: ug_mix = 1.0 ---
slice1 = [d for d in data if abs(d["ug_mix"] - 1.0) < 1e-9]
slice1 = sorted(slice1, key=lambda x: x["alpha"])

x1 = [d["alpha"] for d in slice1]
y1 = [d["mean_f"] for d in slice1]
e1 = [d["std_f"] for d in slice1]

plt.figure()
plt.errorbar(x1, y1, yerr=e1, marker='o', capsize=3)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\mu_f$")
plt.title(r"$\beta = 1.0$")
plt.tight_layout()
plt.savefig("out/alpha_vs_meanf.svg")

