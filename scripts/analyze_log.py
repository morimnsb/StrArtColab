import csv, os
import matplotlib.pyplot as plt

LOG = "outputs_importance_tracked/line_log.csv"
ts, finals, bases, pens = [], [], [], []
with open(LOG, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        ts.append(int(row["t"]))
        finals.append(float(row["final_score"]))
        bases.append(float(row["base_score"]))
        pens.append(int(row["penalty"]))

plt.figure()
plt.plot(ts, bases, label="base score")
plt.plot(ts, pens, label="penalty")
plt.plot(ts, finals, label="final score")
plt.xlabel("step (t)"); plt.legend(); plt.title("Line selection dynamics")
plt.tight_layout()
out = os.path.join(os.path.dirname(LOG), "scores_plot.png")
plt.savefig(out, dpi=150)
print(f"✅ saved {out}")
