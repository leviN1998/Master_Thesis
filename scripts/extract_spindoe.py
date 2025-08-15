import re
import pandas as pd

file_path = "data/spin_dataset/spin_dataset.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Regex für MAE + Spin in rad/s + spin norm in rps
pattern = re.compile(
    r"(spin_[^\n]+).*?(rec_\d).*?(\d+):\s+MAE:\s*([\d\.\-eE]+).*?Spin in rad/s\s*\[([^\]]+)\].*?spin norm in rps\s*([\d\.\-eE]+)",
    re.DOTALL
)

data = []
for match in pattern.finditer(text):
    spin_label = match.group(1).strip()
    rec_label = match.group(2).strip()
    index_num = int(match.group(3))
    mae = float(match.group(4))
    spin_rad_s = [float(x) for x in match.group(5).split()]
    spin_rps = float(match.group(6))

    data.append({
        "Spin_Label": spin_label,
        "Rec_Label": rec_label,
        "Index": index_num,
        "MAE": mae,
        "Spin_rad_s": spin_rad_s,
        "Spin_rps": spin_rps
    })

df = pd.DataFrame(data)
print(df)

# Optional als CSV speichern
df.to_csv("spin_extracted_with_labels.csv", index=False)