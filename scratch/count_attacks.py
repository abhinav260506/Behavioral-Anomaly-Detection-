import pandas as pd

df = pd.read_csv("data/synthetic_telemetry_27k.csv")
df.columns = df.columns.str.strip()

total_events = len(df)
train_end = int(total_events * 0.70)
val_end = int(total_events * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

print(f"Total Dataset Size: {total_events} events\n")

for attack in ["low_and_slow_exfiltration", "insider_drift"]:
    total_cnt = (df["label"] == attack).sum()
    train_cnt = (train_df["label"] == attack).sum()
    val_cnt = (val_df["label"] == attack).sum()
    test_cnt = (test_df["label"] == attack).sum()
    
    print(f"--- {attack.upper()} ---")
    print(f"Total in Dataset:  {total_cnt} events ({total_cnt/total_events*100:.2f}%)")
    print(f"  • Training (70%):   {train_cnt} events")
    print(f"  • Validation (15%): {val_cnt} events")
    print(f"  • Testing (15%):    {test_cnt} events\n")
