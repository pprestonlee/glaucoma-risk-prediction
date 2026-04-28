import pandas as pd

def load_grape(path):
    baseline = pd.read_excel(path, sheet_name="Baseline")
    followup = pd.read_excel(path, sheet_name="Follow-up")
    return baseline, followup