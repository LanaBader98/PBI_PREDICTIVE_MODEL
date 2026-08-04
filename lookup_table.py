import pandas as pd 
import json
from pathlib import Path

df_training = pd.read_csv("/ibex/user/baderl/projects/PBI_project/AI_PBI_Project/data/OSN_PBI_Dataset_Master_CSV.csv")
df_training = df_training[['SMILES', 'MW', 'solvent name', 'membrane', 'pressure (bar)', 'Permeances (LMH/Bar)', 'Average Rejection (0-1)']]
membranes = ['FS_c', 'HF_c', 'HF_nc']
df_training = df_training[df_training['membrane'].isin(membranes)]

pair_summary = (
    df_training
    .groupby(["membrane", "solvent name"])
    ["Permeances (LMH/Bar)"]
    .agg(["count", "nunique", "mean", "median"]))

permeance_table = (
    df_training
    .groupby(
        ["membrane", "solvent name"],
        as_index=False,
    )["Permeances (LMH/Bar)"]
    .median()
)
permeance_lookup = {
    membrane: (
        group.set_index("solvent name")
        ["Permeances (LMH/Bar)"]
        .astype(float)
        .to_dict()
    )
    for membrane, group in permeance_table.groupby("membrane")
}

output_path = Path(
    "/ibex/user/baderl/projects/PBI_project/part_2/"
    "permeances.json"
)

with output_path.open("w") as file:
    json.dump(
        permeance_lookup,
        file,
        indent=4,
        sort_keys=True,
    )
