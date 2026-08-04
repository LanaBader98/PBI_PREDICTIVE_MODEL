import pandas as pd
import numpy as np
import torch
from lightning import pytorch as pl
from pathlib import Path
import joblib 
import json
from chemprop import data, featurizers, models, utils 
from rdkit.Chem import Descriptors
#--------------------------------------------
# Load the mpnn model from best checkpoint after hyperparameter optimization
# Load the preprocessing steps 
#--------------------------------------------
mpnn = models.MPNN.load_from_checkpoint("/ibex/user/baderl/projects/PBI_project/part_2/ray_results/final_model/best_pbi_transfer_model-v1.ckpt")
# print(mpnn)

preprocessing = joblib.load("/ibex/user/baderl/projects/PBI_project/part_2/ray_results/pbi_preprocessing.joblib")
x_ds_scaler = preprocessing["x_ds_scaler"]
membrane_columns = preprocessing["membrane_columns"]
solvent_columns = preprocessing["solvent_columns"]

#--------------------------------------------
# Application dataset
#--------------------------------------------

df_application_ds = pd.read_csv(
    "/ibex/user/baderl/projects/PBI_project/part_2/data/subset_application_DS.csv"
).head(5).copy()
df_application_ds.columns = df_application_ds.columns.str.strip()
df_application_ds["pressure (bar)"] = 10.0

# --------------------------------------------------
# Load membrane/solvent/permeance combinations
# --------------------------------------------------

permeance_json = Path("/ibex/user/baderl/projects/PBI_project/part_2/permeances.json")

with permeance_json.open("r") as file:
    permeance_lookup = json.load(file)

lookup_rows = []

for membrane, solvents in permeance_lookup.items():
    for solvent, permeance in solvents.items():
        lookup_rows.append(
            {
                "membrane": membrane,
                "solvent name": solvent,
                "Permeances (LMH/Bar)": float(permeance),
            })

lookup_table = pd.DataFrame(lookup_rows)
df_application_ds = df_application_ds.merge(lookup_table, how="cross")


# --------------------------------------------------
# The rest is similar to my previous code
# --------------------------------------------------
membrane_ohe = pd.get_dummies(df_application_ds["membrane"]
).reindex(
    columns=membrane_columns)

solvent_ohe = pd.get_dummies(
    df_application_ds["solvent name"]
).reindex(
    columns=solvent_columns,
    # fill_value=0,
    )

smiles_column = 'SMILES'
smis = df_application_ds[smiles_column]

mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]

df_application_ds["MW"] = [
    Descriptors.MolWt(mol) if mol is not None else np.nan
    for mol in mols
]

x_ds = np.concatenate(
    [df_application_ds["pressure (bar)"]
        .to_numpy(dtype=float)
        .reshape(-1, 1),
        df_application_ds["MW"]
        .to_numpy(dtype=float)
        .reshape(-1, 1),
        df_application_ds["Permeances (LMH/Bar)"]
        .to_numpy(dtype=float)
        .reshape(-1, 1),
        membrane_ohe.to_numpy(dtype=float),
        solvent_ohe.to_numpy(dtype=float),
    ],
    axis=1)



datapoints = [data.MoleculeDatapoint(mol, x_d=X_d) for mol, X_d in zip(mols, x_ds)]

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
application_dset = data.MoleculeDataset(datapoints, featurizer)
application_dset.normalize_inputs("X_d", x_ds_scaler) # the same scaler used during training 

application_loader = data.build_dataloader(application_dset, num_workers=0, shuffle=False )

# --------------------------------------------------
# Begin the training 
# --------------------------------------------------

with torch.inference_mode():
    trainer = pl.Trainer(
        logger=None,
        accelerator="cpu",
        devices=1)
    test_preds = trainer.predict(mpnn, application_loader)

test_preds = np.concatenate(test_preds, axis=0)
df_application_ds['pred'] = test_preds
print(df_application_ds.head(10))
df_application_ds.to_csv("/ibex/user/baderl/projects/PBI_project/part_2/data/application_predictions.csv", index=False)