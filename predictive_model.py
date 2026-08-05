import pandas as pd
import numpy as np
import torch
import random
import os
import matplotlib.pyplot as plt
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from chemprop import data, featurizers, models, nn, utils
from chemprop.models import multi
from sklearn.model_selection import train_test_split

# I am just thinking shall I look at the kfolds wdyt 
seed = 42 

os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#--------------------------------------------
# Load message passing because I am freezing the backbone
#--------------------------------------------

agg = nn.NormAggregation()
# Best NF10K model
checkpoint_path = '/ibex/user/baderl/projects/PBI_project/AI_PBI_Project/OSN_NO_PBI/pretraining_OSN_model/ray_results/TorchTrainer_2025-12-30_15-36-00/6a164f09/checkpoint_000009/checkpoint.ckpt'  # best config 4.
OSN_mp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
mp_hparams = OSN_mp['hyper_parameters']['message_passing'].copy()
if 'cls' in mp_hparams:
    del mp_hparams['cls']
mp = nn.BondMessagePassing(**mp_hparams)

state_dict = OSN_mp['state_dict']
mp_state_dict = {k.replace('message_passing.', ''): v for k, v in state_dict.items() if k.startswith('message_passing.')}
mp.load_state_dict(mp_state_dict)

print(mp)

#--------------------------------------------
# Load PBI data
#--------------------------------------------
df_input = pd.read_csv('/ibex/user/baderl/projects/PBI_project/AI_PBI_Project/data/OSN_PBI_Dataset_Master_CSV.csv')
df_input = df_input[['SMILES', 'MW', 'solvent name', 'membrane', 'pressure (bar)', 'Permeances (LMH/Bar)', 'Average Rejection (0-1)']]

"""
Solvent names:      Methanol, Acetonitrile, 2-propanol, Ethanol, Ethyl acetate, Acetone 
Membrane names:     HF_c, HF_nc, FS_c, FS_nc 
Pressures:          10, 20 
# just remember the hollow fiber membranes are at 10 bar and the flat sheet membranes are at 20 bar. 
"""

num_workers = 0 
smiles_columns = 'SMILES' 
target_columns = ['Average Rejection (0-1)'] 

smis = df_input.loc[:, smiles_columns].values
ys = df_input.loc[:, target_columns].values

pressures = df_input['pressure (bar)'].values.reshape(-1, 1) 
molecular_weights = df_input['MW'].values.reshape(-1, 1)
permeances = df_input['Permeances (LMH/Bar)'].values.reshape(-1, 1)

membrane_ohe = pd.get_dummies(df_input['membrane'])
membrane_ohe_array = membrane_ohe.values
solvent_ohe = pd.get_dummies(df_input['solvent name'])
solvent_ohe_array = solvent_ohe.values
x_ds = np.concatenate([pressures, molecular_weights, permeances, membrane_ohe_array, solvent_ohe_array], axis=1)
mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]
datapoints = [data.MoleculeDatapoint(mol, y, x_d=X_d) for mol, y, X_d in zip(mols, ys, x_ds)]

#--------------------------------------------
# Data splitting 
#--------------------------------------------

mols = [d.mol for d in datapoints]  # RDkit Mol objects are use for structure based splits
train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1))
train_data, val_data, test_data = data.split_data_by_indices(
    datapoints, train_indices, val_indices, test_indices
)

#--------------------------------------------
# Data loaders and scaling  
#--------------------------------------------

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
train_dset = data.MoleculeDataset(train_data[0], featurizer)
scaler = train_dset.normalize_targets()
x_ds_scaler = train_dset.normalize_inputs("X_d")

val_dset = data.MoleculeDataset(val_data[0], featurizer)
val_dset.normalize_targets(scaler)
val_dset.normalize_inputs("X_d", x_ds_scaler)

test_dset = data.MoleculeDataset(test_data[0], featurizer)
test_dset.normalize_inputs("X_d", x_ds_scaler)

train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)

#--------------------------------------------
# Final model  
#--------------------------------------------

ffn_input_dim = mp.output_dim + x_ds.shape[1]
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler) # here i have scaled the outputs
ffn = nn.RegressionFFN(input_dim=ffn_input_dim, output_transform=output_transform, dropout = 0.05)
metric_list = [nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.MSE(), nn.metrics.R2Score()]
mpnn = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=metric_list)

#--------------------------------------------
# Freeze the message passing layers  
#--------------------------------------------
mpnn.message_passing.apply(lambda module: module.requires_grad_(False))
mpnn.message_passing.eval()

#--------------------------------------------
# Training the model  
#--------------------------------------------

trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=20, # number of epochs to train for
)

trainer.fit(mpnn, train_loader, val_loader) # start training

results = trainer.test(
    mpnn,
    test_loader,
) # mainly outputs performance metrics on the test set with known labels

prediction_batches = trainer.predict(
    mpnn,
    test_loader,
)

test_predictions_raw = np.concatenate(
    prediction_batches,
    axis=0,
)

test_predictions = np.asarray(test_predictions_raw).reshape(-1)
test_predictions[test_predictions > 1.0] = 1.0 # clipped at 1 


test_targets = np.asarray([
    datapoint.y[0]
    for datapoint in test_data[0]
]).reshape(-1)

# --------------------------------------------
# Rejection parity plot
# --------------------------------------------

plt.figure(figsize=(6, 6))

plt.scatter(
    test_targets,
    test_predictions)

min_rejection = np.min(
    np.array([test_targets, test_predictions])
) - 0.05

max_rejection = np.max(
    np.array([test_targets, test_predictions])
) + 0.05

plt.axline(
    (min_rejection, min_rejection),
    slope=1,
    linestyle="--",
    color="red")

plt.xlabel("Measured rejection")
plt.ylabel("Predicted rejection")
plt.xlim([min_rejection, max_rejection])
plt.ylim([min_rejection, max_rejection])
plt.tight_layout()

plt.savefig(
    "/ibex/user/baderl/projects/PBI_project/part_2/rejection_parity_plot.png",
    dpi=300,
    bbox_inches="tight",
)

# use hyperarameter optimization 
# after this works, you can actually go to prediction.ipynb
