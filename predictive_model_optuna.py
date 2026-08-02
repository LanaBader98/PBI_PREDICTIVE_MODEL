# transfer learning from the checkpoint of the nf10k 
# predict rejections 
# if the model has good accuracy then I am going predict the rejections of the solutes in the application dataset in
# different types of membranes. 

import pandas as pd
from lightning import pytorch as pl
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint

from chemprop import data, featurizers, models, nn, utils
from chemprop.nn import metrics
from chemprop.models import multi
import numpy as np
from torch.utils.data import DataLoader
import seaborn as sns

import matplotlib.pyplot as plt
import csv 

from sklearn.preprocessing import StandardScaler
import torch
import random
from sklearn.model_selection import train_test_split
import os

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import matplotlib.pyplot as plt

from pathlib import Path
import ray
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (RayDDPStrategy, RayLightningEnvironment,
                                 RayTrainReportCallback, prepare_trainer)
from ray.train.torch import TorchTrainer
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import FIFOScheduler

from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

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
# Trying Raytune here for hyperparameter optimization  
#--------------------------------------------

RAY_RESULTS_DIR = Path(
    "/ibex/user/baderl/projects/PBI_project/part_2/ray_results"
)

RAY_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NUM_TRIALS = 1
SEED = 42

pl.seed_everything(
    SEED,
    workers=True,
)

x_d_dim = x_ds.shape[1]

# ============================================================
# 2. BUILD A FRESH TRANSFER-LEARNING MODEL
# ============================================================

def build_transfer_model(
    config,
    mp_hparams,
    mp_state_dict,
    scaler,
    x_d_dim,
):

    mp = nn.BondMessagePassing(
        **mp_hparams
    )

    # Load the pretrained NF-10K message-passing weights
    mp.load_state_dict(
        mp_state_dict
    )

    # Freeze all message-passing parameters
    for parameter in mp.parameters():
        parameter.requires_grad = False

    mp.eval()

    # --------------------------------------------------------
    # Aggregation layer
    # --------------------------------------------------------

    agg = nn.NormAggregation()

    # Size of molecular representation produced by the MPNN
    message_hidden_dim = int(
        mp_hparams["d_h"]
    )

    # The FFN receives:
    # molecular representation + PBI X_d features
    ffn_input_dim = (
        message_hidden_dim + x_d_dim
    )

    # --------------------------------------------------------
    # Convert model outputs back to the original PBI target
    # units using the PBI training scaler
    # --------------------------------------------------------

    output_transform = (
        nn.UnscaleTransform.from_standard_scaler(
            scaler
        )
    )

    # --------------------------------------------------------
    # Build a new PBI-specific FFN using Ray's hyperparameters
    # --------------------------------------------------------

    ffn = nn.RegressionFFN(
        input_dim=ffn_input_dim,
        hidden_dim=int(
            config["ffn_hidden_dim"]
        ),
        n_layers=int(
            config["ffn_num_layers"]
        ),
        dropout=float(
            config["dropout"]
        ),
        output_transform=output_transform,
    )

    batch_norm = True

    metric_list = [
        nn.metrics.RMSE(),
        nn.metrics.MAE(),
    ]

    model = models.MPNN(
        mp,
        agg,
        ffn,
        batch_norm,
        metric_list,
    )

    # Apply trial-specific learning rates
    model.init_lr = float(
        config["init_lr"]
    )

    model.max_lr = float(
        config["max_lr"]
    )

    model.final_lr = float(
        config["final_lr"]
    )

    model.warmup_epochs = int(
        config["warmup_epochs"]
    )

    return model


# ============================================================
# 3. TRAIN ONE RAY WORKER
# ============================================================

def train_model(
    config,
    train_dset,
    val_dset,
    num_workers,
    scaler,
    mp_hparams,
    mp_state_dict,
    x_d_dim,
):

    max_epochs = int(
        config["max_epochs"]
    )

    patience = int(
        config["patience"]
    )

    model = build_transfer_model(
        config=config,
        mp_hparams=mp_hparams,
        mp_state_dict=mp_state_dict,
        scaler=scaler,
        x_d_dim=x_d_dim,
    )

    train_loader = data.build_dataloader(train_dset, num_workers=num_workers, shuffle=True)
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        min_delta=0.0,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=2, # later change to max_epochs
        strategy=RayDDPStrategy(),
        plugins=[
            RayLightningEnvironment()],
        callbacks=[
            early_stopping,
            RayTrainReportCallback()]
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, train_loader, val_loader)



search_space = {
    "ffn_hidden_dim": tune.qrandint(
        lower=300,
        upper=2401,
        q=100,
    ),
    "ffn_num_layers": tune.qrandint(
        lower=1,
        upper=4,
        q=1,
    ),
    "dropout": tune.uniform(
        0.0,
        0.4,
    ),
    "batch_size": tune.choice(
        [16, 32, 64]
    ),
    "init_lr": tune.loguniform(
        1e-6,
        1e-4,
    ),
    "max_lr": tune.loguniform(
        1e-5,
        1e-3,
    ),
    "final_lr": tune.loguniform(
        1e-7,
        1e-5,
    ),
    "warmup_epochs": tune.randint(
        1,
        6,
    ),
    "max_epochs": tune.choice(
        [20, 40, 60, 80]
    ),
    "patience": tune.choice(
        [5, 10, 15]
    ),
}


# ============================================================
# 4. RAY TRAIN RESOURCE SETTINGS
# ============================================================
ray.init()

scheduler = FIFOScheduler()

scaling_config = ScalingConfig(
    num_workers=1,
    use_gpu=True)

checkpoint_config = CheckpointConfig(
    num_to_keep=1,
    checkpoint_score_attribute="val_loss",
    checkpoint_score_order="min")

run_config = RunConfig(
    checkpoint_config=checkpoint_config,
    storage_path=str(
        RAY_RESULTS_DIR))


def train_func(config):
    ray_trainer = TorchTrainer(
        lambda worker_config: train_model(
            config=worker_config,
            train_dset=train_dset,
            val_dset=val_dset,
            num_workers=num_workers,
            scaler=scaler,
            mp_hparams=mp_hparams,
            mp_state_dict=mp_state_dict,
            x_d_dim=x_d_dim,
        ),

        scaling_config=scaling_config,
        run_config=run_config,
        train_loop_config=config,
    )

    result = ray_trainer.fit()
    tune.report(
        metrics=result.metrics,
        checkpoint=result.checkpoint)

search_algorithm = HyperOptSearch(
    n_initial_points=5,
    random_state_seed=SEED)

tune_config = tune.TuneConfig(
    # Define metric and mode only here
    metric="val_loss",
    mode="min",
    num_samples=NUM_TRIALS,
    scheduler=scheduler,
    search_alg=search_algorithm,
    trial_dirname_creator=lambda trial: str(
        trial.trial_id
    ),
)

tuner = tune.Tuner(
    train_func,
    param_space=search_space,
    tune_config=tune_config,
)

results = tuner.fit()

result_df = results.get_dataframe()
result_df

best_result = results.get_best_result()
best_config = best_result.config
best_config

ray.shutdown()


# ============================================================
# 12. GET THE BEST RESULT
# ============================================================

best_result = results.get_best_result(
    metric="val_loss",
    mode="min",
)

best_config = best_result.config
best_checkpoint = best_result.checkpoint

print("\nBest validation loss:")
print(
    best_result.metrics["val_loss"]
)

print("\nBest hyperparameters:")

for parameter_name, parameter_value in best_config.items():
    print(
        f"{parameter_name}: {parameter_value}"
    )

print("\nBest Ray checkpoint:")
print(best_checkpoint)


# ============================================================
# 13. SAVE ALL RAY RESULTS
# ============================================================

ray_results_df = results.get_dataframe()

ray_results_df.to_csv(
    RAY_RESULTS_DIR / "all_ray_trials.csv",
    index=False,
)

with open(
    RAY_RESULTS_DIR / "best_hyperparameters.txt",
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "Best validation loss: "
        f"{best_result.metrics['val_loss']}\n\n"
    )

    for name, value in best_config.items():
        file.write(
            f"{name}: {value}\n"
        )


final_model = build_transfer_model(
    config=best_config,
    mp_hparams=mp_hparams,
    mp_state_dict=mp_state_dict,
    scaler=scaler,
    x_d_dim=x_d_dim,
)

final_train_loader = data.build_dataloader(
    train_dset,
    batch_size=int(
        best_config["batch_size"]
    ),
    num_workers=num_workers,
    shuffle=True,
)

final_val_loader = data.build_dataloader(
    val_dset,
    batch_size=int(
        best_config["batch_size"]
    ),
    num_workers=num_workers,
    shuffle=False,
)

final_test_loader = data.build_dataloader(
    test_dset,
    batch_size=int(
        best_config["batch_size"]
    ),
    num_workers=num_workers,
    shuffle=False,
)


# ============================================================
# 15. SAVE AND TRAIN THE FINAL MODEL
# ============================================================

FINAL_MODEL_DIR = (
    RAY_RESULTS_DIR / "final_model"
)

FINAL_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

final_checkpoint_callback = ModelCheckpoint(
    dirpath=FINAL_MODEL_DIR,
    filename="best_pbi_transfer_model",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)

final_early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=int(
        best_config["patience"]
    ),
)

final_trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=int(
        best_config["max_epochs"]
    ),

    callbacks=[
        final_checkpoint_callback,
        final_early_stopping,
    ],

    enable_progress_bar=True,
)

final_trainer.fit(
    final_model,
    final_train_loader,
    final_val_loader,
)

print("\nFinal model saved at:")
print(
    final_checkpoint_callback.best_model_path
)


# ============================================================
# 16. TEST THE FINAL MODEL
# ============================================================

best_final_model = models.MPNN.load_from_checkpoint(
    final_checkpoint_callback.best_model_path,
    map_location="cpu",
)

test_results = final_trainer.test(
    best_final_model,
    dataloaders=final_test_loader,
)

print("\nFinal test results:")
print(test_results)

ray.shutdown()