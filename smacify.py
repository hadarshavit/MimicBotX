from glob import glob
import numpy as np
from smac.intensification.hyperband import Hyperband
from smac.facade.smac_ac_facade import SMAC4AC
from smac.intensification.hyperband import Hyperband
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, OrdinalHyperparameter, GreaterThanCondition,\
    CategoricalHyperparameter, UniformFloatHyperparameter, EqualsCondition, InCondition
from smac.scenario.scenario import Scenario
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import warnings
import os
import timm
from torch import ne, nn
import network
import learn_bc
warnings.filterwarnings("ignore")
id = 0
# executor = ProcessPoolExecutor(max_workers=3)
def run(config):
    print(config)
    global id
    id += 1
    if config['activation'] == 'gelu':
        activation = timm.models.layers.activations.GELU
    elif config['activation'] == 'relu':
        activation = nn.ReLU
    elif config['activation'] == 'elu':
        activation = nn.ELU
    elif config['activation'] == 'celu':
        activation = nn.CELU

    if config['block'] == 'nexception':
        block = network.NEXcepTionBlock
    elif config['block'] == 'resnet':
        block = network.ResNetBlock
    elif config['block'] == 'convnext':
        block = network.ConvNeXtBlock
    
    return learn_bc.main(activation=activation, block=block, optimizer=config['optimizer'], lr=config['optimizer_lr'],
    scheduler=config['scheduler'], epochs=30, save_id=id)


if __name__ == '__main__':
    cs = ConfigurationSpace()

    cs.add_hyperparameter(
        CategoricalHyperparameter('optimizer', ['fusedlamb', 'adamw', 'nadam', 'adamp']))
    cs.add_hyperparameter(
        UniformFloatHyperparameter('optimizer_lr', lower=1e-6, upper=1e-1, log=True))

    cs.add_hyperparameter(CategoricalHyperparameter('activation', ['relu', 'gelu', 'elu', 'celu']))
    cs.add_hyperparameter(CategoricalHyperparameter('scheduler', [True, False]))
    cs.add_hyperparameter(CategoricalHyperparameter('block', ['nexception', 'convnext']))

    print(cs)    
    scenario = Scenario({
        'run_obj': 'quality',
        'deterministic': True,
        #'wallclock_limit': 18 * 60 * 60,
        'output_dir': '/data/s3092593/mgai_smac',
        'cs': cs,
        "runcount_limit":100
    })
    smac = SMAC4AC(tae_runner=run,
                     n_jobs=1,
                     scenario=scenario,
                     rng=51,# intensifier=Hyperband, intensifier_kwargs={''}
    )
    smac.optimize()
