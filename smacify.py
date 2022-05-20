import numpy as np
from smac.facade.smac_ac_facade import SMAC4AC
from smac.intensification.hyperband import Hyperband
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, OrdinalHyperparameter, GreaterThanCondition,\
    CategoricalHyperparameter, UniformFloatHyperparameter, EqualsCondition, InCondition
from network_reinforce import REINFORCE
from smac.scenario.scenario import Scenario
import numpy as np
import network_reinforce
from concurrent.futures import ProcessPoolExecutor
import warnings
import os
warnings.filterwarnings("ignore")

executor = ProcessPoolExecutor(max_workers=3)
def run(config):
    print(config)
    rs = np.zeros(2000)
    runners = []
    for i in range(3):
        net = [config['n_neurons1']]
        
        if config['n_layers'] >= 2:
            net.append(config['n_neurons2'])
        if config['n_layers'] >= 3:
            net.append(config['n_neurons3'])
        
        runners.append(executor.submit(network_reinforce.run,2000, net, config['activation'], config['optimizer'],
                     batch_size=config['m'], n_steps=config['n'] if config['type'] in ['ac_bootstrapping', 'ac_full'] else -1,
                      learning_rate=config['optimizer_lr'], actor_critic=config['type'] != 'reinforce',
                       baseline_subtraction=config['type'] in ['ac_bootstrapping', 'ac_full'], entropy_regularization=config['eta'] > 0, eta=config['eta']))
    for runner in runners:
        rs += runner.result()
    rs /= 3
    return 1000000 - np.trapz(rs)


if __name__ == '__main__':
    cs = ConfigurationSpace()
    n_layers = cs.add_hyperparameter(OrdinalHyperparameter('n_layers', [1, 2, 3]))
    n_neurons1 = cs.add_hyperparameter(UniformIntegerHyperparameter('n_neurons1', lower=16, upper=512, q=16))
    n_neurons2 = cs.add_hyperparameter(UniformIntegerHyperparameter('n_neurons2', lower=16, upper=512, q=16))
    n_neurons3 = cs.add_hyperparameter(UniformIntegerHyperparameter('n_neurons3', lower=16, upper=512, q=16))
    # n_neurons4 = cs.add_hyperparameter(UniformIntegerHyperparameter('n_neurons4', lower=16, upper=512, q=2))

    cs.add_condition(GreaterThanCondition(n_neurons2, n_layers, 1))
    cs.add_condition(GreaterThanCondition(n_neurons3, n_layers, 2))

    cs.add_hyperparameter(
        CategoricalHyperparameter('optimizer', ['adam', 'lamb', 'sgd', 'adamw']))
    cs.add_hyperparameter(
        UniformFloatHyperparameter('optimizer_lr', lower=1e-5, upper=1e-1, log=True))

    cs.add_hyperparameter(CategoricalHyperparameter('activation', ['relu', 'tanh', 'elu', 'gelu']))

    type = cs.add_hyperparameter(CategoricalHyperparameter('type', ['reinforce', 'ac_bootstrapping', 'ac_baseline_sub',
                                                             'ac_full']))

    cs.add_hyperparameter(UniformFloatHyperparameter('eta', lower=0, upper=1))

    n = cs.add_hyperparameter(UniformIntegerHyperparameter('n', lower=1, upper=300))
    cs.add_condition(InCondition(n, type, ['ac_bootstrapping', 'ac_full']))
    m = cs.add_hyperparameter(UniformIntegerHyperparameter('m', lower=1, upper=500))
    print(cs)    
    scenario = Scenario({
        'run_obj': 'quality',
        'deterministic': True,
        #'wallclock_limit': 18 * 60 * 60,
        'output_dir': '/data/s3092593/rl3_smac',
        'cs': cs,
    })
    smac = SMAC4AC(tae_runner=run,
                     n_jobs=1,
                     scenario=scenario,
                     rng=51,
    )
    smac.optimize()
