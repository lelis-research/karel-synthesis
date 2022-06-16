# TODO: this contains the main calls from [leaps]/pretrain/trainer.py for reference
# The models and args will be adapted to this repo

model = SupervisedRLModel(device, config, envs, dsl, logger, writer, global_logs, config['verbose'])