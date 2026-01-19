import multiprocessing as mp

import sys
import os

# Add the site-packages path
sys.path.append(os.path.join(os.path.dirname(__file__), '.venv/lib/python3.12/site-packages'))

# Now import
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults

# Registers all the ViZDoom environments
def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)

# Sample Factory allows the registration of a custom Neural Network architecture
# See https://github.com/alex-petrenko/sample-factory/blob/master/sf_examples/vizdoom/doom/doom_model.py for more details
def register_vizdoom_models():
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)


def register_vizdoom_components():
    register_vizdoom_envs()
    register_vizdoom_models()

# parse the command line args and create a config
def parse_vizdoom_cfg(argv=None, evaluation=False):
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    # parameters specific to Doom envs
    add_doom_env_args(parser)
    # override Doom default values for algo parameters
    doom_override_defaults(parser)
    # second parsing pass yields the final configuration
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg

# === Main execution guarded ===
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # Import inside guard to avoid top-level side effects
    from sample_factory.train import run_rl
    from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
    from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec
    from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
    from sample_factory.algo.utils.context import global_model_factory
    from sample_factory.envs.env_utils import register_env

    import functools

    # Now safely call setup and training
    register_vizdoom_components()

    cfg = parse_vizdoom_cfg([
        "--env=doom_health_gathering_supreme",
        "--num_workers=36",
        "--num_envs_per_worker=24",
        "--train_for_env_steps=400000000",
        "--experiment=default_experiment",
        "--train_dir=train_dir",
        "--device=gpu",
    ])

    run_rl(cfg)

