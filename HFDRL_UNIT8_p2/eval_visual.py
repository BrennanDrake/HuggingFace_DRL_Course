from sample_factory.enjoy import enjoy
from base64 import b64encode
from IPython.display import HTML

from sample_factory.train import run_rl
from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec
from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.envs.env_utils import register_env
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
import functools


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

if __name__ == '__main__':

      env = "doom_health_gathering_supreme"

      register_vizdoom_components()

      hf_username = "BrennanDrake"  # insert your HuggingFace username here

      cfg = parse_vizdoom_cfg(argv=["--env=doom_health_gathering_supreme", "--num_workers=1", "--save_video", "--no_render", "--max_num_episodes=10", "--push_to_hub", f"--hf_repository={hf_username}/rl_course_vizdoom_health_gathering_supreme"], evaluation=True)
      status = enjoy(cfg)

      mp4 = open('./train_dir/default_experiment/replay.mp4','rb').read()
      data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
      HTML("""
      <video width=640 controls>
            <source src="%s" type="video/mp4">
      </video>
      """ % data_url)