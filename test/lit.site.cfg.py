import os

config.test_source_root = os.path.dirname(__file__)
lit_config.load_config(config, os.path.join(config.test_source_root, 'lit.cfg.py'))
