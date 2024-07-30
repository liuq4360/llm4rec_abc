import json
import os

_domain = os.environ.get('DOMAIN', 'beauty_product')
__filepath = os.path.dirname(__file__)
__domain_resource_dir = os.path.abspath(os.path.join(__filepath, f'data/{_domain}'))

with open(os.path.join(__domain_resource_dir, 'settings.json')) as f:
    environ = json.load(f)

BEAUTY_INFO_FILE = os.path.abspath(os.path.join(__domain_resource_dir, environ['BEAUTY_INFO_FILE']))
TABLE_COL_DESC_FILE = os.path.abspath(os.path.join(__domain_resource_dir, environ['TABLE_COL_DESC_FILE']))
MODEL_CKPT_FILE = os.path.abspath(os.path.join(__domain_resource_dir, environ['MODEL_CKPT_FILE']))
ITEM_SIM_FILE = os.path.abspath(os.path.join(__domain_resource_dir, environ['ITEM_SIM_FILE']))
USE_COLS = environ['USE_COLS']
CATEGORICAL_COLS = environ['CATEGORICAL_COLS']
