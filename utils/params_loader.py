import yaml

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load(path):
	with open(path) as f:
		config = yaml.load(f, Loader=yaml.SafeLoader)  # config is dict
		cfg = AttrDict(config)
	print(cfg)
	return cfg

