import yaml


def parse_config(conf_file):
    """
    解析yaml配置文件
    Args:
        conf_file:yaml配置文件路径
    Returns:
    解析后的配置对象
    """
    with open(conf_file, 'r') as conf:
        conf = yaml.load(conf)
    return conf
