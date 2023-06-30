from box import Box
import toml


def get_cfg():
    """
    获取配置文件，详细参数见config.toml
    Returns:
        Box(Toml())
    """
    cfg = toml.load('config.toml')
    cfg = Box(cfg)
    return cfg
