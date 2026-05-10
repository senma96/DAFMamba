# Python 包的标准做法，使外部代码可以通过 from models import build_model 来使用模型。
# 使用延迟导入避免循环导入问题

def build_model(args):
    from .decoder import build
    return build(args)
