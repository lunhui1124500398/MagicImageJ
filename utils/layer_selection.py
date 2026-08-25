"""关闭测量工具后，活动图层该还给谁的判定逻辑。

抽成纯函数是为了能脱离 napari/Qt 单测：main.py 里只保留"读当前图层名 → 问这里 →
设置 selection.active"这一层壳。
"""

MEASURE_LAYER_NAME = "Measurements"
BATCH_ROI_LAYER_NAME = "Batch_ROI"


def resolve_layer_after_measure(layer_names, remembered_name=None):
    """测量结束后应该激活哪个图层。

    remembered_name: 进入测量模式之前的活动图层名，None 表示没记住。

    优先还原进测量之前那个图层；它已经不在了就回落到 Batch_ROI —— 用户在批量裁剪
    流程里，能继续画框才是最合理的落点。两者都没有则返回 None，交给 napari 自己决定。
    """
    names = list(layer_names)

    if (
        remembered_name
        and remembered_name != MEASURE_LAYER_NAME
        and remembered_name in names
    ):
        return remembered_name

    if BATCH_ROI_LAYER_NAME in names:
        return BATCH_ROI_LAYER_NAME

    return None
