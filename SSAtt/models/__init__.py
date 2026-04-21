import importlib


def build_model(model_name, dataset_name, epochs):
    dataset_to_class = {
        'MI': 'SSAtt_bci',
        'SSVEP': 'SSAtt_mamem',
        'ERN': 'SSAtt_cha'
    }

    if dataset_name not in dataset_to_class:
        raise ValueError(f"未注册的数据集类型: {dataset_name}")

    class_name = dataset_to_class[dataset_name]

    try:
        module = importlib.import_module(f"SSAtt.models.{model_name}")
        model_class = getattr(module, class_name)
    except ImportError:
        raise ImportError(f"无法导入文件 SSAtt/models/{model_name}.py，请检查文件是否存在。")
    except AttributeError:
        raise AttributeError(f"在 {model_name}.py 中找不到类 {class_name}。")

    return model_class(epochs)