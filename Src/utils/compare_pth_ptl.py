
import datetime

import jittor as jt
import numpy as np

import torch
def extract_state_dict(model_data):
    """提取状态字典，处理各种可能的存储格式"""
    if isinstance(model_data, dict):
        # 检查是否有常见的包装键
        if 'state_dict' in model_data:
            return model_data['state_dict']
        elif 'model' in model_data:
            return model_data['model']
        elif 'model_state_dict' in model_data:
            return model_data['model_state_dict']
        else:
            # 假设这就是状态字典
            return model_data
    else:
        # 如果不是字典，可能是模型对象
        if hasattr(model_data, 'state_dict'):
            return model_data.state_dict()
        else:
            raise ValueError("无法从提供的数据中提取状态字典")


def normalize_key_names(state_dict):
    """标准化键名，移除常见的前缀"""
    normalized = {}
    for key, value in state_dict.items():
        clean_key = key
        prefixes_to_remove = ['module.', 'model.', '_orig_mod.']
        for prefix in prefixes_to_remove:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
                break
        normalized[clean_key] = value
    return normalized


def to_numpy_safe(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif hasattr(tensor, 'numpy'):  # Jittor Var
        return tensor.numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        return np.array(tensor)


def compare_models_advanced(model1_data, model2_data, tolerance=1e-5, verbose=True):
    """
    高级模型参数对比函数

    Args:
        model1_data: 第一个模型的数据
        model2_data: 第二个模型的数据
        tolerance: 数值对比的容差
        verbose: 是否打印详细信息
    """
    try:
        state_dict1 = extract_state_dict(model1_data)
        state_dict2 = extract_state_dict(model2_data)

        state_dict1 = normalize_key_names(state_dict1)
        state_dict2 = normalize_key_names(state_dict2)

        keys1 = set(state_dict1.keys())
        keys2 = set(state_dict2.keys())

        print(f"模型1包含 {len(keys1)} 个参数")
        print(f"模型2包含 {len(keys2)} 个参数")


        if keys1 == keys2:
            print(" 两个模型的层名完全一致！")
        else:
            print("两个模型的层名不一致！")
            only_in_1 = keys1 - keys2
            only_in_2 = keys2 - keys1
            if only_in_1:
                print(f"  - 只在模型1中存在的层: {sorted(list(only_in_1))}")
            if only_in_2:
                print(f"  - 只在模型2中存在的层: {sorted(list(only_in_2))}")

        # 对比共同的层
        common_keys = sorted(list(keys1.intersection(keys2)))
        print(f"\n--- 对比 {len(common_keys)} 个共同的层 ---")

        shape_mismatch_count = 0
        value_mismatch_count = 0
        perfect_match_count = 0

        for key in common_keys:
            tensor1 = state_dict1[key]
            tensor2 = state_dict2[key]

            # 转换为numpy
            np_tensor1 = to_numpy_safe(tensor1)
            np_tensor2 = to_numpy_safe(tensor2)

            # 对比形状
            if np_tensor1.shape != np_tensor2.shape:
                if verbose:
                    print(f"层 '{key}': 形状不匹配! "
                          f"模型1: {np_tensor1.shape}, 模型2: {np_tensor2.shape}")
                shape_mismatch_count += 1
                continue

            # 对比数值
            diff = np.abs(np_tensor1 - np_tensor2).max()
            if diff > tolerance:
                if verbose:
                    print(f"层 '{key}': 数值差异过大! 最大差值: {diff:.6f}")
                value_mismatch_count += 1
            if diff == 0:
                print(f"层 '{key}': 完全匹配 (差值为0)")
            else:
                print(f"层 '{key}': 完全匹配 (最大差值: {diff:.3e})")



        print(f"共同层数量: {len(common_keys)}")
        print(f"完全匹配: {perfect_match_count}")
        print(f"形状不匹配: {shape_mismatch_count}")
        print(f"数值差异过大: {value_mismatch_count}")

        if shape_mismatch_count == 0 and value_mismatch_count == 0:
            print("两个模型的参数完全一致！")
            return True
        else:
            print("两个模型存在差异")
            return False

    except Exception as e:
        print(f"对比过程中出现错误: {e}")
        return False


# 使用示例
if __name__ == "__main__":

    model1 = torch.load('../backbone/resnet/resnet50-19c8e357.pth', map_location='cpu')
    model2 = jt.load('../backbone/resnet/resnet50.pkl')


    is_identical = compare_models_advanced(model1, model2, tolerance=1e-5, verbose=True)

    if is_identical:
        print("\n两个模型参数完全一致，可以安全地互相替换使用！")
    else:
        print("\n两个模型参数存在差异，请检查转换过程或模型版本！")
