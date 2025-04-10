from tabpfn import TabPFNClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import shap
import joblib
import os
import time

def main():
    print("开始计算和保存SHAP值...")
    start_time = time.time()
    
    # 加载数据
    print("加载数据...")
    train = pd.read_excel("train.xlsx")
    x_train = train.drop(['Group'], axis=1)
    y_train = train['Group']
    
    # 加载模型
    print("加载模型...")
    model_path = r"model/tabpfn_best_model.pkl"
    tabpfn_model = joblib.load(model_path)
    
    # 获取实际的TabPFN估计器
    if isinstance(tabpfn_model, Pipeline):
        print("Pipeline steps:", tabpfn_model.named_steps.keys())
        tabpfn_estimator = tabpfn_model.named_steps['tabpfn']
        # 使用pipeline的predict方法
        predict_function = tabpfn_model.predict
    else:
        print("Model is not a Pipeline, using direct model")
        tabpfn_estimator = tabpfn_model
        predict_function = tabpfn_model.predict
    
    # 检查特征名列表
    feature_names = list(x_train.columns)
    print(f"Features: {feature_names}")
    
    # 检查是否有部分完成的SHAP值
    partial_shap_path = "partial_shap_values.npz"
    shap_values_path = "saved_shap_values.npz"
    
    # 创建SHAP解释器
    print("创建SHAP解释器...")
    explainer = shap.Explainer(predict_function, x_train)
    
    # 计算SHAP值
    print("开始计算SHAP值，这可能需要几个小时...")
    
    # 使用批处理方式计算SHAP值，每50个样本保存一次
    batch_size = 50
    total_samples = len(x_train)
    all_shap_values = []
    
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        print(f"处理样本 {i+1} 到 {batch_end} (共 {total_samples} 个)...")
        
        # 计算这一批样本的SHAP值
        batch_x = x_train.iloc[i:batch_end]
        batch_shap_values = explainer(batch_x)
        
        # 存储批次结果
        all_shap_values.append(batch_shap_values)
        
        # 每个批次后保存部分结果
        if len(all_shap_values) > 1:
            # 尝试合并已完成的批次
            try:
                partial_results = shap.Explanation.concatenate(all_shap_values)
                np.savez_compressed(partial_shap_path, shap_values=partial_results)
                print(f"已保存部分结果，完成了 {batch_end}/{total_samples} 个样本")
            except Exception as e:
                print(f"保存部分结果时出错: {e}")
                # 继续处理，不中断计算
    
    # 合并所有批次的结果
    if len(all_shap_values) == 1:
        shap_values_numpy = all_shap_values[0]
    else:
        try:
            shap_values_numpy = shap.Explanation.concatenate(all_shap_values)
        except Exception as e:
            print(f"合并SHAP值时出错: {e}")
            print("尝试替代方法...")
            # 如果合并失败，尝试重新计算所有样本的SHAP值
            shap_values_numpy = explainer(x_train)
    
    print(f"SHAP值计算完成，shape: {shap_values_numpy.shape}")
    
    # 保存最终的SHAP值
    print(f"保存SHAP值到 {shap_values_path}")
    np.savez_compressed(shap_values_path, shap_values=shap_values_numpy)
    
    # 保存为更多格式以确保兼容性
    # 1. 保存为pickle文件
    import pickle
    with open("saved_shap_values.pkl", "wb") as f:
        pickle.dump(shap_values_numpy, f)
    
    # 2. 保存基础数值数据为npy文件
    np.save("shap_values_array.npy", shap_values_numpy.values)
    
    # 记录特征名称以备后用
    with open("shap_feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))
    
    # 计算并保存特征重要性
    feature_importance = np.abs(shap_values_numpy.values).mean(0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    feature_importance_df.to_csv("feature_importance.csv", index=False)
    
    end_time = time.time()
    total_time = (end_time - start_time) / 60  # 转换为分钟
    print(f"SHAP值计算和保存完成！总耗时: {total_time:.2f} 分钟")
    print(f"您现在可以运行可视化脚本，它将直接加载保存的SHAP值而不需要重新计算")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()