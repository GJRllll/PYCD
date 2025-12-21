# PYCD/example_rcd.py
import argparse
from wandb_train_test import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练RCD模型')
    parser.add_argument('--model_name', type=str, default='rcd')
    
    # 数据集设置
    parser.add_argument('--dataset', type=str, default='assist2009')

    # 模型结构参数
    parser.add_argument('--emb_dim', type=int, default=64, help='Embedding维度')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=0)

    # 是否使用WandB记录
    parser.add_argument('--use_wandb', type=int, default=1)

    # 保存目录
    parser.add_argument('--save_dir', type=str, default=None, help='模型保存目录')

    args = parser.parse_args()
    params = vars(args)
    main(params)

