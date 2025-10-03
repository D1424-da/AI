#!/bin/bash
# 訓練データ選択実行スクリプト

echo "🔍 訓練データ選択ユーティリティ起動"
echo "=================================="

# データ選択ユーティリティを実行
python data_selector.py training_data

echo ""
echo "選択完了後、以下のコマンドで学習を開始できます："
echo ""
echo "1. インタラクティブ選択:"
echo "   python train_model_memory_efficient.py --interactive"
echo ""
echo "2. コマンドライン指定:"
echo "   python train_model_memory_efficient.py --classes plastic concrete plate"
echo ""
echo "3. 自動選択 (最小1000枚):"
echo "   python train_model_memory_efficient.py --auto-select 1000"
echo ""
echo "4. バランス選択 (5クラス):"
echo "   python train_model_memory_efficient.py --balanced 5"