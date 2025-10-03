"""
訓練データ選択ユーティリティ
Training Data Selection Utility
"""

import os
from pathlib import Path
import json


class DataSelector:
    """訓練データ選択クラス"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
    
    def scan_all_classes(self):
        """全ての利用可能なクラスをスキャン"""
        classes_info = {}
        
        if not self.data_dir.exists():
            print(f"❌ データディレクトリが見つかりません: {self.data_dir}")
            return classes_info
        
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                
                # 画像ファイル数をカウント
                extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
                total_files = 0
                file_types = {}
                
                for ext in extensions:
                    files = list(class_dir.glob(ext))
                    count = len(files)
                    if count > 0:
                        file_types[ext] = count
                        total_files += count
                
                if total_files > 0:
                    classes_info[class_name] = {
                        'total_files': total_files,
                        'file_types': file_types,
                        'path': str(class_dir)
                    }
        
        return classes_info
    
    def display_classes_info(self, classes_info=None):
        """クラス情報を表示"""
        if classes_info is None:
            classes_info = self.scan_all_classes()
        
        if not classes_info:
            print("❌ 利用可能なクラスが見つかりません")
            return
        
        print("\n📊 利用可能な訓練データクラス")
        print("="*60)
        
        total_images = 0
        for i, (class_name, info) in enumerate(classes_info.items()):
            count = info['total_files']
            total_images += count
            
            print(f"  {i+1:2d}. {class_name:<20} {count:>8,}枚")
            
            # ファイルタイプ詳細（オプション）
            if len(info['file_types']) > 1:
                type_details = ', '.join([f"{ext}: {cnt}" for ext, cnt in info['file_types'].items()])
                print(f"      ({type_details})")
        
        print("-" * 60)
        print(f"      総計: {len(classes_info)}クラス, {total_images:,}枚")
        
        return classes_info
    
    def select_classes_by_criteria(self, min_images=100, max_classes=None):
        """基準に基づいてクラスを自動選択"""
        classes_info = self.scan_all_classes()
        
        # 最小画像数でフィルタリング
        filtered_classes = {
            name: info for name, info in classes_info.items()
            if info['total_files'] >= min_images
        }
        
        if not filtered_classes:
            print(f"❌ {min_images}枚以上の画像を持つクラスが見つかりません")
            return []
        
        # 画像数でソート（降順）
        sorted_classes = sorted(filtered_classes.items(), key=lambda x: x[1]['total_files'], reverse=True)
        
        # 最大クラス数で制限
        if max_classes:
            sorted_classes = sorted_classes[:max_classes]
        
        selected_classes = [name for name, info in sorted_classes]
        
        print(f"\n✅ 自動選択結果 (最小{min_images}枚以上):")
        total_selected = 0
        for name in selected_classes:
            count = classes_info[name]['total_files']
            total_selected += count
            print(f"   • {name}: {count:,}枚")
        
        print(f"\n📊 選択クラス: {len(selected_classes)}個, 総画像数: {total_selected:,}枚")
        
        return selected_classes
    
    def create_custom_config(self, selected_classes, output_path="custom_training_config.json"):
        """選択されたクラスでカスタム設定ファイルを作成"""
        config = {
            "class_order": selected_classes,
            "selected_classes_count": len(selected_classes),
            "data_directory": str(self.data_dir),
            "created_timestamp": str(Path().resolve()),
            "image_size": [320, 320],
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 0.001,
            "memory_optimization": True
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ カスタム設定ファイル作成: {output_path}")
        return output_path
    
    def select_balanced_classes(self, target_count=5, balance_threshold=0.3):
        """バランスの取れたクラスを選択"""
        classes_info = self.scan_all_classes()
        
        if len(classes_info) < target_count:
            print(f"⚠️ 利用可能クラス数({len(classes_info)})が目標数({target_count})より少ないです")
            return list(classes_info.keys())
        
        # 画像数の中央値を計算
        image_counts = [info['total_files'] for info in classes_info.values()]
        median_count = sorted(image_counts)[len(image_counts) // 2]
        
        # バランス基準: 中央値の±threshold範囲内
        min_count = int(median_count * (1 - balance_threshold))
        max_count = int(median_count * (1 + balance_threshold))
        
        balanced_classes = {
            name: info for name, info in classes_info.items()
            if min_count <= info['total_files'] <= max_count
        }
        
        # 目標数まで選択
        if len(balanced_classes) >= target_count:
            # 中央値に近い順でソート
            sorted_balanced = sorted(
                balanced_classes.items(), 
                key=lambda x: abs(x[1]['total_files'] - median_count)
            )
            selected = [name for name, info in sorted_balanced[:target_count]]
        else:
            # 不足分は全体から補完
            remaining_classes = {
                name: info for name, info in classes_info.items()
                if name not in balanced_classes
            }
            sorted_remaining = sorted(remaining_classes.items(), key=lambda x: x[1]['total_files'], reverse=True)
            
            selected = list(balanced_classes.keys())
            needed = target_count - len(selected)
            selected.extend([name for name, info in sorted_remaining[:needed]])
        
        print(f"\n🎯 バランス選択結果 (中央値: {median_count:,}枚, 許容範囲: {min_count:,}-{max_count:,}枚):")
        total_selected = 0
        for name in selected:
            count = classes_info[name]['total_files']
            total_selected += count
            balance_ratio = count / median_count
            print(f"   • {name}: {count:,}枚 (比率: {balance_ratio:.2f})")
        
        print(f"\n📊 選択クラス: {len(selected)}個, 総画像数: {total_selected:,}枚")
        
        return selected


def main():
    """メイン関数：対話的なデータ選択"""
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "training_data"
    
    selector = DataSelector(data_dir)
    
    print("🔍 訓練データ選択ユーティリティ")
    print("="*50)
    
    # 利用可能クラスを表示
    classes_info = selector.display_classes_info()
    
    if not classes_info:
        return
    
    print("\n選択モード:")
    print("  1. 手動選択")
    print("  2. 基準による自動選択")
    print("  3. バランス選択")
    
    mode = input("\nモードを選択してください (1-3): ").strip()
    
    if mode == '1':
        # 手動選択は既存のselect_classes_interactiveを使用
        print("手動選択は学習時に行われます")
        
    elif mode == '2':
        min_images = int(input("最小画像数を入力してください (デフォルト: 100): ") or "100")
        max_classes = input("最大クラス数を入力してください (空白で制限なし): ").strip()
        max_classes = int(max_classes) if max_classes else None
        
        selected = selector.select_classes_by_criteria(min_images, max_classes)
        if selected:
            config_path = selector.create_custom_config(selected)
            print(f"\n🚀 学習実行コマンド:")
            print(f"python train_model_memory_efficient.py --config {config_path}")
            
    elif mode == '3':
        target_count = int(input("選択クラス数を入力してください (デフォルト: 5): ") or "5")
        balance_threshold = float(input("バランス許容範囲を入力してください (0.0-1.0, デフォルト: 0.3): ") or "0.3")
        
        selected = selector.select_balanced_classes(target_count, balance_threshold)
        if selected:
            config_path = selector.create_custom_config(selected)
            print(f"\n🚀 学習実行コマンド:")
            print(f"python train_model_memory_efficient.py --config {config_path}")
    
    else:
        print("❌ 無効な選択です")


if __name__ == "__main__":
    main()