"""
データ階層確認ユーティリティ
Data Structure Verification Utility
"""

import os
from pathlib import Path


def scan_data_structure(base_path="../training_data"):
    """データ階層をスキャンして確認（重複排除）"""
    base_dir = Path(base_path)
    
    print("🔍 データ階層確認")
    print("="*50)
    print(f"📁 ベースディレクトリ: {base_dir.resolve()}")
    
    if not base_dir.exists():
        print(f"❌ ディレクトリが存在しません: {base_dir}")
        return {}
    
    structure = {}
    total_images = 0
    
    # サブディレクトリをスキャン
    for item in base_dir.iterdir():
        if item.is_dir():
            class_name = item.name
            
            # 画像ファイルをカウント（重複排除）
            unique_files = set()
            file_types = {}
            
            for file_path in item.iterdir():
                if file_path.is_file():
                    ext_lower = file_path.suffix.lower()
                    if ext_lower in ['.jpg', '.jpeg', '.png', '.bmp']:
                        unique_files.add(file_path)
                        
                        # 拡張子統計（表示用）
                        if ext_lower not in file_types:
                            file_types[ext_lower] = 0
                        file_types[ext_lower] += 1
            
            file_count = len(unique_files)
            
            if file_count > 0:
                structure[class_name] = {
                    'count': file_count,
                    'types': file_types,
                    'path': str(item)
                }
                total_images += file_count
                
                print(f"   📂 {class_name:<15}: {file_count:>8,}枚")
                
                # ファイル種別詳細（2種類以上ある場合）
                if len(file_types) > 1:
                    type_details = ', '.join([f"{ext}: {cnt}" for ext, cnt in file_types.items()])
                    print(f"      └─ {type_details}")
    
    print("-" * 50)
    print(f"📊 総計: {len(structure)}クラス, {total_images:,}枚")
    
    return structure


def validate_config_classes(structure, config_classes):
    """設定ファイルのクラスが実際に存在するかチェック"""
    print("\n🔍 設定クラス検証")
    print("="*30)
    
    missing_classes = []
    available_classes = []
    
    for class_name in config_classes:
        if class_name in structure:
            count = structure[class_name]['count']
            available_classes.append(class_name)
            print(f"   ✅ {class_name}: {count:,}枚")
        else:
            missing_classes.append(class_name)
            print(f"   ❌ {class_name}: 見つかりません")
    
    if missing_classes:
        print(f"\n⚠️ 不足クラス: {missing_classes}")
        print("利用可能なクラス名を確認してください")
    
    print(f"\n📊 利用可能: {len(available_classes)}/{len(config_classes)}クラス")
    
    return available_classes, missing_classes


def suggest_directory_fix():
    """ディレクトリ構造の修正提案"""
    print("\n💡 データ階層設定のヒント")
    print("="*40)
    print("1. 現在の検索パス: ../training_data/")
    print("2. 期待される構造:")
    print("   training_data/")
    print("   ├── plastic/")
    print("   │   ├── image1.jpg")
    print("   │   └── image2.jpg")
    print("   ├── concrete/")
    print("   │   ├── image1.jpg")
    print("   │   └── image2.jpg")
    print("   └── ...")
    print("")
    print("3. コード修正方法:")
    print("   trainer = MemoryOptimizedPileClassifierTrainer('新しいパス')")


if __name__ == "__main__":
    import sys
    
    # コマンドライン引数でパス指定可能
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "../training_data"
    
    # データ構造スキャン
    structure = scan_data_structure(data_path)
    
    if not structure:
        suggest_directory_fix()
        sys.exit(1)
    
    # 設定ファイルのクラスと照合
    try:
        from config_loader import ConfigLoader
        config = ConfigLoader("config.json")
        
        if config.class_order:
            validate_config_classes(structure, config.class_order)
        else:
            print("\n📝 config.jsonにclass_orderが設定されていません")
            
    except Exception as e:
        print(f"\n⚠️ 設定ファイル読み込みエラー: {e}")
    
    print(f"\n🚀 学習開始コマンド:")
    print(f"python train_model_memory_efficient.py")