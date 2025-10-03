"""
学習進捗確認スクリプト
Training Progress Checker
"""

import json
import time
from pathlib import Path
import psutil


def check_training_progress():
    """現在の学習進捗を確認"""
    
    progress_file = Path("training_progress.json")
    
    if not progress_file.exists():
        print("⚠️ 学習進捗ファイルが見つかりません")
        return
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logs = data.get('training_log', [])
        
        if not logs:
            print("📊 学習進捗データがありません")
            return
        
        # 最新の進捗情報
        latest = logs[-1]
        
        print("🔍 現在の学習進捗状況")
        print("=" * 50)
        
        # 基本情報
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"📅 確認時刻: {current_time}")
        print(f"🎯 現在のステージ: {latest.get('stage', 'Unknown')}")
        print(f"📊 エポック進行: {latest.get('epoch', 0)}/{latest.get('total_epochs', 0)}")
        
        # 進捗率
        epoch = latest.get('epoch', 0)
        total_epochs = latest.get('total_epochs', 1)
        progress_percent = (epoch / total_epochs) * 100
        print(f"📈 ステージ進捗率: {progress_percent:.1f}%")
        
        # 性能指標
        metrics = latest.get('metrics', {})
        if metrics:
            print(f"🎯 現在の精度: {metrics.get('val_accuracy', 0):.4f}")
            print(f"📉 現在の損失: {metrics.get('val_loss', 0):.4f}")
        
        # 学習時間
        elapsed = latest.get('elapsed_time', 0)
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        print(f"⏱️  経過時間: {hours}時間{minutes}分")
        
        # ステージ別進捗
        print("\n📊 ステージ別進捗:")
        stages = {}
        for log in logs:
            stage = log['stage']
            if stage not in stages or log['epoch'] > stages[stage]['epoch']:
                stages[stage] = log
        
        for stage_name, stage_data in stages.items():
            epoch = stage_data['epoch']
            total = stage_data['total_epochs']
            progress = (epoch / total) * 100
            
            if progress >= 100:
                status = "✅ 完了"
            elif epoch > 0:
                status = f"🔄 進行中 ({progress:.1f}%)"
            else:
                status = "⏳ 待機中"
            
            print(f"   {stage_name}: {status}")
        
        # システムリソース
        print(f"\n💻 システム状況:")
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        print(f"   メモリ使用率: {memory.percent:.1f}%")
        print(f"   CPU使用率: {cpu_percent:.1f}%")
        
        # 予想残り時間
        if latest.get('stage_elapsed', 0) > 0 and epoch > 0:
            avg_time = latest['stage_elapsed'] / epoch
            remaining_epochs = total_epochs - epoch
            eta_seconds = avg_time * remaining_epochs
            eta_minutes = int(eta_seconds // 60)
            eta_hours = int(eta_minutes // 60)
            
            print(f"\n⏰ 推定残り時間:")
            if eta_hours > 0:
                print(f"   現在ステージ: {eta_hours}時間{eta_minutes%60}分")
            else:
                print(f"   現在ステージ: {eta_minutes}分")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 進捗確認エラー: {e}")


def monitor_training_realtime():
    """リアルタイム監視"""
    print("🔍 リアルタイム学習監視開始")
    print("(Ctrl+C で停止)")
    
    try:
        while True:
            print("\033[2J\033[H")  # 画面クリア
            check_training_progress()
            time.sleep(30)  # 30秒間隔
            
    except KeyboardInterrupt:
        print("\n⏹️  監視を停止しました")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--realtime":
        monitor_training_realtime()
    else:
        check_training_progress()