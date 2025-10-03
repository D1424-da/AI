"""
リアルタイム学習進捗モニター
Real-time Training Progress Monitor
"""

import time
import psutil
import threading
from pathlib import Path
import json


class TrainingMonitor:
    """学習進捗をリアルタイムで監視・表示"""
    
    def __init__(self, log_file="training_progress.json"):
        self.log_file = Path(log_file)
        self.is_monitoring = False
        self.current_stage = None
        self.stage_info = {}
        self.training_start_time = None
        self.stage_start_time = None
        
    def start_monitoring(self, total_stages=3):
        """監視開始"""
        self.is_monitoring = True
        self.training_start_time = time.time()
        self.total_stages = total_stages
        
        print("🔍 学習進捗監視開始")
        print(f"📊 総ステージ数: {total_stages}")
        
        # バックグラウンド監視スレッド開始
        monitor_thread = threading.Thread(target=self._background_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
    def set_current_stage(self, stage_name, stage_params):
        """現在のステージ設定"""
        self.current_stage = stage_name
        self.stage_info = stage_params
        self.stage_start_time = time.time()
        
        print(f"\n🎯 ステージ切り替え: {stage_name}")
        print(f"   データ量: {stage_params.get('subset', 0):,}枚")
        print(f"   エポック数: {stage_params.get('epochs', 0)}")
        print(f"   学習率: {stage_params.get('lr', 0)}")
        
    def update_progress(self, epoch, total_epochs, metrics):
        """進捗更新"""
        if not self.is_monitoring:
            return
            
        current_time = time.time()
        
        # 進捗情報作成
        progress_data = {
            'timestamp': current_time,
            'stage': self.current_stage,
            'epoch': epoch,
            'total_epochs': total_epochs,
            'metrics': metrics,
            'elapsed_time': current_time - self.training_start_time if self.training_start_time else 0,
            'stage_elapsed': current_time - self.stage_start_time if self.stage_start_time else 0
        }
        
        # ログファイル更新
        self._save_progress(progress_data)
        
        # コンソール表示
        self._display_progress(progress_data)
        
    def _display_progress(self, data):
        """進捗をコンソール表示"""
        stage = data['stage']
        epoch = data['epoch']
        total_epochs = data['total_epochs']
        metrics = data['metrics']
        
        # 進捗率計算
        stage_progress = (epoch / total_epochs) * 100
        
        # 進捗バー
        bar_length = 25
        filled = int(bar_length * (epoch / total_epochs))
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # 時刻表示
        current_time = time.strftime("%H:%M:%S")
        
        print(f"\n📊 [{current_time}] {stage} - エポック {epoch}/{total_epochs}")
        print(f"   📈 進捗: [{bar}] {stage_progress:.1f}%")
        
        if metrics:
            acc = metrics.get('accuracy', 0)
            val_acc = metrics.get('val_accuracy', 0)
            loss = metrics.get('loss', 0)
            val_loss = metrics.get('val_loss', 0)
            
            print(f"   🎯 精度: 訓練={acc:.4f} | 検証={val_acc:.4f}")
            print(f"   📉 損失: 訓練={loss:.4f} | 検証={val_loss:.4f}")
        
        # 予想残り時間
        if data['stage_elapsed'] > 0 and epoch > 0:
            avg_time_per_epoch = data['stage_elapsed'] / epoch
            remaining_epochs = total_epochs - epoch
            eta_seconds = avg_time_per_epoch * remaining_epochs
            eta_minutes = int(eta_seconds // 60)
            eta_hours = int(eta_minutes // 60)
            
            if eta_hours > 0:
                print(f"   ⏱️  推定残り時間: {eta_hours}時間{eta_minutes%60}分")
            else:
                print(f"   ⏱️  推定残り時間: {eta_minutes}分")
        
        # メモリ使用量
        memory = psutil.virtual_memory()
        if memory.percent > 75:
            print(f"   ⚠️  メモリ使用率: {memory.percent:.1f}%")
            
        print("   " + "-" * 50)
        
    def _save_progress(self, data):
        """進捗をJSONファイルに保存"""
        try:
            # 既存データ読み込み
            if self.log_file.exists():
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
            else:
                log_data = {'training_log': []}
            
            # 新しいデータ追加
            log_data['training_log'].append(data)
            
            # ファイル保存（最新100件のみ保持）
            log_data['training_log'] = log_data['training_log'][-100:]
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ ログ保存エラー: {e}")
            
    def _background_monitor(self):
        """バックグラウンド監視"""
        while self.is_monitoring:
            try:
                # システムリソース監視
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                if memory.percent > 90:
                    print(f"🚨 メモリ警告: {memory.percent:.1f}%")
                    
                if cpu_percent > 95:
                    print(f"🚨 CPU警告: {cpu_percent:.1f}%")
                    
                time.sleep(30)  # 30秒間隔
                
            except Exception as e:
                print(f"⚠️ 監視エラー: {e}")
                break
                
    def stop_monitoring(self):
        """監視停止"""
        self.is_monitoring = False
        total_time = time.time() - self.training_start_time if self.training_start_time else 0
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        print(f"\n✅ 学習完了!")
        print(f"📊 総学習時間: {hours}時間{minutes}分")
        print(f"📁 ログファイル: {self.log_file}")
        
    def get_summary(self):
        """学習サマリー取得"""
        if not self.log_file.exists():
            return None
            
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logs = data.get('training_log', [])
            if not logs:
                return None
                
            # 最新の各ステージ情報
            stages = {}
            for log in logs:
                stage = log['stage']
                if stage not in stages or log['epoch'] > stages[stage]['epoch']:
                    stages[stage] = log
                    
            return {
                'total_time': logs[-1]['elapsed_time'] if logs else 0,
                'stages': stages,
                'total_logs': len(logs)
            }
            
        except Exception as e:
            print(f"⚠️ サマリー取得エラー: {e}")
            return None


# グローバル監視インスタンス
training_monitor = TrainingMonitor()


def start_training_monitor(total_stages=3):
    """学習監視開始（外部から呼び出し用）"""
    training_monitor.start_monitoring(total_stages)
    

def update_training_progress(stage_name, epoch, total_epochs, metrics):
    """学習進捗更新（外部から呼び出し用）"""
    training_monitor.update_progress(epoch, total_epochs, metrics)
    

def set_training_stage(stage_name, stage_params):
    """学習ステージ設定（外部から呼び出し用）"""
    training_monitor.set_current_stage(stage_name, stage_params)


def stop_training_monitor():
    """学習監視停止（外部から呼び出し用）"""
    training_monitor.stop_monitoring()