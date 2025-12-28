#!/usr/bin/env python3
"""
Remote Training Client for Windows 11 + RX 5700 XT
This script runs on the Windows machine and connects to the main Discord bot
to fetch training data and perform model training locally.
"""

import json
import os
import requests
import subprocess
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('remote_trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RemoteTrainerClient:
    def __init__(self, config_path: str = "remote_config.json"):
        """Initialize the remote training client."""
        self.config = self.load_config(config_path)
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f"Bearer {self.config['auth_token']}",
            'User-Agent': 'RemoteTrainerClient/1.0'
        })

        # Create local directories
        self.data_dir = Path("./data")
        self.models_dir = Path("./models")
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            required_keys = ['bot_host', 'bot_port', 'auth_token']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required config key: {key}")

            return config
        except FileNotFoundError:
            logger.error(f"Config file {config_path} not found. Creating template.")
            template = {
                "bot_host": "your.main.bot.ip",
                "bot_port": 3000,
                "auth_token": "your_secure_auth_token_here",
                "training_interval_hours": 12,
                "min_messages_threshold": 1000,
                "gpu_enabled": True,
                "gpu_type": "rocm",
                "max_epochs": 10,
                "upload_trained_model": True
            }
            with open(config_path, 'w') as f:
                json.dump(template, f, indent=2)
            logger.info(f"Created template config at {config_path}. Please edit and restart.")
            exit(1)

    def test_connection(self) -> bool:
        """Test connection to the main bot API."""
        try:
            response = self.session.get(
                f"http://{self.config['bot_host']}:{self.config['bot_port']}/api/health",
                timeout=10
            )
            if response.status_code == 200:
                logger.info("[SUCCESS] Successfully connected to main bot API")
                return True
            else:
                logger.error(f"[ERROR] API returned status code: {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"[ERROR] Connection failed: {e}")
            return False

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status from main bot."""
        try:
            response = self.session.get(
                f"http://{self.config['bot_host']}:{self.config['bot_port']}/api/training/status",
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get training status: {e}")
            return {}

    def should_start_training(self) -> tuple[bool, str]:
        """Check if training should be started."""
        status = self.get_training_status()

        if not status:
            return False, "Could not get status from main bot"

        if status.get('training_in_progress', False):
            return False, "Training already in progress"

        message_count = status.get('total_messages', 0)
        last_train_count = status.get('last_train_message_count', 0)
        new_messages = message_count - last_train_count

        min_threshold = self.config.get('min_messages_threshold', 1000)

        if new_messages < min_threshold:
            return False, f"Only {new_messages} new messages (need {min_threshold})"

        # Check if enough time has passed since last training
        last_train_date = status.get('last_train_date')
        if last_train_date:
            # Parse ISO timestamp and check interval
            from datetime import datetime, timedelta
            last_train = datetime.fromisoformat(last_train_date.replace('Z', '+00:00'))
            interval_hours = self.config.get('training_interval_hours', 12)
            if datetime.now() - last_train < timedelta(hours=interval_hours):
                return False, f"Training too recent (interval: {interval_hours}h)"

        return True, f"Ready to train with {new_messages} new messages"

    def download_training_data(self) -> Optional[str]:
        """Download training data from main bot."""
        logger.info("[DOWNLOAD] Downloading training data...")

        try:
            response = self.session.post(
                f"http://{self.config['bot_host']}:{self.config['bot_port']}/api/training/export",
                json={"format": "jsonl"},
                timeout=120
            )
            response.raise_for_status()

            training_file = self.data_dir / "training_data.jsonl"
            with open(training_file, 'wb') as f:
                f.write(response.content)

            logger.info(f"[SUCCESS] Training data downloaded to {training_file}")
            return str(training_file)

        except requests.RequestException as e:
            logger.error(f"[ERROR] Failed to download training data: {e}")
            return None

    def setup_gpu_environment(self):
        """Setup GPU environment for RX 5700 XT."""
        logger.info("[GPU] Setting up GPU environment for RX 5700 XT...")

        gpu_type = self.config.get('gpu_type', 'rocm').lower()

        if gpu_type == 'rocm':
            # ROCm environment variables for RX 5700 XT
            os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.1.0'  # RX 5700 XT gfx1010
            os.environ['ROCM_PATH'] = r'C:\Program Files\AMD\ROCm\5.7'
            os.environ['HIP_VISIBLE_DEVICES'] = '0'
            logger.info("[GPU] ROCm environment configured for RX 5700 XT")

        elif gpu_type == 'directml':
            # DirectML for Windows fallback
            os.environ['TORCH_DIRECTML_DEVICE'] = '0'
            logger.info("[GPU] DirectML environment configured")

        else:
            logger.warning("[WARNING] Unknown GPU type, using CPU fallback")

    def train_model(self, training_data_path: str) -> Optional[str]:
        """Train model locally using Windows-optimized script."""
        logger.info("[TRAIN] Starting model training...")

        # Setup GPU environment
        self.setup_gpu_environment()

        # Model output path
        model_name = f"krokenheimer_v{int(time.time())}"
        model_output = self.models_dir / model_name
        model_output.mkdir(exist_ok=True)

        # Prepare training command
        cmd = [
            "python", "train_windows.py",
            training_data_path,
            str(model_output),
            "--epochs", str(self.config.get('max_epochs', 10)),
            "--gpu-type", self.config.get('gpu_type', 'rocm')
        ]

        try:
            logger.info(f"[EXEC] Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            if result.returncode == 0:
                logger.info("[SUCCESS] Training completed successfully!")
                return str(model_output)
            else:
                logger.error(f"[ERROR] Training failed with code {result.returncode}")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("[ERROR] Training timed out after 2 hours")
            return None
        except Exception as e:
            logger.error(f"[ERROR] Training error: {e}")
            return None

    def upload_trained_model(self, model_path: str) -> bool:
        """Upload trained model back to main bot."""
        if not self.config.get('upload_trained_model', True):
            logger.info("[UPLOAD] Model upload disabled in config")
            return True

        logger.info(f"[UPLOAD] Uploading trained model from {model_path}...")

        try:
            # Create zip of model directory
            import zipfile
            zip_path = f"{model_path}.zip"

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, model_path)
                        zipf.write(file_path, arcname)

            # Upload zip file
            with open(zip_path, 'rb') as f:
                files = {'model': f}
                response = self.session.post(
                    f"http://{self.config['bot_host']}:{self.config['bot_port']}/api/training/upload",
                    files=files,
                    timeout=300
                )
                response.raise_for_status()

            # Cleanup
            os.remove(zip_path)

            logger.info("[SUCCESS] Model uploaded successfully!")
            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to upload model: {e}")
            return False

    def notify_training_complete(self, success: bool, model_path: Optional[str] = None):
        """Notify main bot that training is complete."""
        try:
            payload = {
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "model_path": model_path,
                "gpu_type": self.config.get('gpu_type', 'rocm'),
                "client_version": "1.0"
            }

            response = self.session.post(
                f"http://{self.config['bot_host']}:{self.config['bot_port']}/api/training/complete",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            logger.info("[SUCCESS] Training completion notification sent")

        except Exception as e:
            logger.error(f"[ERROR] Failed to notify training completion: {e}")

    def run_training_cycle(self) -> bool:
        """Run a complete training cycle."""
        logger.info("[CYCLE] Starting training cycle...")

        # Check if we should train
        should_train, reason = self.should_start_training()
        if not should_train:
            logger.info(f"[SKIP] Skipping training: {reason}")
            return False

        logger.info(f"[START] Training triggered: {reason}")

        # Download training data
        training_data = self.download_training_data()
        if not training_data:
            return False

        # Train model
        model_path = self.train_model(training_data)
        success = model_path is not None

        if success:
            # Upload model back to main bot
            self.upload_trained_model(model_path)

        # Notify completion
        self.notify_training_complete(success, model_path)

        return success

    def run_forever(self):
        """Run training client in daemon mode."""
        logger.info("[DAEMON] Starting remote training client daemon...")

        if not self.test_connection():
            logger.error("[ERROR] Cannot connect to main bot. Exiting.")
            return

        check_interval = 3600  # Check every hour

        while True:
            try:
                self.run_training_cycle()
            except Exception as e:
                logger.error(f"[ERROR] Training cycle error: {e}")

            logger.info(f"[SLEEP] Sleeping for {check_interval} seconds...")
            time.sleep(check_interval)

def main():
    parser = argparse.ArgumentParser(description="Remote Training Client for Windows 11 + RX 5700 XT")
    parser.add_argument("--config", default="remote_config.json", help="Config file path")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--test-connection", action="store_true", help="Test connection only")
    parser.add_argument("--force-train", action="store_true", help="Force training regardless of checks")

    args = parser.parse_args()

    client = RemoteTrainerClient(args.config)

    if args.test_connection:
        success = client.test_connection()
        exit(0 if success else 1)

    if args.force_train:
        success = client.run_training_cycle()
        exit(0 if success else 1)

    if args.daemon:
        client.run_forever()
    else:
        # Single training cycle
        success = client.run_training_cycle()
        exit(0 if success else 1)

if __name__ == "__main__":
    main()