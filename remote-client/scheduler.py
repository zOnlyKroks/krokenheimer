#!/usr/bin/env python3
"""
Windows Training Scheduler for Remote Training Client
Handles automatic training scheduling and Windows Task Scheduler integration
"""

import json
import os
import sys
import time
import logging
import schedule
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingScheduler:
    def __init__(self, config_path: str = "remote_config.json"):
        """Initialize the training scheduler."""
        self.config_path = config_path
        self.config = self.load_config()
        self.last_check_file = "last_check.json"

    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file {self.config_path} not found. Run setup first.")
            sys.exit(1)

    def save_last_check(self):
        """Save the last check timestamp."""
        with open(self.last_check_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'success': True
            }, f, indent=2)

    def get_last_check(self):
        """Get the last check timestamp."""
        try:
            with open(self.last_check_file, 'r') as f:
                data = json.load(f)
                return datetime.fromisoformat(data['timestamp'])
        except (FileNotFoundError, KeyError, ValueError):
            return datetime.now() - timedelta(hours=24)  # Default to 24 hours ago

    def should_check_training(self):
        """Check if it's time to check for training opportunities."""
        last_check = self.get_last_check()
        interval_hours = self.config.get('training_interval_hours', 12)

        time_since_last = datetime.now() - last_check
        return time_since_last.total_seconds() >= (interval_hours * 3600)

    def run_training_check(self):
        """Run a single training check."""
        logger.info("🔍 Running scheduled training check...")

        try:
            # Run the remote trainer in check mode
            cmd = [sys.executable, "remote_trainer.py", "--config", self.config_path]

            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            if result.returncode == 0:
                logger.info("✅ Training check completed successfully")
                self.save_last_check()
            else:
                logger.error(f"❌ Training check failed with code {result.returncode}")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("❌ Training check timed out after 2 hours")
        except Exception as e:
            logger.error(f"❌ Training check error: {e}")

    def run_continuous_scheduler(self):
        """Run the continuous scheduler (for daemon mode)."""
        interval_hours = self.config.get('training_interval_hours', 12)

        logger.info(f"🕐 Starting continuous scheduler (check every {interval_hours} hours)")

        # Schedule the training check
        schedule.every(interval_hours).hours.do(self.run_training_check)

        # Run initial check if it's time
        if self.should_check_training():
            logger.info("⏰ Running initial training check...")
            self.run_training_check()

        # Main scheduler loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("🛑 Scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

    def create_windows_task(self):
        """Create a Windows Task Scheduler entry for automatic training."""
        logger.info("🪟 Creating Windows Task Scheduler entry...")

        try:
            # Get the Python executable and script paths
            python_exe = sys.executable
            script_path = os.path.abspath("scheduler.py")
            working_dir = os.getcwd()

            # Task name
            task_name = "KrokenheimerTrainingScheduler"

            # Create the task using schtasks command
            interval_hours = self.config.get('training_interval_hours', 12)

            # Create XML for the task
            xml_content = f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Krokenheimer Remote Training Scheduler</Description>
    <Author>Krokenheimer Bot</Author>
  </RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <Repetition>
        <Interval>PT{interval_hours}H</Interval>
      </Repetition>
      <StartBoundary>2025-01-01T09:00:00</StartBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <DisallowStartOnRemoteAppSession>false</DisallowStartOnRemoteAppSession>
    <UseUnifiedSchedulingEngine>true</UseUnifiedSchedulingEngine>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT2H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions>
    <Exec>
      <Command>{python_exe}</Command>
      <Arguments>scheduler.py --task-scheduler</Arguments>
      <WorkingDirectory>{working_dir}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>'''

            # Save XML to temporary file
            xml_file = f"{task_name}.xml"
            with open(xml_file, 'w', encoding='utf-16') as f:
                f.write(xml_content)

            # Create the task
            create_cmd = [
                "schtasks", "/create",
                "/tn", task_name,
                "/xml", xml_file,
                "/f"  # Force create (overwrite if exists)
            ]

            result = subprocess.run(create_cmd, capture_output=True, text=True)

            # Cleanup XML file
            os.remove(xml_file)

            if result.returncode == 0:
                logger.info(f"✅ Windows task '{task_name}' created successfully")
                logger.info(f"   Task will run every {interval_hours} hours")
                logger.info("   You can manage it via Task Scheduler or:")
                logger.info(f"   - View: schtasks /query /tn {task_name}")
                logger.info(f"   - Delete: schtasks /delete /tn {task_name}")
                logger.info(f"   - Run now: schtasks /run /tn {task_name}")
            else:
                logger.error(f"❌ Failed to create Windows task")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")

        except Exception as e:
            logger.error(f"❌ Error creating Windows task: {e}")

    def remove_windows_task(self):
        """Remove the Windows Task Scheduler entry."""
        task_name = "KrokenheimerTrainingScheduler"

        try:
            cmd = ["schtasks", "/delete", "/tn", task_name, "/f"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"✅ Windows task '{task_name}' removed successfully")
            else:
                logger.info(f"ℹ️ Windows task '{task_name}' not found or already removed")

        except Exception as e:
            logger.error(f"❌ Error removing Windows task: {e}")

def main():
    parser = argparse.ArgumentParser(description="Training Scheduler for Remote Training Client")
    parser.add_argument("--config", default="remote_config.json", help="Config file path")

    # Execution modes
    parser.add_argument("--daemon", action="store_true", help="Run as continuous daemon")
    parser.add_argument("--check-once", action="store_true", help="Run single training check")
    parser.add_argument("--task-scheduler", action="store_true", help="Run single check (for Windows Task Scheduler)")

    # Windows Task Scheduler management
    parser.add_argument("--install-task", action="store_true", help="Install Windows Task Scheduler entry")
    parser.add_argument("--remove-task", action="store_true", help="Remove Windows Task Scheduler entry")

    args = parser.parse_args()

    scheduler = TrainingScheduler(args.config)

    if args.install_task:
        scheduler.create_windows_task()

    elif args.remove_task:
        scheduler.remove_windows_task()

    elif args.check_once or args.task_scheduler:
        # Single training check (used by Task Scheduler)
        if scheduler.should_check_training() or args.task_scheduler:
            scheduler.run_training_check()
        else:
            logger.info("⏭️ Not time for training check yet")

    elif args.daemon:
        # Continuous daemon mode
        scheduler.run_continuous_scheduler()

    else:
        # Default: show status and options
        last_check = scheduler.get_last_check()
        next_check = last_check + timedelta(hours=scheduler.config.get('training_interval_hours', 12))

        print("🤖 Krokenheimer Training Scheduler")
        print(f"   Config: {scheduler.config_path}")
        print(f"   Last check: {last_check.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Next check: {next_check.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Interval: {scheduler.config.get('training_interval_hours', 12)} hours")
        print()
        print("Options:")
        print("  --daemon           Run continuous scheduler")
        print("  --check-once       Run single training check")
        print("  --install-task     Install Windows Task Scheduler entry")
        print("  --remove-task      Remove Windows Task Scheduler entry")

if __name__ == "__main__":
    main()