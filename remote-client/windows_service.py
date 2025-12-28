#!/usr/bin/env python3
"""
Windows Service Wrapper for Krokenheimer Remote Training Client
Allows the training scheduler to run as a proper Windows service
"""

import sys
import os
import win32serviceutil
import win32service
import win32event
import logging
import json
import time
from pathlib import Path
import subprocess

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from scheduler import TrainingScheduler
except ImportError:
    # If scheduler import fails, we'll handle it in the service
    TrainingScheduler = None

class KrokenheimerTrainingService(win32serviceutil.ServiceFramework):
    _svc_name_ = "KrokenheimerTrainingService"
    _svc_display_name_ = "Krokenheimer Remote Training Service"
    _svc_description_ = "Automatically performs training on remote Discord bot data using local GPU"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.is_running = True

        # Setup logging for service
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'service.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self._svc_name_)

    def SvcStop(self):
        """Called when the service is stopped."""
        self.logger.info("🛑 Service stop requested")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self.is_running = False
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        """Main service execution."""
        self.logger.info("🚀 Krokenheimer Training Service starting...")

        try:
            # Find config file
            config_path = self.find_config_file()
            if not config_path:
                self.logger.error("❌ Config file not found")
                return

            self.logger.info(f"📄 Using config: {config_path}")

            if TrainingScheduler is None:
                self.logger.error("❌ Scheduler module not available")
                return

            # Initialize scheduler
            scheduler = TrainingScheduler(config_path)
            self.logger.info("✅ Training scheduler initialized")

            # Service main loop
            self.run_service_loop(scheduler)

        except Exception as e:
            self.logger.error(f"❌ Service error: {e}", exc_info=True)
        finally:
            self.logger.info("🏁 Krokenheimer Training Service stopped")

    def find_config_file(self):
        """Find the configuration file."""
        possible_paths = [
            "remote_config.json",
            Path(__file__).parent / "remote_config.json",
            Path.home() / "krokenheimer" / "remote_config.json",
            "C:/ProgramData/Krokenheimer/remote_config.json"
        ]

        for path in possible_paths:
            if Path(path).exists():
                return str(path)

        return None

    def run_service_loop(self, scheduler):
        """Main service loop."""
        check_interval = 3600  # Check every hour
        last_training_check = 0

        while self.is_running:
            try:
                current_time = time.time()

                # Check if it's time for a training check
                if current_time - last_training_check >= check_interval:
                    if scheduler.should_check_training():
                        self.logger.info("🔍 Running scheduled training check...")
                        scheduler.run_training_check()
                        last_training_check = current_time
                    else:
                        self.logger.info("⏭️ Not time for training check yet")

                # Wait for stop event or timeout
                result = win32event.WaitForSingleObject(self.hWaitStop, 30000)  # 30 second timeout

                if result == win32event.WAIT_OBJECT_0:
                    # Stop event was signaled
                    break

            except Exception as e:
                self.logger.error(f"❌ Service loop error: {e}", exc_info=True)
                time.sleep(60)  # Wait a minute before retrying

def install_service():
    """Install the Windows service."""
    try:
        win32serviceutil.InstallService(
            KrokenheimerTrainingService,
            KrokenheimerTrainingService._svc_name_,
            KrokenheimerTrainingService._svc_display_name_,
            description=KrokenheimerTrainingService._svc_description_,
            startType=win32service.SERVICE_AUTO_START
        )
        print("✅ Service installed successfully")
        print("   You can start it with: net start KrokenheimerTrainingService")
        print("   Or via Services.msc")
    except Exception as e:
        print(f"❌ Failed to install service: {e}")

def remove_service():
    """Remove the Windows service."""
    try:
        win32serviceutil.RemoveService(KrokenheimerTrainingService._svc_name_)
        print("✅ Service removed successfully")
    except Exception as e:
        print(f"❌ Failed to remove service: {e}")

def main():
    if len(sys.argv) == 1:
        # Started with no command line arguments - run as service
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (KrokenheimerTrainingService._svc_name_, '')
        )
        win32serviceutil.HandleCommandLine(KrokenheimerTrainingService)
    else:
        command = sys.argv[1].lower()

        if command == 'install':
            install_service()
        elif command == 'remove':
            remove_service()
        elif command == 'start':
            win32serviceutil.StartService(KrokenheimerTrainingService._svc_name_)
            print("✅ Service started")
        elif command == 'stop':
            win32serviceutil.StopService(KrokenheimerTrainingService._svc_name_)
            print("✅ Service stopped")
        elif command == 'restart':
            win32serviceutil.RestartService(KrokenheimerTrainingService._svc_name_)
            print("✅ Service restarted")
        else:
            print("🤖 Krokenheimer Training Service")
            print("Usage:")
            print("  python windows_service.py install   - Install as Windows service")
            print("  python windows_service.py remove    - Remove Windows service")
            print("  python windows_service.py start     - Start service")
            print("  python windows_service.py stop      - Stop service")
            print("  python windows_service.py restart   - Restart service")

if __name__ == '__main__':
    # Import servicemanager here to avoid import errors when pywin32 is not available
    try:
        import servicemanager
        main()
    except ImportError:
        print("❌ pywin32 not installed. Windows service functionality not available.")
        print("   Install with: pip install pywin32")
        sys.exit(1)