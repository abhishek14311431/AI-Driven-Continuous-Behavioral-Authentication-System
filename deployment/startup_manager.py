"""
Auto-start on boot functionality for different operating systems.
"""

import platform
import os
import sys
from pathlib import Path


class StartupManager:
    """Manage auto-start on system boot."""
    
    def __init__(self, script_path: str = None):
        """
        Initialize startup manager.
        
        Args:
            script_path: Path to the main script to run on startup
        """
        self.system = platform.system()
        self.script_path = script_path or self._get_default_script_path()
    
    def _get_default_script_path(self) -> str:
        """Get default script path."""
        # Assume main.py is in the project root
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / "main.py")
    
    def install_startup(self) -> bool:
        """Install startup script."""
        try:
            if self.system == "Windows":
                return self._install_windows()
            elif self.system == "Darwin":
                return self._install_macos()
            else:
                return self._install_linux()
        except Exception as e:
            print(f"Error installing startup: {e}")
            return False
    
    def uninstall_startup(self) -> bool:
        """Uninstall startup script."""
        try:
            if self.system == "Windows":
                return self._uninstall_windows()
            elif self.system == "Darwin":
                return self._uninstall_macos()
            else:
                return self._uninstall_linux()
        except Exception as e:
            print(f"Error uninstalling startup: {e}")
            return False
    
    def _install_windows(self) -> bool:
        """Install Windows startup (Task Scheduler or Startup folder)."""
        try:
            import win32com.client
            
            # Use Task Scheduler
            scheduler = win32com.client.Dispatch("Schedule.Service")
            scheduler.Connect()
            
            root_folder = scheduler.GetFolder("\\")
            task_def = scheduler.NewTask(0)
            
            # Create trigger (at logon)
            trigger = task_def.Triggers.Create(9)  # TASK_TRIGGER_LOGON
            trigger.Enabled = True
            
            # Create action (run Python script)
            action = task_def.Actions.Create(0)  # TASK_ACTION_EXEC
            action.Path = sys.executable
            action.Arguments = f'"{self.script_path}"'
            
            # Register task
            TASK_CREATE_OR_UPDATE = 6
            root_folder.RegisterTaskDefinition(
                "AIBehaviorAuth",
                task_def,
                TASK_CREATE_OR_UPDATE,
                None,
                None,
                3  # TASK_LOGON_SERVICE_ACCOUNT
            )
            
            print("Windows startup installed successfully")
            return True
        except ImportError:
            # Fallback to Startup folder
            startup_folder = os.path.join(
                os.getenv('APPDATA'),
                'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup'
            )
            os.makedirs(startup_folder, exist_ok=True)
            
            # Create batch file
            batch_file = os.path.join(startup_folder, 'ai_behavior_auth.bat')
            with open(batch_file, 'w') as f:
                f.write(f'@echo off\n')
                f.write(f'cd /d "{os.path.dirname(self.script_path)}"\n')
                f.write(f'"{sys.executable}" "{self.script_path}"\n')
            
            print("Windows startup installed (Startup folder)")
            return True
        except Exception as e:
            print(f"Error installing Windows startup: {e}")
            return False
    
    def _uninstall_windows(self) -> bool:
        """Uninstall Windows startup."""
        try:
            import win32com.client
            
            scheduler = win32com.client.Dispatch("Schedule.Service")
            scheduler.Connect()
            root_folder = scheduler.GetFolder("\\")
            root_folder.DeleteTask("AIBehaviorAuth", 0)
            
            print("Windows startup uninstalled")
            return True
        except:
            # Remove from Startup folder
            startup_folder = os.path.join(
                os.getenv('APPDATA'),
                'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup'
            )
            batch_file = os.path.join(startup_folder, 'ai_behavior_auth.bat')
            if os.path.exists(batch_file):
                os.remove(batch_file)
                print("Windows startup uninstalled (Startup folder)")
                return True
            return False
    
    def _install_macos(self) -> bool:
        """Install macOS startup (LaunchAgent)."""
        try:
            home = os.path.expanduser("~")
            launch_agents_dir = os.path.join(home, "Library", "LaunchAgents")
            os.makedirs(launch_agents_dir, exist_ok=True)
            
            plist_file = os.path.join(launch_agents_dir, "com.aibehaviorauth.plist")
            
            plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.aibehaviorauth</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>{self.script_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>"""
            
            with open(plist_file, 'w') as f:
                f.write(plist_content)
            
            # Load the launch agent
            os.system(f"launchctl load {plist_file}")
            
            print("macOS startup installed successfully")
            return True
        except Exception as e:
            print(f"Error installing macOS startup: {e}")
            return False
    
    def _uninstall_macos(self) -> bool:
        """Uninstall macOS startup."""
        try:
            home = os.path.expanduser("~")
            plist_file = os.path.join(home, "Library", "LaunchAgents", "com.aibehaviorauth.plist")
            
            if os.path.exists(plist_file):
                os.system(f"launchctl unload {plist_file}")
                os.remove(plist_file)
                print("macOS startup uninstalled")
                return True
            return False
        except Exception as e:
            print(f"Error uninstalling macOS startup: {e}")
            return False
    
    def _install_linux(self) -> bool:
        """Install Linux startup (systemd service or .desktop file)."""
        try:
            # Try systemd first
            if os.path.exists("/etc/systemd"):
                service_file = "/etc/systemd/system/ai-behavior-auth.service"
                service_content = f"""[Unit]
Description=AI Behavior Authentication Service
After=network.target

[Service]
Type=simple
User={os.getenv('USER')}
WorkingDirectory={os.path.dirname(self.script_path)}
ExecStart={sys.executable} {self.script_path}
Restart=always

[Install]
WantedBy=multi-user.target
"""
                with open(service_file, 'w') as f:
                    f.write(service_content)
                
                os.system("systemctl daemon-reload")
                os.system("systemctl enable ai-behavior-auth.service")
                
                print("Linux startup installed (systemd)")
                return True
            else:
                # Fallback to autostart directory
                home = os.path.expanduser("~")
                autostart_dir = os.path.join(home, ".config", "autostart")
                os.makedirs(autostart_dir, exist_ok=True)
                
                desktop_file = os.path.join(autostart_dir, "ai-behavior-auth.desktop")
                desktop_content = f"""[Desktop Entry]
Type=Application
Name=AI Behavior Auth
Exec={sys.executable} {self.script_path}
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
"""
                with open(desktop_file, 'w') as f:
                    f.write(desktop_content)
                
                print("Linux startup installed (autostart)")
                return True
        except Exception as e:
            print(f"Error installing Linux startup: {e}")
            return False
    
    def _uninstall_linux(self) -> bool:
        """Uninstall Linux startup."""
        try:
            # Try systemd
            if os.path.exists("/etc/systemd/system/ai-behavior-auth.service"):
                os.system("systemctl disable ai-behavior-auth.service")
                os.remove("/etc/systemd/system/ai-behavior-auth.service")
                os.system("systemctl daemon-reload")
                print("Linux startup uninstalled (systemd)")
                return True
            else:
                # Remove autostart file
                home = os.path.expanduser("~")
                desktop_file = os.path.join(home, ".config", "autostart", "ai-behavior-auth.desktop")
                if os.path.exists(desktop_file):
                    os.remove(desktop_file)
                    print("Linux startup uninstalled (autostart)")
                    return True
            return False
        except Exception as e:
            print(f"Error uninstalling Linux startup: {e}")
            return False
