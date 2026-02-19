"""
System lock implementation for security actions.
"""

import platform
import subprocess
import os
from typing import Optional


class LockController:
    """Control system lock functionality."""
    
    def __init__(self, lock_enabled: bool = True):
        self.lock_enabled = lock_enabled
        self.system = platform.system()
    
    def lock_system(self) -> bool:
        """
        Lock the system.
        
        Returns:
            True if lock command executed successfully
        """
        if not self.lock_enabled:
            print("Lock disabled - would lock system here")
            return False
        
        try:
            if self.system == "Windows":
                # Windows lock
                subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=True)
                return True
            elif self.system == "Darwin":
                # macOS lock
                subprocess.run(["pmset", "displaysleepnow"], check=True)
                # Alternative: osascript -e 'tell application "System Events" to keystroke "q" using {control down, command down}'
                return True
            else:
                # Linux - try various methods
                try:
                    # Try gnome-screensaver-command
                    subprocess.run(["gnome-screensaver-command", "--lock"], check=True)
                    return True
                except:
                    try:
                        # Try xdg-screensaver
                        subprocess.run(["xdg-screensaver", "lock"], check=True)
                        return True
                    except:
                        try:
                            # Try i3lock or similar
                            subprocess.run(["i3lock"], check=True)
                            return True
                        except:
                            print("Could not find lock command for Linux")
                            return False
        except subprocess.CalledProcessError as e:
            print(f"Error locking system: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error locking system: {e}")
            return False
    
    def enable_lock(self):
        """Enable lock functionality."""
        self.lock_enabled = True
    
    def disable_lock(self):
        """Disable lock functionality."""
        self.lock_enabled = False
    
    def is_lock_enabled(self) -> bool:
        """Check if lock is enabled."""
        return self.lock_enabled


class SecurityActionExecutor:
    """Execute security actions based on decisions."""
    
    def __init__(self, lock_controller: Optional[LockController] = None):
        self.lock_controller = lock_controller or LockController()
        self.action_history = []
    
    def execute_action(self, decision: dict):
        """
        Execute security action based on decision.
        
        Args:
            decision: Decision dictionary from ThresholdLogic
        """
        action = decision.get('action')
        should_lock = decision.get('should_lock', False)
        
        if should_lock and action == 'lock':
            print(f"SECURITY ALERT: Locking system - Confidence: {decision.get('confidence', 0):.2f}")
            success = self.lock_controller.lock_system()
            if success:
                print("System locked successfully")
            else:
                print("Failed to lock system")
        elif action == 'monitor':
            print(f"MONITORING: Suspicious activity detected - Confidence: {decision.get('confidence', 0):.2f}")
        elif action == 'allow':
            # Normal operation
            pass
        
        # Record action
        self.action_history.append({
            'action': action,
            'timestamp': decision.get('timestamp'),
            'confidence': decision.get('confidence'),
            'user_match': decision.get('user_match')
        })
    
    def get_action_history(self, limit: int = 100):
        """Get recent action history."""
        return self.action_history[-limit:]
