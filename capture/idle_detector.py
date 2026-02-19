"""
STEP 1 (Part 3): Idle Detection (NO ML)
========================================

This module detects user absence/inactivity periods.
NO machine learning - just clean data collection.

Tracks idle gaps in user activity for behavioral analysis.
"""

import time
import csv
import os
from typing import Optional
from dataclasses import dataclass
import platform

# Platform-specific imports
try:
    if platform.system() == "Windows":
        import win32api
    elif platform.system() == "Darwin":
        try:
            from Quartz import CGEventSourceSecondsSinceLastEventType, kCGEventSourceStateID
        except ImportError:
            pass
except ImportError:
    pass


@dataclass
class IdleEvent:
    """
    Represents an idle period detection.
    This is RAW data - no ML processing.
    """
    timestamp: float      # Unix timestamp
    idle_duration: float  # Seconds since last activity
    is_idle: bool         # Whether idle threshold is exceeded


class IdleDetector:
    """
    Detects user inactivity/idle periods.
    
    This is STEP 1 of the implementation - pure data collection.
    Used to track gaps in user activity for behavioral analysis.
    """
    
    def __init__(self, log_file: str = "data/raw/idle_logs.csv", idle_threshold: float = 60.0):
        """
        Initialize idle detector.
        
        Args:
            log_file: Path to CSV file for storing idle events
            idle_threshold: Seconds of inactivity to consider as idle (default: 60s)
        """
        self.log_file = log_file
        self.idle_threshold = idle_threshold
        self.is_monitoring = False
        self.last_activity_time = time.time()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def get_idle_time(self) -> float:
        """
        Get seconds since last user activity.
        
        Returns:
            Idle time in seconds
        """
        try:
            if platform.system() == "Windows":
                # Windows - use time-based tracking
                return time.time() - self.last_activity_time
            elif platform.system() == "Darwin":
                # macOS - use Quartz API
                try:
                    idle_time = CGEventSourceSecondsSinceLastEventType(
                        kCGEventSourceStateID.kCGEventSourceStateHIDSystemState,
                        kCGEventSourceStateID.kCGAnyInputEventType
                    )
                    return idle_time
                except:
                    return time.time() - self.last_activity_time
            else:
                # Linux - try xprintidle (milliseconds)
                try:
                    import subprocess
                    result = subprocess.run(['xprintidle'], capture_output=True, text=True, timeout=1)
                    if result.returncode == 0:
                        return float(result.stdout.strip()) / 1000.0
                except:
                    pass
                return time.time() - self.last_activity_time
        except Exception as e:
            # Fallback to time-based tracking
            return time.time() - self.last_activity_time
    
    def update_activity(self):
        """Update last activity timestamp (call when user activity detected)."""
        self.last_activity_time = time.time()
    
    def check_idle(self) -> IdleEvent:
        """
        Check if user is currently idle.
        
        Returns:
            IdleEvent with current idle status
        """
        idle_duration = self.get_idle_time()
        is_idle = idle_duration >= self.idle_threshold
        
        event = IdleEvent(
            timestamp=time.time(),
            idle_duration=idle_duration,
            is_idle=is_idle
        )
        
        return event
    
    def start_monitoring(self, interval: float = 5.0):
        """
        Start monitoring idle state.
        
        Args:
            interval: Check interval in seconds (default: 5.0)
        """
        self.is_monitoring = True
        self.last_activity_time = time.time()
        
        # Initialize CSV file
        try:
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'idle_duration', 'is_idle'])
        except Exception as e:
            print(f"Error initializing idle log file: {e}")
            return
        
        print(f"[OK] Idle detection started. Logging to: {self.log_file}")
        print(f"  Idle threshold: {self.idle_threshold}s, Check interval: {interval}s")
    
    def monitor_loop(self, duration: Optional[float] = None, interval: float = 5.0):
        """
        Run monitoring loop.
        
        Args:
            duration: Duration to monitor in seconds (None = until stopped)
            interval: Check interval in seconds
        """
        start_time = time.time()
        
        while self.is_monitoring:
            event = self.check_idle()
            self.save_event(event)
            
            if duration and (time.time() - start_time) >= duration:
                break
            
            time.sleep(interval)
    
    def save_event(self, event: IdleEvent):
        """
        Save idle event to CSV file.
        
        Args:
            event: IdleEvent to save
        """
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    event.timestamp,
                    event.idle_duration,
                    event.is_idle
                ])
        except Exception as e:
            print(f"Error saving idle event: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.is_monitoring = False
        print("Idle detection stopped.")
    
    def is_user_idle(self) -> bool:
        """
        Quick check if user is currently idle.
        
        Returns:
            True if user is idle, False otherwise
        """
        return self.get_idle_time() >= self.idle_threshold
