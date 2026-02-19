"""
STEP 2: Keyboard Behavior Capture (NO ML)
=========================================

This module captures raw keyboard typing behavior data.
NO machine learning is performed here - only clean data collection.

Captures:
- Key press time
- Key release time
- Key hold duration
- Inter-key delay (time between keystrokes)

All data is stored as raw CSV logs for later feature engineering.
"""

import time
import csv
import os
from typing import List, Optional
from dataclasses import dataclass
from collections import defaultdict

# pynput is required for keyboard capture
try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    keyboard = None


@dataclass
class KeyEvent:
    """
    Represents a single keyboard event (press or release).
    This is RAW data - no ML processing.
    """
    timestamp: float      # Unix timestamp
    key: str              # Key identifier (e.g., 'a', 'space', 'enter')
    event_type: str      # 'press' or 'release'
    key_code: Optional[int] = None  # Optional key code


@dataclass
class Keystroke:
    """
    Represents a complete keystroke (press + release).
    Contains timing information for behavioral analysis.
    """
    key: str              # Key identifier
    press_time: float     # When key was pressed
    release_time: float    # When key was released
    hold_duration: float   # How long key was held (release_time - press_time)
    key_code: Optional[int] = None
    inter_key_delay: Optional[float] = None  # Time since previous keystroke release


class KeyboardCapture:
    """
    Captures keyboard typing behavior for behavioral authentication.
    
    This is STEP 2 of the implementation - pure data collection.
    No feature engineering, no ML - just clean, stable raw data.
    
    Requirements:
    - pynput library must be installed: pip install pynput
    """
    
    def __init__(self, log_file: str = "data/raw/keyboard_logs.csv"):
        """
        Initialize keyboard capture.
        
        Args:
            log_file: Path to CSV file for storing raw keyboard events
        """
        if not KEYBOARD_AVAILABLE:
            raise ImportError(
                "pynput library is required for keyboard capture. "
                "Install it with: pip install pynput"
            )
        
        self.log_file = log_file
        self.is_capturing: bool = False
        self.listener: Optional[keyboard.Listener] = None
        self.events_buffer: List[KeyEvent] = []
        self.keystrokes_buffer: List[Keystroke] = []
        
        # Track pressed keys to calculate hold duration
        self.pressed_keys: dict = {}  # key_str -> press_time
        
        # Track last release time for inter-key delay calculation
        self.last_release_time: Optional[float] = None
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def _normalize_key(self, key) -> str:
        """
        Normalize key representation to consistent string format.
        
        Args:
            key: pynput key object
        
        Returns:
            Normalized key string
        """
        try:
            # Handle special keys
            if hasattr(key, 'char') and key.char is not None:
                return key.char
            else:
                # Special key (e.g., Key.space, Key.enter)
                key_str = str(key)
                if key_str.startswith("Key."):
                    return key_str.replace("Key.", "").lower()
                return key_str.lower()
        except:
            return str(key)
    
    def _get_key_code(self, key) -> Optional[int]:
        """Extract key code if available."""
        try:
            if hasattr(key, 'value'):
                return key.value.vk if hasattr(key.value, 'vk') else None
            return None
        except:
            return None
    
    def on_press(self, key):
        """
        Handle key press event.
        Called automatically by pynput listener.
        """
        if not self.is_capturing:
            return
        
        try:
            key_str = self._normalize_key(key)
            key_code = self._get_key_code(key)
            timestamp = time.time()
            
            # Create press event
            event = KeyEvent(
                timestamp=timestamp,
                key=key_str,
                event_type='press',
                key_code=key_code
            )
            
            # Store event
            self.events_buffer.append(event)
            self.pressed_keys[key_str] = timestamp
            self.save_event(event)
            
        except Exception as e:
            # Silent error handling - don't interrupt capture
            pass
    
    def on_release(self, key):
        """
        Handle key release event.
        Called automatically by pynput listener.
        """
        if not self.is_capturing:
            return
        
        try:
            key_str = self._normalize_key(key)
            key_code = self._get_key_code(key)
            timestamp = time.time()
            
            # Create release event
            event = KeyEvent(
                timestamp=timestamp,
                key=key_str,
                event_type='release',
                key_code=key_code
            )
            
            # Store event
            self.events_buffer.append(event)
            
            # Calculate hold duration if we have the press time
            hold_duration = None
            if key_str in self.pressed_keys:
                press_time = self.pressed_keys[key_str]
                hold_duration = timestamp - press_time
                del self.pressed_keys[key_str]
            
            # Calculate inter-key delay (time since last keystroke release)
            inter_key_delay = None
            if self.last_release_time is not None:
                inter_key_delay = timestamp - self.last_release_time
            
            # Create complete keystroke record
            if hold_duration is not None:
                keystroke = Keystroke(
                    key=key_str,
                    press_time=press_time if key_str in self.pressed_keys else timestamp,
                    release_time=timestamp,
                    hold_duration=hold_duration,
                    key_code=key_code,
                    inter_key_delay=inter_key_delay
                )
                self.keystrokes_buffer.append(keystroke)
            
            # Update last release time
            self.last_release_time = timestamp
            
            # Save release event
            self.save_event(event)
            
        except Exception as e:
            # Silent error handling
            pass
    
    def start_capture(self):
        """
        Start keyboard capture.
        Begins listening to keyboard events in the background.
        """
        if not KEYBOARD_AVAILABLE:
            raise ImportError("pynput library is required for keyboard capture")
        
        self.is_capturing = True
        self.events_buffer.clear()
        self.keystrokes_buffer.clear()
        self.pressed_keys.clear()
        self.last_release_time = None
        
        # Initialize CSV file with headers
        try:
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'key',
                    'event_type',
                    'key_code'
                ])
        except Exception as e:
            print(f"Error initializing keyboard log file: {e}")
            return
        
        # Start keyboard listener
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
        
        print(f"[OK] Keyboard capture started. Logging to: {self.log_file}")
        print(f"  Press ESC to stop (if running interactively)")
    
    def save_event(self, event: KeyEvent):
        """
        Save a single keyboard event to CSV file.
        
        Args:
            event: KeyEvent to save
        """
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    event.timestamp,
                    event.key,
                    event.event_type,
                    event.key_code if event.key_code is not None else ''
                ])
        except Exception as e:
            print(f"Error saving keyboard event: {e}")
    
    def stop_capture(self):
        """Stop keyboard capture."""
        self.is_capturing = False
        
        if self.listener:
            self.listener.stop()
            self.listener.join(timeout=1.0)
            self.listener = None
        
        print(f"Keyboard capture stopped. Captured {len(self.keystrokes_buffer)} keystrokes.")
    
    def get_recent_events(self, count: int = 100) -> List[KeyEvent]:
        """
        Get recent keyboard events from buffer.
        
        Args:
            count: Number of recent events to return
        
        Returns:
            List of recent KeyEvent objects
        """
        return self.events_buffer[-count:] if self.events_buffer else []
    
    def get_recent_keystrokes(self, count: int = 100) -> List[Keystroke]:
        """
        Get recent complete keystrokes from buffer.
        
        Args:
            count: Number of recent keystrokes to return
        
        Returns:
            List of recent Keystroke objects
        """
        return self.keystrokes_buffer[-count:] if self.keystrokes_buffer else []
    
    def calculate_inter_key_delays(self, keystrokes: List[Keystroke] = None) -> List[float]:
        """
        Calculate inter-key delays from keystrokes.
        
        Args:
            keystrokes: List of keystrokes (uses buffer if None)
        
        Returns:
            List of inter-key delays in seconds
        """
        if keystrokes is None:
            keystrokes = self.keystrokes_buffer
        
        delays = []
        for i in range(1, len(keystrokes)):
            # Delay = time from previous release to current press
            delay = keystrokes[i].press_time - keystrokes[i-1].release_time
            delays.append(delay)
        
        return delays
    
    def get_keystroke_count(self) -> int:
        """Get total number of keystrokes captured."""
        return len(self.keystrokes_buffer)
    
    def clear_buffer(self):
        """Clear the events buffer (keeps file intact)."""
        self.events_buffer.clear()
        self.keystrokes_buffer.clear()
        self.pressed_keys.clear()


# Test function for development
if __name__ == "__main__":
    print("Testing keyboard capture...")
    print("Type some text for 10 seconds, then press ESC to stop...")
    print("(Note: You may need to run with appropriate permissions)\n")
    
    try:
        capture = KeyboardCapture(log_file="test_keyboard_logs.csv")
        capture.start_capture()
        
        # Wait for ESC key to stop
        def on_press(key):
            try:
                if key == keyboard.Key.esc:
                    capture.stop_capture()
                    return False
            except:
                pass
        
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
        
        print(f"\nCaptured {capture.get_keystroke_count()} keystrokes")
        print(f"Log file: {capture.log_file}")
        
        # Show sample keystrokes
        recent = capture.get_recent_keystrokes(5)
        if recent:
            print("\nSample keystrokes:")
            for ks in recent:
                print(f"  Key: {ks.key}, Hold: {ks.hold_duration:.3f}s, "
                      f"Delay: {ks.inter_key_delay:.3f}s" if ks.inter_key_delay else "N/A")
    
    except ImportError as e:
        print(f"Error: {e}")
        print("Install pynput with: pip install pynput")
    except Exception as e:
        print(f"Error: {e}")