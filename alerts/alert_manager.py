"""
Alert queue and send logic - manages all alert channels.
"""

import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from queue import Queue
import threading

from .email_alert import EmailAlert
from .whatsapp_alert import WhatsAppAlert


class AlertManager:
    """Manage and queue alerts across multiple channels."""
    
    def __init__(self,
                 alert_queue_file: str = "data/alerts/alert_queue.json",
                 email_config: Optional[Dict] = None,
                 whatsapp_config: Optional[Dict] = None):
        """
        Initialize alert manager.
        
        Args:
            alert_queue_file: File to store queued alerts
            email_config: Email configuration dict
            whatsapp_config: WhatsApp configuration dict
        """
        self.alert_queue_file = alert_queue_file
        self.alert_queue = Queue()
        self.email_alert = None
        self.whatsapp_alert = None
        
        # Initialize email alerts
        if email_config:
            self.email_alert = EmailAlert(**email_config)
        
        # Initialize WhatsApp alerts
        if whatsapp_config:
            self.whatsapp_alert = WhatsAppAlert(**whatsapp_config)
        
        # Load queued alerts from file
        self._load_queue()
        
        # Start background thread for processing queue
        self.processing = False
        self.worker_thread = None
    
    def _load_queue(self):
        """Load queued alerts from file."""
        if os.path.exists(self.alert_queue_file):
            try:
                with open(self.alert_queue_file, 'r') as f:
                    alerts = json.load(f)
                    for alert in alerts:
                        self.alert_queue.put(alert)
            except Exception as e:
                print(f"Error loading alert queue: {e}")
    
    def _save_queue(self):
        """Save queued alerts to file."""
        try:
            os.makedirs(os.path.dirname(self.alert_queue_file), exist_ok=True)
            
            # Convert queue to list
            alerts = []
            temp_queue = Queue()
            while not self.alert_queue.empty():
                alert = self.alert_queue.get()
                alerts.append(alert)
                temp_queue.put(alert)
            
            # Restore queue
            while not temp_queue.empty():
                self.alert_queue.put(temp_queue.get())
            
            # Save to file
            with open(self.alert_queue_file, 'w') as f:
                json.dump(alerts, f, indent=2)
        except Exception as e:
            print(f"Error saving alert queue: {e}")
    
    def queue_alert(self, decision: Dict, channels: List[str] = None):
        """
        Queue an alert for sending.
        
        Args:
            decision: Security decision dictionary
            channels: List of channels to use ['email', 'whatsapp'] (uses all if None)
        """
        channels = channels or ['email', 'whatsapp']
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'channels': channels,
            'sent': False
        }
        
        self.alert_queue.put(alert)
        self._save_queue()
    
    def send_alert(self, decision: Dict, channels: List[str] = None) -> Dict:
        """
        Send alert immediately (and queue if fails).
        
        Args:
            decision: Security decision dictionary
            channels: List of channels to use
        
        Returns:
            Dictionary with send status for each channel
        """
        channels = channels or ['email', 'whatsapp']
        results = {}
        
        if 'email' in channels and self.email_alert:
            results['email'] = self.email_alert.send_security_alert(decision)
        
        if 'whatsapp' in channels and self.whatsapp_alert:
            results['whatsapp'] = self.whatsapp_alert.send_security_alert(decision)
        
        # If any failed, queue for retry
        if not all(results.values()):
            self.queue_alert(decision, channels)
        
        return results
    
    def process_queue(self):
        """Process queued alerts (runs in background thread)."""
        self.processing = True
        
        while self.processing or not self.alert_queue.empty():
            try:
                if not self.alert_queue.empty():
                    alert = self.alert_queue.get(timeout=1)
                    
                    if not alert.get('sent', False):
                        results = self.send_alert(alert['decision'], alert['channels'])
                        
                        # Mark as sent if all succeeded
                        if all(results.values()):
                            alert['sent'] = True
                        else:
                            # Re-queue if failed
                            self.alert_queue.put(alert)
                        
                        self._save_queue()
                else:
                    import time
                    time.sleep(5)  # Wait before checking again
            except Exception as e:
                print(f"Error processing alert queue: {e}")
                import time
                time.sleep(5)
    
    def start_background_processing(self):
        """Start background thread for processing alerts."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.process_queue, daemon=True)
            self.worker_thread.start()
            print("Alert processing started")
    
    def stop_background_processing(self):
        """Stop background processing."""
        self.processing = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        print("Alert processing stopped")
    
    def get_queue_size(self) -> int:
        """Get number of queued alerts."""
        return self.alert_queue.qsize()
    
    def clear_queue(self):
        """Clear all queued alerts."""
        while not self.alert_queue.empty():
            self.alert_queue.get()
        self._save_queue()
