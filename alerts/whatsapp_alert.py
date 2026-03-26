"""
WhatsApp notification system (optional, requires API).
"""

import requests
from typing import Dict, Optional
from datetime import datetime


class WhatsAppAlert:
    """Send WhatsApp alerts via API (e.g., Twilio, WhatsApp Business API)."""
    
    def __init__(self,
                 api_url: str = "",
                 api_key: str = "",
                 phone_number: str = ""):
        """
        Initialize WhatsApp alert system.
        
        Args:
            api_url: WhatsApp API endpoint URL
            api_key: API key for authentication
            phone_number: Recipient phone number (with country code)
        """
        self.api_url = api_url
        self.api_key = api_key
        self.phone_number = phone_number
        self.enabled = bool(api_url and api_key and phone_number)
    
    def send_message(self,
                    message: str,
                    phone_number: Optional[str] = None) -> bool:
        """
        Send WhatsApp message.
        
        Args:
            message: Message to send
            phone_number: Recipient phone number (uses default if None)
        
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            print("WhatsApp alerts not configured")
            return False
        
        phone_number = phone_number or self.phone_number
        
        try:
            # Enhanced for CallMeBot / standard REST APIs
            params = {
                'phone': phone_number,
                'text': message,
                'apikey': self.api_key
            }
            
            response = requests.get(
                self.api_url,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"WhatsApp alert sent to {phone_number}")
                return True
            else:
                # Fallback for JSON-based APIs
                payload = {'to': phone_number, 'message': message}
                headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
                return response.status_code == 200
        except Exception as e:
            print(f"Error sending WhatsApp alert: {e}")
            return False
    
    def send_security_alert(self, decision: Dict) -> bool:
        """
        Send security alert via WhatsApp.
        
        Args:
            decision: Security decision dictionary
            
        Returns:
            True if sent success
        """
        if not self.enabled:
            return False
            
        action = decision.get('action', 'unknown').upper()
        confidence = decision.get('confidence', 0)
        risk_level = decision.get('risk_level', 'unknown').upper()
        timestamp = datetime.fromtimestamp(decision.get('timestamp', datetime.now().timestamp())).strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"""
🚨 *SECURITY ALERT* 🚨
*Action*: {action}
*Risk*: {risk_level}
*Confidence*: {confidence:.2%}
*Time*: {timestamp}

A behavioral mismatch was detected on your system. 
Action taken: *{action}*.
        """
        
        return self.send_message(message)
        
        Args:
            decision: Security decision dictionary
        
        Returns:
            True if sent successfully
        """
        action = decision.get('action', 'unknown')
        confidence = decision.get('confidence', 0.0)
        risk_level = decision.get('risk_level', 'unknown')
        
        message = f"""🚨 Security Alert 🚨

Action: {action.upper()}
Risk: {risk_level.upper()}
Confidence: {confidence:.1%}
Time: {datetime.now().strftime('%H:%M:%S')}

AI Behavior Auth System"""
        
        if action == 'lock':
            message = "🔒 SYSTEM LOCKED 🔒\n\n" + message
        
        return self.send_message(message)
    
    def configure(self,
                 api_url: str = None,
                 api_key: str = None,
                 phone_number: str = None):
        """Update WhatsApp configuration."""
        if api_url is not None:
            self.api_url = api_url
        if api_key is not None:
            self.api_key = api_key
        if phone_number is not None:
            self.phone_number = phone_number
        
        self.enabled = bool(self.api_url and self.api_key and self.phone_number)
