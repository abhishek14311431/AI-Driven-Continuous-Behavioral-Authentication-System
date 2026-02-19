"""
Email notification system for security alerts.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional
from datetime import datetime


class EmailAlert:
    """Send email alerts for security events."""
    
    def __init__(self,
                 smtp_server: str = "smtp.gmail.com",
                 smtp_port: int = 587,
                 sender_email: str = "",
                 sender_password: str = "",
                 recipient_email: str = ""):
        """
        Initialize email alert system.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            sender_email: Sender email address
            sender_password: Sender email password or app password
            recipient_email: Recipient email address
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        self.enabled = bool(sender_email and sender_password and recipient_email)
    
    def send_alert(self,
                  subject: str,
                  message: str,
                  recipient: Optional[str] = None) -> bool:
        """
        Send email alert.
        
        Args:
            subject: Email subject
            message: Email message body
            recipient: Recipient email (uses default if None)
        
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            print("Email alerts not configured")
            return False
        
        recipient = recipient or self.recipient_email
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, recipient, text)
            server.quit()
            
            print(f"Email alert sent to {recipient}")
            return True
        except Exception as e:
            print(f"Error sending email alert: {e}")
            return False
    
    def send_security_alert(self, decision: Dict) -> bool:
        """
        Send security alert based on decision.
        
        Args:
            decision: Security decision dictionary
        
        Returns:
            True if sent successfully
        """
        action = decision.get('action', 'unknown')
        confidence = decision.get('confidence', 0.0)
        user_match = decision.get('user_match', False)
        risk_level = decision.get('risk_level', 'unknown')
        
        subject = f"Security Alert: {action.upper()} - {risk_level.upper()} Risk"
        
        message = f"""
SECURITY ALERT - Behavior Authentication System

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Action: {action.upper()}
Risk Level: {risk_level.upper()}
Confidence Score: {confidence:.2%}
User Match: {'Yes' if user_match else 'No'}
Predicted User ID: {decision.get('predicted_user_id', 'N/A')}
Expected User ID: {decision.get('expected_user_id', 'N/A')}

This is an automated alert from the AI Behavior Authentication System.
"""
        
        if action == 'lock':
            message += "\n⚠️ SYSTEM HAS BEEN LOCKED DUE TO SUSPICIOUS BEHAVIOR ⚠️"
        
        return self.send_alert(subject, message)
    
    def configure(self,
                 smtp_server: str = None,
                 smtp_port: int = None,
                 sender_email: str = None,
                 sender_password: str = None,
                 recipient_email: str = None):
        """Update email configuration."""
        if smtp_server is not None:
            self.smtp_server = smtp_server
        if smtp_port is not None:
            self.smtp_port = smtp_port
        if sender_email is not None:
            self.sender_email = sender_email
        if sender_password is not None:
            self.sender_password = sender_password
        if recipient_email is not None:
            self.recipient_email = recipient_email
        
        self.enabled = bool(self.sender_email and self.sender_password and self.recipient_email)
