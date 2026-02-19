"""
Background service that runs the behavior authentication system.
"""

import time
import threading
import yaml
import os
import pandas as pd
from typing import Optional
from datetime import datetime

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from capture.cursor_capture import CursorCapture
from capture.keyboard_capture import KeyboardCapture
from capture.idle_detector import IdleDetector
from features.feature_fusion import FeatureFusion
from decision.inference_engine import InferenceEngine
from decision.threshold_logic import ThresholdLogic, SecurityDecisionMaker
from decision.lock_controller import LockController, SecurityActionExecutor
from alerts.alert_manager import AlertManager
from model.adaptive_update import AdaptiveModelUpdater
from utils.logger import setup_logger


class BehaviorAuthService:
    """Main background service for behavior authentication."""
    
    def __init__(self, config_file: str = "deployment/config.yaml"):
        """Initialize service with configuration."""
        self.config = self._load_config(config_file)
        self.logger = setup_logger(log_file=self.config.get('service', {}).get('log_file', 'logs/behavior_auth.log'))
        
        # Initialize components
        self.cursor_capture = None
        self.keyboard_capture = None
        self.idle_detector = None
        self.feature_fusion = FeatureFusion()
        self.inference_engine = None
        self.threshold_logic = None
        self.lock_controller = LockController(
            lock_enabled=self.config.get('lock', {}).get('enabled', True)
        )
        self.action_executor = SecurityActionExecutor(self.lock_controller)
        self.alert_manager = None
        self.adaptive_updater = None
        self.decision_maker = None
        
        self.is_running = False
        self.service_thread = None
        
        # Data buffers for sliding window
        self.cursor_buffer = []
        self.keyboard_buffer = []
        self.last_feature_time = time.time()
    
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from YAML file."""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def initialize(self):
        """Initialize all components."""
        try:
            # Initialize capture modules
            self.cursor_capture = CursorCapture()
            self.keyboard_capture = KeyboardCapture()
            self.idle_detector = IdleDetector(
                idle_threshold=self.config.get('capture', {}).get('idle_threshold', 60.0)
            )
            
            # Initialize inference engine
            model_file = self.config.get('model', {}).get('model_file', 'model/behavior_model.pkl')
            scaler_file = self.config.get('model', {}).get('scaler_file', 'model/feature_scaler.pkl')
            
            try:
                self.inference_engine = InferenceEngine(model_file, scaler_file)
            except FileNotFoundError:
                self.logger.warning("Model not found. System will run in data collection mode only.")
            
            # Initialize threshold logic
            security_config = self.config.get('security', {})
            self.threshold_logic = ThresholdLogic(
                allow_threshold=security_config.get('allow_threshold', 0.8),
                monitor_threshold=security_config.get('monitor_threshold', 0.5),
                lock_threshold=security_config.get('lock_threshold', 0.3)
            )
            
            # Initialize decision maker (combines inference + threshold)
            if self.inference_engine:
                self.decision_maker = SecurityDecisionMaker(
                    self.inference_engine,
                    self.threshold_logic
                )
            
            # Initialize adaptive updater
            adaptive_config = self.config.get('adaptive', {})
            self.adaptive_updater = AdaptiveModelUpdater(
                update_threshold=adaptive_config.get('update_threshold', 100),
                min_update_interval=adaptive_config.get('min_update_interval', 3600.0),
                min_confidence_for_training=adaptive_config.get('min_confidence_for_training', 0.85)
            )
            
            # Initialize alert manager
            alert_config = self.config.get('alerts', {})
            if alert_config.get('enabled', False):
                email_config = None
                whatsapp_config = None
                
                if alert_config.get('email', {}).get('enabled', False):
                    email_config = alert_config.get('email', {})
                
                if alert_config.get('whatsapp', {}).get('enabled', False):
                    whatsapp_config = alert_config.get('whatsapp', {})
                
                self.alert_manager = AlertManager(
                    email_config=email_config,
                    whatsapp_config=whatsapp_config
                )
                self.alert_manager.start_background_processing()
            
            self.logger.info("Service initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing service: {e}")
            raise
    
    def start(self):
        """Start the service."""
        if self.is_running:
            self.logger.warning("Service already running")
            return
        
        self.initialize()
        
        # Start capture modules
        self.cursor_capture.start_capture()
        self.keyboard_capture.start_capture()
        self.idle_detector.start_monitoring()
        
        self.is_running = True
        self.service_thread = threading.Thread(target=self._service_loop, daemon=True)
        self.service_thread.start()
        
        self.logger.info("Behavior authentication service started")
    
    def stop(self):
        """Stop the service."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop capture modules
        if self.cursor_capture:
            self.cursor_capture.stop_capture()
        if self.keyboard_capture:
            self.keyboard_capture.stop_capture()
        if self.idle_detector:
            self.idle_detector.stop_monitoring()
        
        # Stop alert manager
        if self.alert_manager:
            self.alert_manager.stop_background_processing()
        
        if self.service_thread:
            self.service_thread.join(timeout=5)
        
        self.logger.info("Behavior authentication service stopped")
    
    def _service_loop(self):
        """Main service loop - uses sliding window (2-5 seconds)."""
        check_interval = self.config.get('service', {}).get('check_interval', 2.0)
        window_size = self.config.get('features', {}).get('window_size', 5.0)
        
        # Set inference engine window size
        if self.inference_engine:
            self.inference_engine.set_window_size(window_size)
        
        while self.is_running:
            try:
                # Collect recent events (sliding window)
                if self.cursor_capture:
                    cursor_events = self.cursor_capture.get_recent_events(1000)
                    if cursor_events:
                        self.cursor_buffer.extend(cursor_events)
                
                if self.keyboard_capture:
                    keyboard_events = self.keyboard_capture.get_recent_events(1000)
                    if keyboard_events:
                        self.keyboard_buffer.extend(keyboard_events)
                
                # Process features using sliding window (every check_interval)
                current_time = time.time()
                if (current_time - self.last_feature_time) >= check_interval:
                    self._process_features(window_size)
                    self.last_feature_time = current_time
                
                # Check idle state
                if self.idle_detector and self.idle_detector.is_user_idle():
                    self.logger.debug("User is idle")
                
                # Check for adaptive update (less frequently)
                if self.adaptive_updater and self.adaptive_updater.should_update():
                    self.logger.info("Triggering adaptive model update...")
                    result = self.adaptive_updater.incremental_update(use_only_verified=True)
                    if result['status'] == 'success':
                        self.logger.info(f"Model updated successfully. Samples used: {result.get('samples_used', 0)}")
                        # Reload model after update
                        if self.inference_engine:
                            self.inference_engine._load_model()
                
                time.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Error in service loop: {e}")
                time.sleep(check_interval)
    
    def _process_features(self, window_size: float):
        """Process features and make security decisions."""
        try:
            # Convert buffers to DataFrames
            cursor_df = None
            keyboard_df = None
            
            if self.cursor_buffer:
                cursor_data = []
                for event in self.cursor_buffer[-1000:]:  # Last 1000 events
                    cursor_data.append({
                        'timestamp': event.timestamp,
                        'x': event.x,
                        'y': event.y,
                        'dx': event.dx,
                        'dy': event.dy,
                        'velocity': event.velocity,
                        'acceleration': event.acceleration
                    })
                cursor_df = pd.DataFrame(cursor_data)
            
            if self.keyboard_buffer:
                keyboard_data = []
                for event in self.keyboard_buffer[-1000:]:  # Last 1000 events
                    keyboard_data.append({
                        'timestamp': event.timestamp,
                        'key': event.key,
                        'event_type': event.event_type,
                        'key_code': event.key_code
                    })
                keyboard_df = pd.DataFrame(keyboard_data)
            
            # Extract features
            features_df = self.feature_fusion.fuse_features(cursor_df, keyboard_df, window_size)
            
            # Make prediction and decision if model is available
            if self.decision_maker:
                try:
                    # Make decision using sliding window
                    decision = self.decision_maker.make_decision(cursor_df, keyboard_df)
                    
                    # CRITICAL: Mark high-confidence owner sessions for adaptive learning
                    if (decision.get('is_owner', False) and 
                        decision.get('confidence', 0) >= 0.85 and
                        self.adaptive_updater):
                        self.adaptive_updater.mark_verified_owner_session(
                            decision['confidence'],
                            decision['timestamp']
                        )
                    
                    # Execute action (lock if not-owner detected)
                    self.action_executor.execute_action(decision)
                    
                    # Send alerts if needed
                    if self.alert_manager and (decision.get('should_lock') or decision.get('should_monitor')):
                        self.alert_manager.send_alert(decision)
                    
                    # Log decision
                    action = decision.get('action', 'unknown')
                    confidence = decision.get('confidence', 0)
                    is_owner = decision.get('is_owner', False)
                    self.logger.info(
                        f"Decision: {action.upper()} | "
                        f"Owner: {is_owner} | "
                        f"Confidence: {confidence:.2%}"
                    )
                    
                    # CRITICAL: Save features with correct label
                    # Label = 1 if owner, 0 if not-owner
                    label = 1 if is_owner and confidence >= 0.5 else 0
                    features_df['label'] = label
                    
                except Exception as e:
                    self.logger.error(f"Error in prediction: {e}")
                    # Save with unknown label if prediction fails
                    features_df['label'] = 0
            else:
                # No model - save as owner data for initial training
                features_df['label'] = 1
            
            # Save features (with label for training)
            self.feature_fusion.save_features(features_df, append=True)
            
            # Clear old buffers (keep last window)
            buffer_size = int(window_size * 100)  # Rough estimate
            if len(self.cursor_buffer) > buffer_size:
                self.cursor_buffer = self.cursor_buffer[-buffer_size:]
            if len(self.keyboard_buffer) > buffer_size:
                self.keyboard_buffer = self.keyboard_buffer[-buffer_size:]
        
        except Exception as e:
            self.logger.error(f"Error processing features: {e}")


def main():
    """Main entry point for background service."""
    service = BehaviorAuthService()
    
    try:
        service.start()
        
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping service...")
        service.stop()
    except Exception as e:
        print(f"Error: {e}")
        service.stop()


if __name__ == "__main__":
    main()
