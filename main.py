"""
Main entry point for AI Behavior Authentication System.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from deployment.background_service import BehaviorAuthService
from deployment.startup_manager import StartupManager
from model.train_model import BehaviorModelTrainer
from model.evaluate_model import ModelEvaluator
from model.adaptive_update import AdaptiveModelUpdater
from utils.logger import setup_logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AI Behavior Authentication System')
    parser.add_argument('--mode', choices=['service', 'train', 'evaluate', 'update', 'install-startup', 'uninstall-startup'],
                       default='service', help='Operation mode')
    parser.add_argument('--config', type=str, default='deployment/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--model-file', type=str, default='model/behavior_model.pkl',
                       help='Model file path')
    parser.add_argument('--features-file', type=str, default='data/processed/behavior_features.csv',
                       help='Features file path')
    
    args = parser.parse_args()
    
    logger = setup_logger()
    
    if args.mode == 'service':
        # Run as background service
        logger.info("Starting behavior authentication service...")
        service = BehaviorAuthService(config_file=args.config)
        
        try:
            service.start()
            
            # Keep running
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping service...")
            service.stop()
        except Exception as e:
            logger.error(f"Error: {e}")
            service.stop()
            sys.exit(1)
    
    elif args.mode == 'train':
        # Train model
        logger.info("Training behavior authentication model...")
        trainer = BehaviorModelTrainer(features_file=args.features_file)
        
        try:
            metrics = trainer.train()
            logger.info(f"Training metrics: {metrics}")
            trainer.save_model()
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Training error: {e}")
            sys.exit(1)
    
    elif args.mode == 'evaluate':
        # Evaluate model
        logger.info("Evaluating behavior authentication model...")
        evaluator = ModelEvaluator(model_file=args.model_file, features_file=args.features_file)
        
        try:
            metrics = evaluator.evaluate()
            evaluator.print_evaluation(metrics)
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            sys.exit(1)
    
    elif args.mode == 'update':
        # Update model
        logger.info("Checking for model update...")
        updater = AdaptiveModelUpdater(features_file=args.features_file)
        
        try:
            info = updater.get_update_info()
            logger.info(f"Update info: {info}")
            
            if updater.should_update():
                logger.info("Performing model update...")
                result = updater.incremental_update()
                logger.info(f"Update result: {result}")
            else:
                logger.info("No update needed at this time")
        except Exception as e:
            logger.error(f"Update error: {e}")
            sys.exit(1)
    
    elif args.mode == 'install-startup':
        # Install startup
        logger.info("Installing startup script...")
        startup_manager = StartupManager()
        
        if startup_manager.install_startup():
            logger.info("Startup installed successfully")
        else:
            logger.error("Failed to install startup")
            sys.exit(1)
    
    elif args.mode == 'uninstall-startup':
        # Uninstall startup
        logger.info("Uninstalling startup script...")
        startup_manager = StartupManager()
        
        if startup_manager.uninstall_startup():
            logger.info("Startup uninstalled successfully")
        else:
            logger.error("Failed to uninstall startup")
            sys.exit(1)


if __name__ == "__main__":
    main()
