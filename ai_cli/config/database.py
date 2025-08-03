"""Database management for storing API keys, settings, and usage history."""

import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from cryptography.fernet import Fernet
import base64
import os

from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session


Base = declarative_base()


class APIKey(Base):
    """API key storage model."""
    __tablename__ = 'api_keys'
    
    provider = Column(String, primary_key=True)
    encrypted_key = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Setting(Base):
    """Settings storage model."""
    __tablename__ = 'settings'
    
    key = Column(String, primary_key=True)
    value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UsageLog(Base):
    """Usage tracking model."""
    __tablename__ = 'usage_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    command = Column(String, nullable=False)
    tokens_used = Column(Integer, default=0)
    cost_usd = Column(String, default="0.00")
    duration_ms = Column(Integer, default=0)
    success = Column(String, default="true")  # SQLite doesn't have native boolean
    created_at = Column(DateTime, default=datetime.utcnow)


class CommandHistory(Base):
    """Command history model."""
    __tablename__ = 'command_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    command = Column(String, nullable=False)
    args = Column(Text)  # JSON string
    success = Column(String, default="true")
    output_summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Manages SQLite database operations for the AI CLI."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or self._get_default_db_path()
        self._encryption_key: Optional[bytes] = None
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize database
        self._initialize_database()
    
    def _get_default_db_path(self) -> Path:
        """Get the default database file path."""
        home_dir = Path.home()
        config_dir = home_dir / ".syntax"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "syntax.db"
    
    def _initialize_database(self) -> None:
        """Initialize database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def _get_encryption_key(self) -> bytes:
        """Get or create encryption key for API keys."""
        if self._encryption_key is None:
            key_file = self.db_path.parent / ".encryption_key"
            
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self._encryption_key = f.read()
            else:
                # Generate new key
                self._encryption_key = Fernet.generate_key()
                # Store key securely
                with open(key_file, 'wb') as f:
                    f.write(self._encryption_key)
                # Restrict file permissions
                os.chmod(key_file, 0o600)
        
        return self._encryption_key
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt a string value."""
        key = self._get_encryption_key()
        fernet = Fernet(key)
        encrypted = fernet.encrypt(value.encode())
        return base64.b64encode(encrypted).decode()
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt an encrypted string value."""
        key = self._get_encryption_key()
        fernet = Fernet(key)
        encrypted_bytes = base64.b64decode(encrypted_value.encode())
        decrypted = fernet.decrypt(encrypted_bytes)
        return decrypted.decode()
    
    def store_api_key(self, provider: str, api_key: str) -> bool:
        """Store an encrypted API key for a provider."""
        try:
            encrypted_key = self._encrypt_value(api_key)
            
            with self.SessionLocal() as session:
                # Check if key already exists
                existing_key = session.query(APIKey).filter_by(provider=provider).first()
                
                if existing_key:
                    existing_key.encrypted_key = encrypted_key
                    existing_key.updated_at = datetime.utcnow()
                else:
                    new_key = APIKey(
                        provider=provider,
                        encrypted_key=encrypted_key
                    )
                    session.add(new_key)
                
                session.commit()
                return True
                
        except Exception as e:
            print(f"Error storing API key: {e}")
            return False
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Retrieve and decrypt an API key for a provider."""
        try:
            with self.SessionLocal() as session:
                api_key_record = session.query(APIKey).filter_by(provider=provider).first()
                
                if api_key_record:
                    return self._decrypt_value(api_key_record.encrypted_key)
                
                return None
                
        except Exception as e:
            print(f"Error retrieving API key: {e}")
            return None
    
    def list_providers(self) -> List[str]:
        """List all providers with stored API keys."""
        try:
            with self.SessionLocal() as session:
                providers = session.query(APIKey.provider).all()
                return [provider[0] for provider in providers]
                
        except Exception as e:
            print(f"Error listing providers: {e}")
            return []
    
    def delete_api_key(self, provider: str) -> bool:
        """Delete an API key for a provider."""
        try:
            with self.SessionLocal() as session:
                api_key_record = session.query(APIKey).filter_by(provider=provider).first()
                
                if api_key_record:
                    session.delete(api_key_record)
                    session.commit()
                    return True
                
                return False
                
        except Exception as e:
            print(f"Error deleting API key: {e}")
            return False
    
    def store_setting(self, key: str, value: Any) -> bool:
        """Store a setting value."""
        try:
            # Convert value to JSON string
            value_str = json.dumps(value) if not isinstance(value, str) else value
            
            with self.SessionLocal() as session:
                existing_setting = session.query(Setting).filter_by(key=key).first()
                
                if existing_setting:
                    existing_setting.value = value_str
                    existing_setting.updated_at = datetime.utcnow()
                else:
                    new_setting = Setting(key=key, value=value_str)
                    session.add(new_setting)
                
                session.commit()
                return True
                
        except Exception as e:
            print(f"Error storing setting: {e}")
            return False
    
    def get_setting(self, key: str) -> Optional[Any]:
        """Retrieve a setting value."""
        try:
            with self.SessionLocal() as session:
                setting_record = session.query(Setting).filter_by(key=key).first()
                
                if setting_record:
                    try:
                        return json.loads(setting_record.value)
                    except json.JSONDecodeError:
                        return setting_record.value
                
                return None
                
        except Exception as e:
            print(f"Error retrieving setting: {e}")
            return None
    
    def log_usage(self, provider: str, model: str, command: str, 
                  tokens_used: int = 0, cost_usd: str = "0.00", 
                  duration_ms: int = 0, success: bool = True) -> bool:
        """Log usage statistics."""
        try:
            with self.SessionLocal() as session:
                usage_log = UsageLog(
                    provider=provider,
                    model=model,
                    command=command,
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                    duration_ms=duration_ms,
                    success="true" if success else "false"
                )
                session.add(usage_log)
                session.commit()
                return True
                
        except Exception as e:
            print(f"Error logging usage: {e}")
            return False
    
    def get_usage_stats(self, provider: Optional[str] = None, 
                       days: int = 30) -> Dict[str, Any]:
        """Get usage statistics."""
        try:
            with self.SessionLocal() as session:
                query = session.query(UsageLog)
                
                if provider:
                    query = query.filter_by(provider=provider)
                
                # Filter by days
                cutoff_date = datetime.utcnow().replace(day=datetime.utcnow().day - days)
                query = query.filter(UsageLog.created_at >= cutoff_date)
                
                logs = query.all()
                
                stats = {
                    'total_requests': len(logs),
                    'total_tokens': sum(log.tokens_used for log in logs),
                    'total_cost': sum(float(log.cost_usd) for log in logs),
                    'average_duration': sum(log.duration_ms for log in logs) / len(logs) if logs else 0,
                    'success_rate': sum(1 for log in logs if log.success == "true") / len(logs) if logs else 0,
                    'by_provider': {},
                    'by_command': {}
                }
                
                # Group by provider
                for log in logs:
                    if log.provider not in stats['by_provider']:
                        stats['by_provider'][log.provider] = {
                            'requests': 0, 'tokens': 0, 'cost': 0.0
                        }
                    stats['by_provider'][log.provider]['requests'] += 1
                    stats['by_provider'][log.provider]['tokens'] += log.tokens_used
                    stats['by_provider'][log.provider]['cost'] += float(log.cost_usd)
                
                # Group by command
                for log in logs:
                    if log.command not in stats['by_command']:
                        stats['by_command'][log.command] = 0
                    stats['by_command'][log.command] += 1
                
                return stats
                
        except Exception as e:
            print(f"Error getting usage stats: {e}")
            return {}
    
    def store_command_history(self, command: str, args: Dict[str, Any], 
                            success: bool = True, output_summary: str = "") -> bool:
        """Store command history."""
        try:
            with self.SessionLocal() as session:
                history_entry = CommandHistory(
                    command=command,
                    args=json.dumps(args),
                    success="true" if success else "false",
                    output_summary=output_summary
                )
                session.add(history_entry)
                session.commit()
                return True
                
        except Exception as e:
            print(f"Error storing command history: {e}")
            return False
    
    def get_command_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent command history."""
        try:
            with self.SessionLocal() as session:
                history_records = (session.query(CommandHistory)
                                 .order_by(CommandHistory.created_at.desc())
                                 .limit(limit)
                                 .all())
                
                history = []
                for record in history_records:
                    history.append({
                        'id': record.id,
                        'command': record.command,
                        'args': json.loads(record.args) if record.args else {},
                        'success': record.success == "true",
                        'output_summary': record.output_summary,
                        'created_at': record.created_at.isoformat()
                    })
                
                return history
                
        except Exception as e:
            print(f"Error getting command history: {e}")
            return []
