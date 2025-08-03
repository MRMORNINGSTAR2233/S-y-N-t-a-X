"""Vector store management using ChromaDB for contextual memory and retrieval."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: Optional[Path] = None):
        self.persist_directory = persist_directory or self._get_default_persist_dir()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Create collections
        self._initialize_collections()
    
    def _get_default_persist_dir(self) -> Path:
        """Get default ChromaDB persistence directory."""
        return Path.cwd() / ".chroma"
    
    def _initialize_collections(self) -> None:
        """Initialize ChromaDB collections."""
        # Code snippets collection
        self.code_collection = self.client.get_or_create_collection(
            name="code_snippets",
            embedding_function=self.embedding_function,
            metadata={"description": "Code snippets and context"}
        )
        
        # Conversation history collection
        self.conversation_collection = self.client.get_or_create_collection(
            name="conversations",
            embedding_function=self.embedding_function,
            metadata={"description": "Chat and command history"}
        )
        
        # Project context collection
        self.context_collection = self.client.get_or_create_collection(
            name="project_context",
            embedding_function=self.embedding_function,
            metadata={"description": "Project-specific context and patterns"}
        )
        
        # Error patterns collection
        self.error_collection = self.client.get_or_create_collection(
            name="error_patterns",
            embedding_function=self.embedding_function,
            metadata={"description": "Error patterns and solutions"}
        )
    
    def _generate_doc_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate a unique document ID based on content and metadata."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        metadata_str = json.dumps(metadata, sort_keys=True)
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
        return f"{content_hash}_{metadata_hash}"
    
    def add_code_snippet(self, code: str, file_path: str, language: str, 
                        function_name: Optional[str] = None, 
                        class_name: Optional[str] = None,
                        description: Optional[str] = None) -> bool:
        """Add a code snippet to the vector store."""
        try:
            metadata = {
                "type": "code",
                "file_path": file_path,
                "language": language,
                "function_name": function_name or "",
                "class_name": class_name or "",
                "description": description or "",
                "timestamp": datetime.utcnow().isoformat(),
                "lines": len(code.split('\n'))
            }
            
            doc_id = self._generate_doc_id(code, metadata)
            
            # Check if already exists
            try:
                existing = self.code_collection.get(ids=[doc_id])
                if existing['ids']:
                    return True  # Already exists
            except:
                pass  # Document doesn't exist
            
            self.code_collection.add(
                documents=[code],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            return True
            
        except Exception as e:
            print(f"Error adding code snippet: {e}")
            return False
    
    def search_code(self, query: str, language: Optional[str] = None, 
                   file_path: Optional[str] = None, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for code snippets using semantic similarity."""
        try:
            where_filter = {"type": "code"}
            
            if language:
                where_filter["language"] = language
            if file_path:
                where_filter["file_path"] = file_path
            
            results = self.code_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            formatted_results = []
            for i, doc_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'id': doc_id,
                    'code': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching code: {e}")
            return []
    
    def add_conversation(self, command: str, response: str, context: Dict[str, Any]) -> bool:
        """Add a conversation to the vector store."""
        try:
            conversation_text = f"Command: {command}\nResponse: {response}"
            
            metadata = {
                "type": "conversation",
                "command": command,
                "timestamp": datetime.utcnow().isoformat(),
                "context": json.dumps(context),
                "success": context.get("success", True)
            }
            
            doc_id = self._generate_doc_id(conversation_text, metadata)
            
            self.conversation_collection.add(
                documents=[conversation_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            return True
            
        except Exception as e:
            print(f"Error adding conversation: {e}")
            return False
    
    def search_conversations(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search conversation history for relevant context."""
        try:
            results = self.conversation_collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"type": "conversation"}
            )
            
            formatted_results = []
            for i, doc_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'id': doc_id,
                    'conversation': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching conversations: {e}")
            return []
    
    def add_project_context(self, content: str, context_type: str, 
                          file_paths: List[str], tags: List[str] = None) -> bool:
        """Add project-specific context information."""
        try:
            metadata = {
                "type": "project_context",
                "context_type": context_type,
                "file_paths": json.dumps(file_paths),
                "tags": json.dumps(tags or []),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            doc_id = self._generate_doc_id(content, metadata)
            
            self.context_collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            return True
            
        except Exception as e:
            print(f"Error adding project context: {e}")
            return False
    
    def search_project_context(self, query: str, context_type: Optional[str] = None, 
                             n_results: int = 10) -> List[Dict[str, Any]]:
        """Search project context for relevant information."""
        try:
            where_filter = {"type": "project_context"}
            if context_type:
                where_filter["context_type"] = context_type
            
            results = self.context_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            formatted_results = []
            for i, doc_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'id': doc_id,
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching project context: {e}")
            return []
    
    def add_error_pattern(self, error_message: str, solution: str, 
                         language: str, error_type: str, file_path: str = "") -> bool:
        """Add an error pattern and its solution."""
        try:
            content = f"Error: {error_message}\nSolution: {solution}"
            
            metadata = {
                "type": "error_pattern",
                "error_message": error_message,
                "solution": solution,
                "language": language,
                "error_type": error_type,
                "file_path": file_path,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            doc_id = self._generate_doc_id(content, metadata)
            
            self.error_collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            return True
            
        except Exception as e:
            print(f"Error adding error pattern: {e}")
            return False
    
    def search_error_patterns(self, error_query: str, language: Optional[str] = None, 
                            n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar error patterns and solutions."""
        try:
            where_filter = {"type": "error_pattern"}
            if language:
                where_filter["language"] = language
            
            results = self.error_collection.query(
                query_texts=[error_query],
                n_results=n_results,
                where=where_filter
            )
            
            formatted_results = []
            for i, doc_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'id': doc_id,
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching error patterns: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about all collections."""
        try:
            stats = {}
            
            collections = [
                ("code_snippets", self.code_collection),
                ("conversations", self.conversation_collection),
                ("project_context", self.context_collection),
                ("error_patterns", self.error_collection)
            ]
            
            for name, collection in collections:
                try:
                    count = collection.count()
                    stats[name] = {"document_count": count}
                except Exception as e:
                    stats[name] = {"error": str(e)}
            
            return stats
            
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}
    
    def cleanup_old_entries(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old entries from collections."""
        try:
            cutoff_date = datetime.utcnow().replace(day=datetime.utcnow().day - days_to_keep)
            cutoff_iso = cutoff_date.isoformat()
            
            deleted_counts = {}
            
            collections = [
                ("conversations", self.conversation_collection),
                ("project_context", self.context_collection),
                ("error_patterns", self.error_collection)
            ]
            
            for name, collection in collections:
                try:
                    # Get all documents
                    all_docs = collection.get()
                    
                    # Find old documents
                    old_doc_ids = []
                    for i, metadata in enumerate(all_docs['metadatas']):
                        if metadata.get('timestamp', '') < cutoff_iso:
                            old_doc_ids.append(all_docs['ids'][i])
                    
                    # Delete old documents
                    if old_doc_ids:
                        collection.delete(ids=old_doc_ids)
                    
                    deleted_counts[name] = len(old_doc_ids)
                    
                except Exception as e:
                    deleted_counts[name] = f"Error: {e}"
            
            return deleted_counts
            
        except Exception as e:
            print(f"Error cleaning up old entries: {e}")
            return {}
    
    def export_collection(self, collection_name: str, output_path: Path) -> bool:
        """Export a collection to a JSON file."""
        try:
            collection_map = {
                "code_snippets": self.code_collection,
                "conversations": self.conversation_collection,
                "project_context": self.context_collection,
                "error_patterns": self.error_collection
            }
            
            if collection_name not in collection_map:
                print(f"Unknown collection: {collection_name}")
                return False
            
            collection = collection_map[collection_name]
            all_data = collection.get()
            
            export_data = {
                "collection_name": collection_name,
                "export_timestamp": datetime.utcnow().isoformat(),
                "document_count": len(all_data['ids']),
                "documents": []
            }
            
            for i, doc_id in enumerate(all_data['ids']):
                export_data["documents"].append({
                    "id": doc_id,
                    "document": all_data['documents'][i],
                    "metadata": all_data['metadatas'][i]
                })
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting collection: {e}")
            return False
