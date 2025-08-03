"""File operations utilities for reading, writing, and manipulating files."""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import fnmatch
import tempfile
import json
import yaml
import toml


class FileOperations:
    """Handles file system operations for the AI CLI."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "syntax_ai_cli"
        self.temp_dir.mkdir(exist_ok=True)
    
    def read_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """Read file content safely."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                return None
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    
    def write_file(self, file_path: Union[str, Path], content: str, 
                   backup: bool = True) -> bool:
        """Write content to file with optional backup."""
        try:
            file_path = Path(file_path)
            
            # Create backup if file exists and backup is requested
            if backup and file_path.exists():
                backup_path = file_path.with_suffix(file_path.suffix + '.backup')
                shutil.copy2(file_path, backup_path)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            print(f"Error writing file {file_path}: {e}")
            return False
    
    def append_to_file(self, file_path: Union[str, Path], content: str) -> bool:
        """Append content to a file."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            print(f"Error appending to file {file_path}: {e}")
            return False
    
    def copy_file(self, source: Union[str, Path], destination: Union[str, Path]) -> bool:
        """Copy a file to destination."""
        try:
            source_path = Path(source)
            dest_path = Path(destination)
            
            # Ensure destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source_path, dest_path)
            return True
            
        except Exception as e:
            print(f"Error copying file {source} to {destination}: {e}")
            return False
    
    def move_file(self, source: Union[str, Path], destination: Union[str, Path]) -> bool:
        """Move a file to destination."""
        try:
            source_path = Path(source)
            dest_path = Path(destination)
            
            # Ensure destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(source_path), str(dest_path))
            return True
            
        except Exception as e:
            print(f"Error moving file {source} to {destination}: {e}")
            return False
    
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """Delete a file."""
        try:
            Path(file_path).unlink()
            return True
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            return False
    
    def create_directory(self, dir_path: Union[str, Path]) -> bool:
        """Create a directory and its parents."""
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory {dir_path}: {e}")
            return False
    
    def list_files(self, directory: Union[str, Path], 
                   pattern: str = "*", recursive: bool = False) -> List[Path]:
        """List files in directory matching pattern."""
        try:
            dir_path = Path(directory)
            
            if recursive:
                # Use rglob for recursive search
                return list(dir_path.rglob(pattern))
            else:
                # Use glob for non-recursive search
                return list(dir_path.glob(pattern))
                
        except Exception as e:
            print(f"Error listing files in {directory}: {e}")
            return []
    
    def find_files(self, root_dir: Union[str, Path], 
                   patterns: List[str], 
                   exclude_patterns: List[str] = None,
                   max_files: int = 1000) -> List[Dict[str, Any]]:
        """Find files matching patterns with exclusions."""
        exclude_patterns = exclude_patterns or [
            ".git/*", "node_modules/*", "__pycache__/*", "*.pyc",
            ".venv/*", "venv/*", "*.log", ".DS_Store"
        ]
        
        found_files = []
        root_path = Path(root_dir)
        
        try:
            for pattern in patterns:
                if len(found_files) >= max_files:
                    break
                
                for file_path in root_path.rglob(pattern):
                    if len(found_files) >= max_files:
                        break
                    
                    # Check exclusions
                    rel_path = file_path.relative_to(root_path)
                    excluded = False
                    
                    for exclude_pattern in exclude_patterns:
                        if fnmatch.fnmatch(str(rel_path), exclude_pattern):
                            excluded = True
                            break
                    
                    if not excluded and file_path.is_file():
                        file_info = {
                            "path": str(file_path),
                            "relative_path": str(rel_path),
                            "size": file_path.stat().st_size,
                            "modified": file_path.stat().st_mtime,
                            "extension": file_path.suffix.lower()
                        }
                        found_files.append(file_info)
            
            return found_files
            
        except Exception as e:
            print(f"Error finding files: {e}")
            return []
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get detailed file information."""
        try:
            path = Path(file_path)
            stat = path.stat()
            
            return {
                "path": str(path.absolute()),
                "name": path.name,
                "extension": path.suffix.lower(),
                "size": stat.st_size,
                "size_human": self._format_size(stat.st_size),
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "is_file": path.is_file(),
                "is_directory": path.is_dir(),
                "permissions": oct(stat.st_mode)[-3:],
                "line_count": self._count_lines(path) if path.is_file() else 0
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def read_json(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Read and parse JSON file."""
        try:
            content = self.read_file(file_path)
            if content:
                return json.loads(content)
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in {file_path}: {e}")
            return None
        except Exception as e:
            print(f"Error reading JSON file {file_path}: {e}")
            return None
    
    def write_json(self, file_path: Union[str, Path], data: Dict[str, Any], 
                   indent: int = 2) -> bool:
        """Write data to JSON file."""
        try:
            content = json.dumps(data, indent=indent, ensure_ascii=False)
            return self.write_file(file_path, content)
        except Exception as e:
            print(f"Error writing JSON file {file_path}: {e}")
            return False
    
    def read_yaml(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Read and parse YAML file."""
        try:
            content = self.read_file(file_path)
            if content:
                return yaml.safe_load(content)
            return None
        except yaml.YAMLError as e:
            print(f"Error parsing YAML in {file_path}: {e}")
            return None
        except Exception as e:
            print(f"Error reading YAML file {file_path}: {e}")
            return None
    
    def write_yaml(self, file_path: Union[str, Path], data: Dict[str, Any]) -> bool:
        """Write data to YAML file."""
        try:
            content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
            return self.write_file(file_path, content)
        except Exception as e:
            print(f"Error writing YAML file {file_path}: {e}")
            return False
    
    def read_toml(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Read and parse TOML file."""
        try:
            content = self.read_file(file_path)
            if content:
                return toml.loads(content)
            return None
        except toml.TomlDecodeError as e:
            print(f"Error parsing TOML in {file_path}: {e}")
            return None
        except Exception as e:
            print(f"Error reading TOML file {file_path}: {e}")
            return None
    
    def write_toml(self, file_path: Union[str, Path], data: Dict[str, Any]) -> bool:
        """Write data to TOML file."""
        try:
            content = toml.dumps(data)
            return self.write_file(file_path, content)
        except Exception as e:
            print(f"Error writing TOML file {file_path}: {e}")
            return False
    
    def create_temp_file(self, content: str, suffix: str = ".tmp") -> Optional[Path]:
        """Create a temporary file with content."""
        try:
            import tempfile
            fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=self.temp_dir)
            
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return Path(temp_path)
            
        except Exception as e:
            print(f"Error creating temp file: {e}")
            return None
    
    def cleanup_temp_files(self) -> bool:
        """Clean up temporary files."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(exist_ok=True)
            return True
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")
            return False
    
    def is_text_file(self, file_path: Union[str, Path]) -> bool:
        """Check if file is a text file."""
        try:
            path = Path(file_path)
            
            # Check by extension
            text_extensions = {
                '.txt', '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',
                '.css', '.html', '.xml', '.json', '.yaml', '.yml', '.toml',
                '.md', '.rst', '.ini', '.cfg', '.conf', '.sh', '.bash',
                '.sql', '.r', '.rb', '.php', '.go', '.rs', '.swift', '.kt'
            }
            
            if path.suffix.lower() in text_extensions:
                return True
            
            # Check by reading a small portion
            try:
                with open(path, 'rb') as f:
                    chunk = f.read(1024)
                    return b'\x00' not in chunk  # Binary files often contain null bytes
            except:
                return False
                
        except Exception:
            return False
    
    def get_directory_structure(self, directory: Union[str, Path], 
                              max_depth: int = 3) -> Dict[str, Any]:
        """Get directory structure as a tree."""
        def _build_tree(path: Path, current_depth: int) -> Dict[str, Any]:
            if current_depth > max_depth:
                return {"name": path.name, "type": "directory", "truncated": True}
            
            if path.is_file():
                return {
                    "name": path.name,
                    "type": "file",
                    "size": path.stat().st_size,
                    "extension": path.suffix.lower()
                }
            
            children = []
            try:
                for child in sorted(path.iterdir()):
                    # Skip hidden files and common ignored directories
                    if child.name.startswith('.') or child.name in ['node_modules', '__pycache__']:
                        continue
                    
                    children.append(_build_tree(child, current_depth + 1))
                    
                    # Limit number of children to avoid huge trees
                    if len(children) >= 50:
                        children.append({"name": "...", "type": "truncated"})
                        break
                        
            except PermissionError:
                children = [{"name": "Permission denied", "type": "error"}]
            
            return {
                "name": path.name,
                "type": "directory",
                "children": children
            }
        
        try:
            return _build_tree(Path(directory), 0)
        except Exception as e:
            return {"error": str(e)}
