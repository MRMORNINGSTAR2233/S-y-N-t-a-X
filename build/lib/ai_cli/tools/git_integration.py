"""Git integration utilities for version control operations."""

import git
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import subprocess
import os


class GitManager:
    """Manages Git operations and repository information."""
    
    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.repo: Optional[git.Repo] = None
        self._initialize_repo()
    
    def _initialize_repo(self) -> None:
        """Initialize git repository connection."""
        try:
            # Find the git repository
            current_path = self.repo_path
            while current_path != current_path.parent:
                if (current_path / '.git').exists():
                    self.repo = git.Repo(str(current_path))
                    return
                current_path = current_path.parent
            print("Warning: Not in a git repository")
        except git.exc.GitError as e:
            print(f"Warning: Git error: {e}")
        except Exception as e:
            print(f"Warning: Failed to initialize git repository: {e}")
    
    def is_git_repo(self) -> bool:
        """Check if current directory is a git repository."""
        return self.repo is not None
    
    def get_current_branch(self) -> Optional[str]:
        """Get the current git branch name."""
        if not self.repo:
            return None
        
        try:
            return self.repo.active_branch.name
        except Exception:
            return None
    
    def get_repo_info(self) -> Dict[str, Any]:
        """Get comprehensive repository information."""
        if not self.repo:
            return {"is_git_repo": False}
        
        try:
            # Get remote info
            remotes = []
            for remote in self.repo.remotes:
                remotes.append({
                    "name": remote.name,
                    "url": list(remote.urls)[0] if remote.urls else None
                })
            
            # Get recent commits
            recent_commits = []
            for commit in list(self.repo.iter_commits(max_count=5)):
                recent_commits.append({
                    "hash": commit.hexsha[:8],
                    "message": commit.message.strip(),
                    "author": str(commit.author),
                    "date": commit.committed_datetime.isoformat()
                })
            
            return {
                "is_git_repo": True,
                "current_branch": self.get_current_branch(),
                "remotes": remotes,
                "recent_commits": recent_commits,
                "repo_root": str(self.repo.working_dir),
                "is_dirty": self.repo.is_dirty(),
                "untracked_files": self.repo.untracked_files[:10],  # Limit to 10
            }
            
        except Exception as e:
            return {"is_git_repo": True, "error": str(e)}
    
    def get_file_status(self, file_path: str) -> Dict[str, Any]:
        """Get git status for a specific file."""
        if not self.repo:
            return {"status": "not_in_repo"}
        
        try:
            # Check if file is in repo
            rel_path = os.path.relpath(file_path, self.repo.working_dir)
            
            # Get file status
            status = {
                "is_tracked": False,
                "is_modified": False,
                "is_staged": False,
                "is_untracked": False
            }
            
            # Check if file exists in index
            try:
                self.repo.index.entries[rel_path]
                status["is_tracked"] = True
            except KeyError:
                status["is_untracked"] = True
            
            # Check if file is modified
            if rel_path in [item.a_path for item in self.repo.index.diff(None)]:
                status["is_modified"] = True
            
            # Check if file is staged
            if rel_path in [item.a_path for item in self.repo.index.diff("HEAD")]:
                status["is_staged"] = True
            
            return status
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_changed_files(self, include_untracked: bool = True) -> List[Dict[str, str]]:
        """Get list of changed files in the repository."""
        if not self.repo:
            return []
        
        changed_files = []
        
        try:
            # Modified files (unstaged)
            for item in self.repo.index.diff(None):
                changed_files.append({
                    "path": item.a_path,
                    "status": "modified",
                    "staged": False
                })
            
            # Staged files
            for item in self.repo.index.diff("HEAD"):
                changed_files.append({
                    "path": item.a_path,
                    "status": "staged",
                    "staged": True
                })
            
            # Untracked files
            if include_untracked:
                for file_path in self.repo.untracked_files:
                    changed_files.append({
                        "path": file_path,
                        "status": "untracked",
                        "staged": False
                    })
            
            return changed_files
            
        except Exception as e:
            print(f"Error getting changed files: {e}")
            return []
    
    def get_file_diff(self, file_path: str) -> Optional[str]:
        """Get git diff for a specific file."""
        if not self.repo:
            return None
        
        try:
            rel_path = os.path.relpath(file_path, self.repo.working_dir)
            
            # Get diff for the file
            diff = self.repo.git.diff(rel_path)
            return diff if diff else None
            
        except Exception as e:
            print(f"Error getting diff for {file_path}: {e}")
            return None
    
    def stage_file(self, file_path: str) -> bool:
        """Stage a file for commit."""
        if not self.repo:
            return False
        
        try:
            rel_path = os.path.relpath(file_path, self.repo.working_dir)
            self.repo.index.add([rel_path])
            return True
        except Exception as e:
            print(f"Error staging {file_path}: {e}")
            return False
    
    def commit_changes(self, message: str, files: Optional[List[str]] = None) -> bool:
        """Commit changes with a message."""
        if not self.repo:
            return False
        
        try:
            if files:
                # Stage specific files
                rel_paths = [os.path.relpath(f, self.repo.working_dir) for f in files]
                self.repo.index.add(rel_paths)
            
            # Commit
            self.repo.index.commit(message)
            return True
            
        except Exception as e:
            print(f"Error committing changes: {e}")
            return False
    
    def create_branch(self, branch_name: str, checkout: bool = True) -> bool:
        """Create a new git branch."""
        if not self.repo:
            return False
        
        try:
            new_branch = self.repo.create_head(branch_name)
            if checkout:
                new_branch.checkout()
            return True
        except Exception as e:
            print(f"Error creating branch {branch_name}: {e}")
            return False
    
    def checkout_branch(self, branch_name: str) -> bool:
        """Checkout an existing branch."""
        if not self.repo:
            return False
        
        try:
            self.repo.git.checkout(branch_name)
            return True
        except Exception as e:
            print(f"Error checking out branch {branch_name}: {e}")
            return False
    
    def get_branches(self) -> Dict[str, List[str]]:
        """Get list of local and remote branches."""
        if not self.repo:
            return {"local": [], "remote": []}
        
        try:
            local_branches = [str(branch) for branch in self.repo.branches]
            remote_branches = [str(branch) for branch in self.repo.remote().refs]
            
            return {
                "local": local_branches,
                "remote": remote_branches
            }
        except Exception as e:
            print(f"Error getting branches: {e}")
            return {"local": [], "remote": []}
    
    def get_commit_history(self, file_path: Optional[str] = None, max_count: int = 10) -> List[Dict[str, Any]]:
        """Get commit history for repository or specific file."""
        if not self.repo:
            return []
        
        try:
            commits = []
            
            if file_path:
                rel_path = os.path.relpath(file_path, self.repo.working_dir)
                commit_iter = self.repo.iter_commits(paths=rel_path, max_count=max_count)
            else:
                commit_iter = self.repo.iter_commits(max_count=max_count)
            
            for commit in commit_iter:
                commits.append({
                    "hash": commit.hexsha,
                    "short_hash": commit.hexsha[:8],
                    "message": commit.message.strip(),
                    "author": str(commit.author),
                    "email": commit.author.email,
                    "date": commit.committed_datetime.isoformat(),
                    "files_changed": len(list(commit.stats.files.keys()))
                })
            
            return commits
            
        except Exception as e:
            print(f"Error getting commit history: {e}")
            return []
    
    def get_file_blame(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """Get git blame information for a file."""
        if not self.repo:
            return None
        
        try:
            rel_path = os.path.relpath(file_path, self.repo.working_dir)
            blame_data = []
            
            for commit, lines in self.repo.blame("HEAD", rel_path):
                for line_no, line in enumerate(lines, 1):
                    blame_data.append({
                        "line_number": line_no,
                        "line_content": line,
                        "commit_hash": commit.hexsha[:8],
                        "author": str(commit.author),
                        "date": commit.committed_datetime.isoformat()
                    })
            
            return blame_data
            
        except Exception as e:
            print(f"Error getting blame for {file_path}: {e}")
            return None
    
    def is_file_ignored(self, file_path: str) -> bool:
        """Check if a file is ignored by git."""
        if not self.repo:
            return False
        
        try:
            # Use git check-ignore command
            result = subprocess.run(
                ["git", "check-ignore", file_path],
                cwd=self.repo.working_dir,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_repo_stats(self) -> Dict[str, Any]:
        """Get repository statistics."""
        if not self.repo:
            return {}
        
        try:
            stats = {
                "total_commits": 0,
                "contributors": set(),
                "file_count": 0,
                "languages": {}
            }
            
            # Count commits and contributors
            for commit in self.repo.iter_commits():
                stats["total_commits"] += 1
                stats["contributors"].add(str(commit.author))
                
                # Limit to avoid performance issues
                if stats["total_commits"] >= 1000:
                    break
            
            stats["contributors"] = list(stats["contributors"])
            stats["contributor_count"] = len(stats["contributors"])
            
            # Count files by extension
            for root, dirs, files in os.walk(self.repo.working_dir):
                # Skip .git directory
                if '.git' in root:
                    continue
                
                for file in files:
                    stats["file_count"] += 1
                    ext = Path(file).suffix.lower()
                    if ext:
                        stats["languages"][ext] = stats["languages"].get(ext, 0) + 1
            
            return stats
            
        except Exception as e:
            print(f"Error getting repo stats: {e}")
            return {}
