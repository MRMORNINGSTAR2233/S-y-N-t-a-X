"""Navigator agent for finding and searching code across the codebase."""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re

from ai_cli.llms.manager import LLMManager, ChatMessage
from ai_cli.memory.vector_store import VectorStore
from ai_cli.tools.git_integration import GitManager
from ai_cli.tools.file_operations import FileOperations
from ai_cli.tools.code_analysis import CodeAnalyzer


@dataclass
class NavigationResult:
    """Result of navigation or search operation."""
    found: bool
    location: Optional[str] = None
    line_number: Optional[int] = None
    context: Optional[str] = None
    confidence: float = 0.0
    additional_matches: List[Dict[str, Any]] = None


class NavigatorAgent:
    """Agent for navigating and searching code across the codebase."""
    
    def __init__(self, llm_manager: LLMManager, vector_store: VectorStore, git_manager: GitManager):
        self.llm_manager = llm_manager
        self.vector_store = vector_store
        self.git_manager = git_manager
        self.file_ops = FileOperations()
        self.code_analyzer = CodeAnalyzer()
    
    async def navigate_to_symbol(self, symbol: str, symbol_type: Optional[str] = None,
                                target_file: Optional[str] = None) -> Dict[str, Any]:
        """Navigate to a specific symbol (function, class, variable) in the codebase."""
        
        search_results = []
        
        # If target file is specified, search there first
        if target_file and Path(target_file).exists():
            file_results = await self._search_symbol_in_file(symbol, target_file, symbol_type)
            if file_results.found:
                return {
                    "found": True,
                    "location": f"{target_file}:{file_results.line_number}",
                    "context": file_results.context,
                    "confidence": file_results.confidence
                }
            search_results.append(file_results)
        
        # Search across the entire codebase
        codebase_results = await self._search_symbol_in_codebase(symbol, symbol_type)
        
        if codebase_results:
            best_match = max(codebase_results, key=lambda x: x.confidence)
            if best_match.confidence > 0.7:
                return {
                    "found": True,
                    "location": f"{best_match.location}:{best_match.line_number}",
                    "context": best_match.context,
                    "confidence": best_match.confidence,
                    "additional_matches": [
                        {
                            "location": f"{r.location}:{r.line_number}",
                            "confidence": r.confidence,
                            "context": r.context[:100] + "..." if r.context and len(r.context) > 100 else r.context
                        }
                        for r in codebase_results[1:5]  # Show up to 4 additional matches
                    ]
                }
        
        # Use vector search as fallback
        vector_results = self.vector_store.search_code(
            query=f"{symbol_type or ''} {symbol}",
            n_results=5
        )
        
        if vector_results:
            for result in vector_results:
                if symbol.lower() in result['code'].lower():
                    return {
                        "found": True,
                        "location": result['metadata'].get('file_path', 'Unknown'),
                        "context": result['code'][:200] + "...",
                        "confidence": result['similarity'],
                        "source": "vector_search"
                    }
        
        return {
            "found": False,
            "error": f"Symbol '{symbol}' not found in codebase",
            "searched_locations": len(search_results)
        }
    
    async def _search_symbol_in_file(self, symbol: str, file_path: str, 
                                   symbol_type: Optional[str] = None) -> NavigationResult:
        """Search for a symbol in a specific file."""
        
        try:
            content = self.file_ops.read_file(file_path)
            if not content:
                return NavigationResult(found=False)
            
            language = self.code_analyzer.detect_language(file_path)
            
            # Use language-specific analysis if available
            if language == 'python':
                return await self._search_python_symbol(symbol, content, symbol_type)
            elif language in ['javascript', 'typescript']:
                return await self._search_javascript_symbol(symbol, content, symbol_type)
            else:
                return await self._search_generic_symbol(symbol, content, symbol_type)
                
        except Exception as e:
            print(f"Error searching in {file_path}: {e}")
            return NavigationResult(found=False)
    
    async def _search_python_symbol(self, symbol: str, content: str, 
                                  symbol_type: Optional[str] = None) -> NavigationResult:
        """Search for Python symbol using AST analysis."""
        
        try:
            import ast
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == symbol:
                    if not symbol_type or symbol_type.lower() in ['class', 'cls']:
                        context = self._extract_context(content, node.lineno)
                        return NavigationResult(
                            found=True,
                            line_number=node.lineno,
                            context=context,
                            confidence=1.0
                        )
                
                elif isinstance(node, ast.FunctionDef) and node.name == symbol:
                    if not symbol_type or symbol_type.lower() in ['function', 'func', 'def', 'method']:
                        context = self._extract_context(content, node.lineno)
                        return NavigationResult(
                            found=True,
                            line_number=node.lineno,
                            context=context,
                            confidence=1.0
                        )
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == symbol:
                            if not symbol_type or symbol_type.lower() in ['variable', 'var']:
                                context = self._extract_context(content, node.lineno)
                                return NavigationResult(
                                    found=True,
                                    line_number=node.lineno,
                                    context=context,
                                    confidence=0.9
                                )
            
            return NavigationResult(found=False)
            
        except Exception as e:
            print(f"Error in Python AST analysis: {e}")
            return await self._search_generic_symbol(symbol, content, symbol_type)
    
    async def _search_javascript_symbol(self, symbol: str, content: str, 
                                      symbol_type: Optional[str] = None) -> NavigationResult:
        """Search for JavaScript/TypeScript symbol using regex patterns."""
        
        lines = content.split('\n')
        
        # Patterns for different symbol types
        patterns = {
            'function': [
                rf'function\s+{re.escape(symbol)}\s*\(',
                rf'{re.escape(symbol)}\s*[:=]\s*(?:async\s+)?function',
                rf'{re.escape(symbol)}\s*[:=]\s*\([^)]*\)\s*=>'
            ],
            'class': [
                rf'class\s+{re.escape(symbol)}\s*(?:extends|{{)',
            ],
            'variable': [
                rf'(?:const|let|var)\s+{re.escape(symbol)}\s*[=:]',
                rf'{re.escape(symbol)}\s*[:=]\s*'
            ]
        }
        
        # If symbol_type is specified, only check those patterns
        if symbol_type and symbol_type.lower() in patterns:
            search_patterns = patterns[symbol_type.lower()]
        else:
            # Search all patterns
            search_patterns = []
            for pattern_list in patterns.values():
                search_patterns.extend(pattern_list)
        
        for line_no, line in enumerate(lines, 1):
            for pattern in search_patterns:
                if re.search(pattern, line):
                    context = self._extract_context(content, line_no)
                    return NavigationResult(
                        found=True,
                        line_number=line_no,
                        context=context,
                        confidence=0.9
                    )
        
        return NavigationResult(found=False)
    
    async def _search_generic_symbol(self, symbol: str, content: str, 
                                   symbol_type: Optional[str] = None) -> NavigationResult:
        """Generic symbol search using simple pattern matching."""
        
        lines = content.split('\n')
        
        # Look for exact symbol matches
        for line_no, line in enumerate(lines, 1):
            if symbol in line:
                # Check if it's a likely definition (not just a reference)
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in ['def ', 'class ', 'function ', 'var ', 'let ', 'const ']):
                    context = self._extract_context(content, line_no)
                    return NavigationResult(
                        found=True,
                        line_number=line_no,
                        context=context,
                        confidence=0.7
                    )
        
        # If no definition found, return first occurrence
        for line_no, line in enumerate(lines, 1):
            if symbol in line:
                context = self._extract_context(content, line_no)
                return NavigationResult(
                    found=True,
                    line_number=line_no,
                    context=context,
                    confidence=0.5
                )
        
        return NavigationResult(found=False)
    
    def _extract_context(self, content: str, line_number: int, context_lines: int = 3) -> str:
        """Extract context around a specific line."""
        
        lines = content.split('\n')
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        
        context_text = []
        for i in range(start, end):
            marker = ">>> " if i == line_number - 1 else "    "
            context_text.append(f"{marker}{i + 1:4d}: {lines[i]}")
        
        return '\n'.join(context_text)
    
    async def _search_symbol_in_codebase(self, symbol: str, 
                                       symbol_type: Optional[str] = None) -> List[NavigationResult]:
        """Search for symbol across the entire codebase."""
        
        results = []
        
        # Get all code files
        code_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.go', '.rs']
        code_files = []
        
        for ext in code_extensions:
            files = self.file_ops.find_files(
                root_dir=Path.cwd(),
                patterns=[f'*{ext}'],
                max_files=100  # Limit to avoid performance issues
            )
            code_files.extend([f['path'] for f in files])
        
        # Search in each file
        for file_path in code_files[:50]:  # Limit concurrent searches
            try:
                result = await self._search_symbol_in_file(symbol, file_path, symbol_type)
                if result.found:
                    result.location = file_path
                    results.append(result)
            except Exception as e:
                continue  # Skip files that can't be processed
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)
    
    async def search_codebase(self, query: str, search_type: str = "semantic",
                            include_patterns: Optional[List[str]] = None,
                            exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Search the codebase using different strategies."""
        
        results = {
            "query": query,
            "search_type": search_type,
            "results": [],
            "total_files_searched": 0,
            "search_time_ms": 0
        }
        
        import time
        start_time = time.time()
        
        if search_type == "semantic":
            results.update(await self._semantic_search(query, include_patterns, exclude_patterns))
        elif search_type == "keyword":
            results.update(await self._keyword_search(query, include_patterns, exclude_patterns))
        else:
            # Hybrid search - combine both approaches
            semantic_results = await self._semantic_search(query, include_patterns, exclude_patterns)
            keyword_results = await self._keyword_search(query, include_patterns, exclude_patterns)
            
            # Merge and deduplicate results
            all_results = semantic_results.get("results", []) + keyword_results.get("results", [])
            unique_results = self._deduplicate_search_results(all_results)
            results["results"] = unique_results
            results["total_files_searched"] = max(
                semantic_results.get("total_files_searched", 0),
                keyword_results.get("total_files_searched", 0)
            )
        
        results["search_time_ms"] = int((time.time() - start_time) * 1000)
        
        return results
    
    async def _semantic_search(self, query: str, include_patterns: Optional[List[str]] = None,
                             exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform semantic search using vector store."""
        
        # Search code snippets
        code_results = self.vector_store.search_code(query=query, n_results=10)
        
        # Search project context
        context_results = self.vector_store.search_project_context(query=query, n_results=5)
        
        # Search conversations for relevant context
        conversation_results = self.vector_store.search_conversations(query=query, n_results=3)
        
        formatted_results = []
        
        # Format code results
        for result in code_results:
            formatted_results.append({
                "type": "code",
                "file_path": result['metadata'].get('file_path', 'Unknown'),
                "content": result['code'],
                "similarity": result['similarity'],
                "language": result['metadata'].get('language', 'unknown'),
                "function_name": result['metadata'].get('function_name', ''),
                "class_name": result['metadata'].get('class_name', '')
            })
        
        # Format context results
        for result in context_results:
            formatted_results.append({
                "type": "context",
                "content": result['content'],
                "similarity": result['similarity'],
                "context_type": result['metadata'].get('context_type', 'unknown'),
                "file_paths": result['metadata'].get('file_paths', [])
            })
        
        return {
            "results": sorted(formatted_results, key=lambda x: x['similarity'], reverse=True),
            "total_files_searched": len(set(r.get('file_path', '') for r in formatted_results))
        }
    
    async def _keyword_search(self, query: str, include_patterns: Optional[List[str]] = None,
                            exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform keyword search using file system search."""
        
        # Get files to search
        search_patterns = include_patterns or ['*.py', '*.js', '*.ts', '*.jsx', '*.tsx', '*.java', '*.cpp', '*.c']
        exclude_patterns = exclude_patterns or ['.git/*', 'node_modules/*', '__pycache__/*']
        
        files_to_search = []
        for pattern in search_patterns:
            files = self.file_ops.find_files(
                root_dir=Path.cwd(),
                patterns=[pattern],
                exclude_patterns=exclude_patterns,
                max_files=200
            )
            files_to_search.extend(files)
        
        # Remove duplicates
        unique_files = {f['path']: f for f in files_to_search}.values()
        
        results = []
        query_words = query.lower().split()
        
        for file_info in unique_files:
            try:
                content = self.file_ops.read_file(file_info['path'])
                if not content:
                    continue
                
                content_lower = content.lower()
                
                # Check if all query words are present
                if not all(word in content_lower for word in query_words):
                    continue
                
                # Find matching lines
                lines = content.split('\n')
                matching_lines = []
                
                for line_no, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    if any(word in line_lower for word in query_words):
                        matching_lines.append({
                            "line_number": line_no,
                            "content": line.strip(),
                            "relevance": sum(1 for word in query_words if word in line_lower)
                        })
                
                if matching_lines:
                    # Sort by relevance
                    matching_lines.sort(key=lambda x: x['relevance'], reverse=True)
                    
                    results.append({
                        "type": "keyword_match",
                        "file_path": file_info['path'],
                        "matches": matching_lines[:5],  # Top 5 matches per file
                        "total_matches": len(matching_lines),
                        "file_size": file_info['size'],
                        "language": self.code_analyzer.detect_language(file_info['path'])
                    })
                    
            except Exception as e:
                continue  # Skip files that can't be processed
        
        # Sort results by total matches
        results.sort(key=lambda x: x['total_matches'], reverse=True)
        
        return {
            "results": results,
            "total_files_searched": len(list(unique_files))
        }
    
    def _deduplicate_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate search results."""
        
        seen_files = set()
        unique_results = []
        
        for result in results:
            file_path = result.get('file_path', '')
            if file_path and file_path not in seen_files:
                seen_files.add(file_path)
                unique_results.append(result)
            elif not file_path:  # Context results without file paths
                unique_results.append(result)
        
        return unique_results
