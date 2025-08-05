"""Code analysis utilities using tree-sitter and AST parsing."""

import re
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import subprocess
import json

try:
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


class CodeAnalyzer:
    """Analyzes code structure, syntax, and patterns."""
    
    def __init__(self):
        self.language_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.sql': 'sql',
            '.sh': 'bash',
            '.bash': 'bash',
            '.zsh': 'zsh',
            '.css': 'css',
            '.html': 'html',
            '.xml': 'xml',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.md': 'markdown',
            '.txt': 'text'
        }
        
        self.project_indicators = {
            'python': ['requirements.txt', 'pyproject.toml', 'setup.py', 'Pipfile'],
            'javascript': ['package.json', 'yarn.lock', 'package-lock.json'],
            'typescript': ['tsconfig.json', 'package.json'],
            'java': ['pom.xml', 'build.gradle', 'build.xml'],
            'go': ['go.mod', 'go.sum'],
            'rust': ['Cargo.toml', 'Cargo.lock'],
            'ruby': ['Gemfile', 'Gemfile.lock'],
            'php': ['composer.json', 'composer.lock'],
        }
    
    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        return self.language_extensions.get(ext)
    
    def detect_project_type(self, directory: Optional[str] = None) -> Dict[str, Any]:
        """Detect project type and framework from directory structure."""
        if not directory:
            directory = Path.cwd()
        else:
            directory = Path(directory)
        
        detected = {
            'languages': [],
            'frameworks': [],
            'build_tools': [],
            'package_managers': [],
            'config_files': []
        }
        
        # Check for project indicator files
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                file_name = file_path.name.lower()
                
                # Language detection
                for lang, indicators in self.project_indicators.items():
                    if file_name in [ind.lower() for ind in indicators]:
                        if lang not in detected['languages']:
                            detected['languages'].append(lang)
                
                # Framework detection
                detected.update(self._detect_frameworks(file_path))
                
                # Config files
                if any(file_name.endswith(ext) for ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']):
                    detected['config_files'].append(str(file_path.relative_to(directory)))
        
        return detected
    
    def _detect_frameworks(self, file_path: Path) -> Dict[str, List[str]]:
        """Detect frameworks from specific files."""
        frameworks = []
        build_tools = []
        package_managers = []
        
        file_name = file_path.name.lower()
        
        # JavaScript/TypeScript frameworks
        if file_name == 'package.json':
            try:
                with open(file_path, 'r') as f:
                    package_data = json.load(f)
                
                deps = {**package_data.get('dependencies', {}), **package_data.get('devDependencies', {})}
                
                # React
                if 'react' in deps:
                    frameworks.append('react')
                if 'next' in deps or 'next.js' in deps:
                    frameworks.append('nextjs')
                if '@angular/core' in deps:
                    frameworks.append('angular')
                if 'vue' in deps:
                    frameworks.append('vue')
                if 'express' in deps:
                    frameworks.append('express')
                if 'fastify' in deps:
                    frameworks.append('fastify')
                
                package_managers.append('npm')
                
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Python frameworks
        elif file_name in ['requirements.txt', 'pyproject.toml']:
            try:
                content = file_path.read_text().lower()
                
                if 'django' in content:
                    frameworks.append('django')
                if 'flask' in content:
                    frameworks.append('flask')
                if 'fastapi' in content:
                    frameworks.append('fastapi')
                if 'streamlit' in content:
                    frameworks.append('streamlit')
                if 'pytest' in content:
                    frameworks.append('pytest')
                
                package_managers.append('pip')
                
            except FileNotFoundError:
                pass
        
        # Build tools
        if file_name in ['makefile', 'dockerfile', 'docker-compose.yml']:
            build_tools.append(file_name.replace('.yml', '').replace('.yaml', ''))
        
        return {
            'frameworks': frameworks,
            'build_tools': build_tools,
            'package_managers': package_managers
        }
    
    def analyze_python_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze Python file structure using AST."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            analysis = {
                'classes': [],
                'functions': [],
                'imports': [],
                'variables': [],
                'complexity': 0,
                'line_count': len(content.split('\n')),
                'docstrings': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis['classes'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'docstring': ast.get_docstring(node)
                    })
                
                elif isinstance(node, ast.FunctionDef):
                    analysis['functions'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node),
                        'is_async': isinstance(node, ast.AsyncFunctionDef)
                    })
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis['imports'].append({
                                'module': alias.name,
                                'alias': alias.asname,
                                'line': node.lineno
                            })
                    else:  # ImportFrom
                        for alias in node.names:
                            analysis['imports'].append({
                                'module': node.module,
                                'name': alias.name,
                                'alias': alias.asname,
                                'line': node.lineno
                            })
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            analysis['variables'].append({
                                'name': target.id,
                                'line': node.lineno
                            })
            
            # Calculate cyclomatic complexity
            analysis['complexity'] = self._calculate_complexity(tree)
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of Python code."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Add complexity for control structures
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1
        
        return complexity
    
    def analyze_javascript_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript file using regex patterns."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'functions': [],
                'classes': [],
                'imports': [],
                'exports': [],
                'variables': [],
                'line_count': len(content.split('\n')),
                'is_typescript': file_path.endswith(('.ts', '.tsx'))
            }
            
            # Function declarations
            func_pattern = r'(?:function\s+(\w+)|(\w+)\s*[:=]\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))'
            for match in re.finditer(func_pattern, content):
                func_name = match.group(1) or match.group(2)
                if func_name:
                    line_no = content[:match.start()].count('\n') + 1
                    analysis['functions'].append({
                        'name': func_name,
                        'line': line_no
                    })
            
            # Class declarations
            class_pattern = r'class\s+(\w+)'
            for match in re.finditer(class_pattern, content):
                line_no = content[:match.start()].count('\n') + 1
                analysis['classes'].append({
                    'name': match.group(1),
                    'line': line_no
                })
            
            # Import statements
            import_pattern = r'import\s+(?:{[^}]+}|\w+|\*\s+as\s+\w+)\s+from\s+[\'"]([^\'"]+)[\'"]'
            for match in re.finditer(import_pattern, content):
                line_no = content[:match.start()].count('\n') + 1
                analysis['imports'].append({
                    'module': match.group(1),
                    'line': line_no
                })
            
            # Export statements
            export_pattern = r'export\s+(?:default\s+)?(?:class|function|const|let|var)?\s*(\w+)?'
            for match in re.finditer(export_pattern, content):
                if match.group(1):
                    line_no = content[:match.start()].count('\n') + 1
                    analysis['exports'].append({
                        'name': match.group(1),
                        'line': line_no
                    })
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def validate_code(self, code: str, language: str, existing_code: str = "") -> Dict[str, Any]:
        """Validate code for syntax errors and quality issues."""
        validation = {
            'syntax_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'quality_score': 0.8
        }
        
        if language == 'python':
            validation.update(self._validate_python_code(code))
        elif language in ['javascript', 'typescript']:
            validation.update(self._validate_javascript_code(code))
        
        # General code quality checks
        validation.update(self._check_code_quality(code, language))
        
        return validation
    
    def _validate_python_code(self, code: str) -> Dict[str, Any]:
        """Validate Python code syntax."""
        result = {'errors': [], 'warnings': []}
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            result['syntax_valid'] = False
            result['errors'].append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            result['syntax_valid'] = False
            result['errors'].append(f"Parse error: {str(e)}")
        
        # Additional Python-specific checks
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            # Check for common issues
            if line.strip().endswith(';;'):
                result['warnings'].append(f"Line {i}: Unnecessary double semicolon")
            
            if 'print(' in line and not line.strip().startswith('#'):
                result['suggestions'].append(f"Line {i}: Consider using logging instead of print")
        
        return result
    
    def _validate_javascript_code(self, code: str) -> Dict[str, Any]:
        """Validate JavaScript code syntax."""
        result = {'errors': [], 'warnings': []}
        
        # Basic syntax checks using regex
        lines = code.split('\n')
        brace_count = 0
        paren_count = 0
        
        for i, line in enumerate(lines, 1):
            brace_count += line.count('{') - line.count('}')
            paren_count += line.count('(') - line.count(')')
            
            # Check for common issues
            if '==' in line and '===' not in line:
                result['suggestions'].append(f"Line {i}: Consider using strict equality (===)")
            
            if 'var ' in line:
                result['suggestions'].append(f"Line {i}: Consider using const or let instead of var")
        
        if brace_count != 0:
            result['errors'].append("Mismatched braces")
            result['syntax_valid'] = False
        
        if paren_count != 0:
            result['errors'].append("Mismatched parentheses")
            result['syntax_valid'] = False
        
        return result
    
    def _check_code_quality(self, code: str, language: str) -> Dict[str, Any]:
        """Check general code quality metrics."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        quality = {
            'line_count': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'comment_ratio': 0,
            'average_line_length': 0,
            'long_lines': []
        }
        
        comment_lines = 0
        total_length = 0
        
        for i, line in enumerate(lines, 1):
            line_length = len(line)
            total_length += line_length
            
            # Check for comments
            if language == 'python' and line.strip().startswith('#'):
                comment_lines += 1
            elif language in ['javascript', 'typescript'] and line.strip().startswith('//'):
                comment_lines += 1
            
            # Check for very long lines
            if line_length > 100:
                quality['long_lines'].append(i)
        
        if non_empty_lines:
            quality['comment_ratio'] = comment_lines / len(non_empty_lines)
            quality['average_line_length'] = total_length / len(lines)
        
        # Calculate quality score
        score = 0.8  # Base score
        
        # Adjust based on comment ratio
        if quality['comment_ratio'] > 0.1:
            score += 0.1
        elif quality['comment_ratio'] < 0.05:
            score -= 0.1
        
        # Adjust based on line length
        if quality['average_line_length'] > 120:
            score -= 0.1
        
        # Adjust based on long lines
        if len(quality['long_lines']) > len(lines) * 0.1:
            score -= 0.1
        
        quality['quality_score'] = max(0.0, min(1.0, score))
        
        return quality
    
    def analyze_codebase_patterns(self, directory: Path) -> Dict[str, Any]:
        """Analyze patterns across the entire codebase."""
        patterns = {
            'naming_conventions': {},
            'import_patterns': {},
            'common_functions': {},
            'file_organization': {},
            'code_style': {}
        }
        
        try:
            code_files = []
            for ext in ['.py', '.js', '.ts', '.jsx', '.tsx']:
                code_files.extend(directory.rglob(f'*{ext}'))
            
            # Limit analysis to avoid performance issues
            code_files = code_files[:50]
            
            for file_path in code_files:
                if file_path.is_file():
                    lang = self.detect_language(str(file_path))
                    if lang == 'python':
                        file_analysis = self.analyze_python_file(str(file_path))
                    elif lang in ['javascript', 'typescript']:
                        file_analysis = self.analyze_javascript_file(str(file_path))
                    else:
                        continue
                    
                    # Extract patterns
                    self._extract_naming_patterns(file_analysis, patterns['naming_conventions'])
                    self._extract_import_patterns(file_analysis, patterns['import_patterns'])
            
            return patterns
            
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_naming_patterns(self, file_analysis: Dict[str, Any], naming_patterns: Dict[str, Any]):
        """Extract naming convention patterns from file analysis."""
        functions = file_analysis.get('functions', [])
        classes = file_analysis.get('classes', [])
        
        for func in functions:
            name = func['name']
            if '_' in name:
                naming_patterns['snake_case'] = naming_patterns.get('snake_case', 0) + 1
            elif name[0].islower() and any(c.isupper() for c in name[1:]):
                naming_patterns['camelCase'] = naming_patterns.get('camelCase', 0) + 1
        
        for cls in classes:
            name = cls['name']
            if name[0].isupper():
                naming_patterns['PascalCase'] = naming_patterns.get('PascalCase', 0) + 1
    
    def _extract_import_patterns(self, file_analysis: Dict[str, Any], import_patterns: Dict[str, Any]):
        """Extract import patterns from file analysis."""
        imports = file_analysis.get('imports', [])
        
        for imp in imports:
            module = imp.get('module', '')
            if module:
                import_patterns[module] = import_patterns.get(module, 0) + 1
    
    def get_file_metrics(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a single file."""
        language = self.detect_language(file_path)
        
        metrics = {
            'language': language,
            'file_path': file_path,
            'file_size': Path(file_path).stat().st_size if Path(file_path).exists() else 0
        }
        
        if language == 'python':
            metrics.update(self.analyze_python_file(file_path))
        elif language in ['javascript', 'typescript']:
            metrics.update(self.analyze_javascript_file(file_path))
        
        return metrics
