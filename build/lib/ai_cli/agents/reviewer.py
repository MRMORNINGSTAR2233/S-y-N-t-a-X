"""Code reviewer agent for analyzing code quality and providing feedback."""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from ai_cli.llms.manager import LLMManager, ChatMessage
from ai_cli.memory.vector_store import VectorStore
from ai_cli.tools.git_integration import GitManager
from ai_cli.tools.file_operations import FileOperations
from ai_cli.tools.code_analysis import CodeAnalyzer


@dataclass
class ReviewIssue:
    """Represents a code review issue."""
    type: str  # 'security', 'performance', 'style', 'logic', 'maintainability'
    severity: str  # 'critical', 'major', 'minor', 'info'
    file_path: str
    line_number: Optional[int]
    title: str
    description: str
    suggestion: str
    code_snippet: Optional[str] = None


class ReviewerAgent:
    """Agent for comprehensive code review and analysis."""
    
    def __init__(self, llm_manager: LLMManager, vector_store: VectorStore, git_manager: GitManager):
        self.llm_manager = llm_manager
        self.vector_store = vector_store
        self.git_manager = git_manager
        self.file_ops = FileOperations()
        self.code_analyzer = CodeAnalyzer()
        
        self.review_categories = {
            'security': 'Security vulnerabilities and risks',
            'performance': 'Performance issues and optimizations', 
            'style': 'Code style and formatting',
            'logic': 'Logic errors and bugs',
            'maintainability': 'Code maintainability and readability',
            'testing': 'Testing coverage and quality',
            'documentation': 'Documentation and comments'
        }
    
    async def review_code(self, file_patterns: Optional[List[str]] = None,
                         review_scope: str = 'all',
                         output_file: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive code review."""
        
        review_start = datetime.now()
        
        # Get files to review
        files_to_review = await self._get_files_for_review(file_patterns)
        
        if not files_to_review:
            return {
                "success": False,
                "error": "No files found for review",
                "files_analyzed": 0
            }
        
        # Perform review
        all_issues = []
        review_summary = {
            "files_analyzed": 0,
            "total_issues": 0,
            "issues_by_severity": {"critical": 0, "major": 0, "minor": 0, "info": 0},
            "issues_by_type": {category: 0 for category in self.review_categories.keys()},
            "quality_score": 0.0
        }
        
        for file_path in files_to_review:
            try:
                file_issues = await self._review_single_file(file_path, review_scope)
                all_issues.extend(file_issues)
                review_summary["files_analyzed"] += 1
                
                # Update counters
                for issue in file_issues:
                    review_summary["issues_by_severity"][issue.severity] += 1
                    review_summary["issues_by_type"][issue.type] += 1
                
            except Exception as e:
                print(f"Error reviewing {file_path}: {e}")
        
        review_summary["total_issues"] = len(all_issues)
        review_summary["quality_score"] = self._calculate_quality_score(all_issues, review_summary["files_analyzed"])
        
        # Generate review report
        report = self._generate_review_report(all_issues, review_summary, review_start)
        
        # Save to file if requested
        if output_file:
            self.file_ops.write_file(output_file, report)
        
        return {
            "success": True,
            "issues": [self._issue_to_dict(issue) for issue in all_issues],
            "summary": review_summary,
            "report": report,
            "files_analyzed": review_summary["files_analyzed"],
            "review_time": (datetime.now() - review_start).total_seconds()
        }
    
    async def _get_files_for_review(self, file_patterns: Optional[List[str]] = None) -> List[str]:
        """Get list of files to review."""
        
        if file_patterns:
            # Use provided patterns
            all_files = []
            for pattern in file_patterns:
                files = self.file_ops.find_files(
                    root_dir=Path.cwd(),
                    patterns=[pattern],
                    max_files=100
                )
                all_files.extend([f['path'] for f in files])
            return all_files
        
        # Default: review changed files or common code files
        changed_files = self.git_manager.get_changed_files()
        if changed_files:
            return [f['path'] for f in changed_files if self._is_reviewable_file(f['path'])]
        
        # Fallback: review common code files
        common_patterns = ['*.py', '*.js', '*.ts', '*.jsx', '*.tsx', '*.java', '*.cpp', '*.c']
        files = []
        for pattern in common_patterns:
            pattern_files = self.file_ops.find_files(
                root_dir=Path.cwd(),
                patterns=[pattern],
                max_files=20  # Limit for performance
            )
            files.extend([f['path'] for f in pattern_files])
        
        return files
    
    def _is_reviewable_file(self, file_path: str) -> bool:
        """Check if file should be included in review."""
        
        # Check file extension
        reviewable_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.rb', '.php'}
        ext = Path(file_path).suffix.lower()
        
        if ext not in reviewable_extensions:
            return False
        
        # Check if it's a text file and not too large
        if not self.file_ops.is_text_file(file_path):
            return False
        
        file_info = self.file_ops.get_file_info(file_path)
        if file_info.get('size', 0) > 1024 * 1024:  # Skip files larger than 1MB
            return False
        
        return True
    
    async def _review_single_file(self, file_path: str, review_scope: str) -> List[ReviewIssue]:
        """Review a single file and return issues."""
        
        content = self.file_ops.read_file(file_path)
        if not content:
            return []
        
        language = self.code_analyzer.detect_language(file_path)
        if not language:
            return []
        
        # Perform static analysis
        static_issues = await self._static_analysis(file_path, content, language)
        
        # Perform AI-based review
        ai_issues = await self._ai_code_review(file_path, content, language, review_scope)
        
        # Combine and deduplicate issues
        all_issues = static_issues + ai_issues
        return self._deduplicate_issues(all_issues)
    
    async def _static_analysis(self, file_path: str, content: str, language: str) -> List[ReviewIssue]:
        """Perform static code analysis."""
        
        issues = []
        
        # Use code analyzer for basic validation
        validation = self.code_analyzer.validate_code(content, language)
        
        # Convert validation results to review issues
        for error in validation.get('errors', []):
            issues.append(ReviewIssue(
                type='logic',
                severity='critical',
                file_path=file_path,
                line_number=None,
                title='Syntax Error',
                description=error,
                suggestion='Fix the syntax error to make the code parseable.'
            ))
        
        for warning in validation.get('warnings', []):
            issues.append(ReviewIssue(
                type='style',
                severity='minor',
                file_path=file_path,
                line_number=None,
                title='Code Warning',
                description=warning,
                suggestion='Consider addressing this warning to improve code quality.'
            ))
        
        # Additional language-specific checks
        if language == 'python':
            issues.extend(self._python_specific_checks(file_path, content))
        elif language in ['javascript', 'typescript']:
            issues.extend(self._javascript_specific_checks(file_path, content))
        
        return issues
    
    def _python_specific_checks(self, file_path: str, content: str) -> List[ReviewIssue]:
        """Python-specific code review checks."""
        
        issues = []
        lines = content.split('\n')
        
        for line_no, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for security issues
            if 'eval(' in line or 'exec(' in line:
                issues.append(ReviewIssue(
                    type='security',
                    severity='critical',
                    file_path=file_path,
                    line_number=line_no,
                    title='Dangerous eval/exec usage',
                    description='Using eval() or exec() can lead to code injection vulnerabilities.',
                    suggestion='Use safer alternatives like ast.literal_eval() or refactor the code.',
                    code_snippet=line.strip()
                ))
            
            # Check for performance issues
            if line_stripped.startswith('import ') and ' import *' in line:
                issues.append(ReviewIssue(
                    type='performance',
                    severity='minor',
                    file_path=file_path,
                    line_number=line_no,
                    title='Wildcard import',
                    description='Wildcard imports can impact performance and readability.',
                    suggestion='Import specific functions/classes instead of using wildcard.',
                    code_snippet=line.strip()
                ))
            
            # Check for maintainability issues
            if len(line) > 120:
                issues.append(ReviewIssue(
                    type='style',
                    severity='minor',
                    file_path=file_path,
                    line_number=line_no,
                    title='Long line',
                    description=f'Line is {len(line)} characters long.',
                    suggestion='Break long lines for better readability (PEP 8 recommends 79-88 characters).',
                    code_snippet=line.strip()[:50] + '...'
                ))
        
        return issues
    
    def _javascript_specific_checks(self, file_path: str, content: str) -> List[ReviewIssue]:
        """JavaScript/TypeScript-specific code review checks."""
        
        issues = []
        lines = content.split('\n')
        
        for line_no, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for security issues
            if 'eval(' in line or 'innerHTML' in line:
                issues.append(ReviewIssue(
                    type='security',
                    severity='major',
                    file_path=file_path,
                    line_number=line_no,
                    title='Potential XSS vulnerability',
                    description='Using eval() or innerHTML can lead to XSS attacks.',
                    suggestion='Use safer alternatives like textContent or proper sanitization.',
                    code_snippet=line.strip()
                ))
            
            # Check for style issues
            if '==' in line and '===' not in line and '!=' in line and '!==' not in line:
                issues.append(ReviewIssue(
                    type='style',
                    severity='minor',
                    file_path=file_path,
                    line_number=line_no,
                    title='Loose equality comparison',
                    description='Using == instead of === can lead to unexpected behavior.',
                    suggestion='Use strict equality (===) and inequality (!==) operators.',
                    code_snippet=line.strip()
                ))
            
            # Check for performance issues
            if '.querySelector(' in line and 'document.querySelector' in line:
                issues.append(ReviewIssue(
                    type='performance',
                    severity='info',
                    file_path=file_path,
                    line_number=line_no,
                    title='DOM query optimization',
                    description='Multiple DOM queries can impact performance.',
                    suggestion='Consider caching DOM elements or using more efficient selectors.',
                    code_snippet=line.strip()
                ))
        
        return issues
    
    async def _ai_code_review(self, file_path: str, content: str, language: str, review_scope: str) -> List[ReviewIssue]:
        """Perform AI-based code review."""
        
        # Build review prompt
        system_prompt = self._build_review_system_prompt(language, review_scope)
        user_prompt = self._build_review_user_prompt(file_path, content, language)
        
        # Get best model for code review
        model = await self.llm_manager.get_best_model_for_task("code_review")
        
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ]
        
        try:
            response = await self.llm_manager.chat_completion(
                messages=messages,
                model=model,
                temperature=0.2
            )
            
            # Parse AI response into issues
            return self._parse_ai_review_response(response.content, file_path)
            
        except Exception as e:
            print(f"Error in AI code review: {e}")
            return []
    
    def _build_review_system_prompt(self, language: str, review_scope: str) -> str:
        """Build system prompt for AI code review."""
        
        scope_description = self.review_categories.get(review_scope, 'all aspects of code quality')
        
        return f"""You are an expert code reviewer specializing in {language}.
Your task is to review code and identify issues related to {scope_description}.

Focus on:
- Security vulnerabilities and potential risks
- Performance bottlenecks and optimization opportunities
- Code style and formatting issues
- Logic errors and potential bugs
- Maintainability and readability concerns
- Missing error handling
- Code complexity and design patterns

For each issue found, provide:
1. Issue type (security/performance/style/logic/maintainability)
2. Severity (critical/major/minor/info)
3. Line number (if applicable)
4. Clear description of the problem
5. Specific suggestion for improvement

Format your response as a structured list."""
    
    def _build_review_user_prompt(self, file_path: str, content: str, language: str) -> str:
        """Build user prompt for AI code review."""
        
        return f"""Please review the following {language} code file: {file_path}

```{language}
{content}
```

Provide a detailed code review focusing on potential issues and improvements."""
    
    def _parse_ai_review_response(self, response: str, file_path: str) -> List[ReviewIssue]:
        """Parse AI review response into ReviewIssue objects."""
        
        issues = []
        
        # Simple parsing logic - in production, you might want more sophisticated parsing
        sections = response.split('\n\n')
        
        for section in sections:
            if not section.strip():
                continue
            
            # Try to extract issue information
            lines = section.split('\n')
            
            issue_data = {
                'type': 'maintainability',
                'severity': 'minor',
                'line_number': None,
                'title': 'Code Review Issue',
                'description': section,
                'suggestion': 'Please review and improve as suggested.'
            }
            
            # Look for patterns in the AI response
            for line in lines:
                line_lower = line.lower()
                
                # Extract severity
                if 'critical' in line_lower:
                    issue_data['severity'] = 'critical'
                elif 'major' in line_lower:
                    issue_data['severity'] = 'major'
                elif 'minor' in line_lower:
                    issue_data['severity'] = 'minor'
                
                # Extract type
                for issue_type in self.review_categories.keys():
                    if issue_type in line_lower:
                        issue_data['type'] = issue_type
                        break
                
                # Extract line number
                import re
                line_match = re.search(r'line\s+(\d+)', line_lower)
                if line_match:
                    issue_data['line_number'] = int(line_match.group(1))
            
            # Extract title from first line
            if lines:
                issue_data['title'] = lines[0][:100]  # Limit title length
            
            issues.append(ReviewIssue(
                type=issue_data['type'],
                severity=issue_data['severity'],
                file_path=file_path,
                line_number=issue_data['line_number'],
                title=issue_data['title'],
                description=issue_data['description'],
                suggestion=issue_data['suggestion']
            ))
        
        return issues
    
    def _deduplicate_issues(self, issues: List[ReviewIssue]) -> List[ReviewIssue]:
        """Remove duplicate issues."""
        
        seen = set()
        unique_issues = []
        
        for issue in issues:
            # Create a simple hash for deduplication
            issue_key = (issue.file_path, issue.line_number, issue.title)
            
            if issue_key not in seen:
                seen.add(issue_key)
                unique_issues.append(issue)
        
        return unique_issues
    
    def _calculate_quality_score(self, issues: List[ReviewIssue], files_analyzed: int) -> float:
        """Calculate overall code quality score."""
        
        if not files_analyzed:
            return 0.0
        
        # Start with perfect score
        score = 100.0
        
        # Deduct points based on issues
        for issue in issues:
            if issue.severity == 'critical':
                score -= 20
            elif issue.severity == 'major':
                score -= 10
            elif issue.severity == 'minor':
                score -= 5
            else:  # info
                score -= 2
        
        # Normalize by number of files
        score = score / files_analyzed
        
        # Ensure score is between 0 and 100
        return max(0.0, min(100.0, score))
    
    def _generate_review_report(self, issues: List[ReviewIssue], summary: Dict[str, Any], start_time: datetime) -> str:
        """Generate a comprehensive review report."""
        
        report_lines = [
            "# Code Review Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Review duration: {(datetime.now() - start_time).total_seconds():.2f} seconds",
            "",
            "## Summary",
            f"- Files analyzed: {summary['files_analyzed']}",
            f"- Total issues found: {summary['total_issues']}",
            f"- Quality score: {summary['quality_score']:.1f}/100",
            "",
            "### Issues by Severity",
        ]
        
        for severity, count in summary['issues_by_severity'].items():
            if count > 0:
                report_lines.append(f"- {severity.title()}: {count}")
        
        report_lines.extend([
            "",
            "### Issues by Type",
        ])
        
        for issue_type, count in summary['issues_by_type'].items():
            if count > 0:
                report_lines.append(f"- {issue_type.title()}: {count}")
        
        if issues:
            report_lines.extend([
                "",
                "## Detailed Issues",
                ""
            ])
            
            # Group issues by file
            issues_by_file = {}
            for issue in issues:
                if issue.file_path not in issues_by_file:
                    issues_by_file[issue.file_path] = []
                issues_by_file[issue.file_path].append(issue)
            
            for file_path, file_issues in issues_by_file.items():
                report_lines.append(f"### {file_path}")
                report_lines.append("")
                
                for issue in file_issues:
                    report_lines.append(f"**{issue.title}** ({issue.severity.upper()})")
                    if issue.line_number:
                        report_lines.append(f"Line: {issue.line_number}")
                    report_lines.append(f"Type: {issue.type}")
                    report_lines.append(f"Description: {issue.description}")
                    report_lines.append(f"Suggestion: {issue.suggestion}")
                    if issue.code_snippet:
                        report_lines.append(f"Code: `{issue.code_snippet}`")
                    report_lines.append("")
        
        return '\n'.join(report_lines)
    
    def _issue_to_dict(self, issue: ReviewIssue) -> Dict[str, Any]:
        """Convert ReviewIssue to dictionary."""
        
        return {
            "type": issue.type,
            "severity": issue.severity,
            "file_path": issue.file_path,
            "line_number": issue.line_number,
            "title": issue.title,
            "description": issue.description,
            "suggestion": issue.suggestion,
            "code_snippet": issue.code_snippet
        }
