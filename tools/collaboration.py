"""
Collaboration Tools for the Finance Analyst AI Agent.
Implements features for sharing analyses, multi-user support, and team workflows.
"""

import os
import json
import uuid
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import sqlite3
from langchain.tools import BaseTool


class AnalysisReport:
    """Class representing a financial analysis report that can be shared."""
    
    def __init__(self, title: str, author: str, content: Dict[str, Any], 
                 tickers: List[str] = None, tags: List[str] = None):
        """
        Initialize a new analysis report.
        
        Args:
            title: Title of the analysis report
            author: Creator of the analysis
            content: Dictionary containing analysis data and results
            tickers: List of ticker symbols related to this analysis
            tags: List of tags for categorization
        """
        self.id = str(uuid.uuid4())
        self.title = title
        self.author = author
        self.content = content
        self.tickers = tickers or []
        self.tags = tags or []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.shared_with = []
        self.comments = []
        
    def add_comment(self, author: str, text: str) -> Dict[str, Any]:
        """Add a comment to the analysis."""
        comment = {
            "id": str(uuid.uuid4()),
            "author": author,
            "text": text,
            "timestamp": datetime.now()
        }
        self.comments.append(comment)
        self.updated_at = datetime.now()
        return comment
    
    def share_with(self, user_id: str) -> None:
        """Share this analysis with another user."""
        if user_id not in self.shared_with:
            self.shared_with.append(user_id)
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the analysis to a dictionary for storage."""
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "content": self.content,
            "tickers": self.tickers,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "shared_with": self.shared_with,
            "comments": self.comments
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisReport':
        """Create an analysis report from a dictionary."""
        report = cls(
            title=data["title"],
            author=data["author"],
            content=data["content"],
            tickers=data["tickers"],
            tags=data["tags"]
        )
        report.id = data["id"]
        report.created_at = datetime.fromisoformat(data["created_at"])
        report.updated_at = datetime.fromisoformat(data["updated_at"])
        report.shared_with = data["shared_with"]
        report.comments = data["comments"]
        return report


class AnalysisRepository:
    """Repository for storing and retrieving financial analyses."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize the repository with database connection.
        
        Args:
            db_path: Path to SQLite database file (default: analyses.db in data directory)
        """
        if db_path is None:
            # Create data directory if it doesn't exist
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            db_path = data_dir / "analyses.db"
        
        self.db_path = str(db_path)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create analyses table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            content TEXT NOT NULL,
            tickers TEXT,
            tags TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')
        
        # Create shares table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS shares (
            analysis_id TEXT,
            user_id TEXT,
            PRIMARY KEY (analysis_id, user_id),
            FOREIGN KEY (analysis_id) REFERENCES analyses (id)
        )
        ''')
        
        # Create comments table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id TEXT PRIMARY KEY,
            analysis_id TEXT,
            author TEXT NOT NULL,
            text TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (analysis_id) REFERENCES analyses (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save(self, report: AnalysisReport) -> str:
        """
        Save an analysis report to the database.
        
        Args:
            report: The AnalysisReport to save
            
        Returns:
            The ID of the saved report
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save main report
        cursor.execute(
            "INSERT OR REPLACE INTO analyses VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                report.id,
                report.title,
                report.author,
                json.dumps(report.content),
                json.dumps(report.tickers),
                json.dumps(report.tags),
                report.created_at.isoformat(),
                report.updated_at.isoformat()
            )
        )
        
        # Delete existing shares and comments
        cursor.execute("DELETE FROM shares WHERE analysis_id = ?", (report.id,))
        cursor.execute("DELETE FROM comments WHERE analysis_id = ?", (report.id,))
        
        # Save shares
        for user_id in report.shared_with:
            cursor.execute(
                "INSERT INTO shares VALUES (?, ?)",
                (report.id, user_id)
            )
        
        # Save comments
        for comment in report.comments:
            cursor.execute(
                "INSERT INTO comments VALUES (?, ?, ?, ?, ?)",
                (
                    comment["id"],
                    report.id,
                    comment["author"],
                    comment["text"],
                    comment["timestamp"].isoformat() if isinstance(comment["timestamp"], datetime) else comment["timestamp"]
                )
            )
        
        conn.commit()
        conn.close()
        
        return report.id
    
    def get_by_id(self, report_id: str) -> Optional[AnalysisReport]:
        """Retrieve an analysis report by its ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get report
        cursor.execute("SELECT * FROM analyses WHERE id = ?", (report_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # Convert to dictionary
        report_data = dict(row)
        report_data["content"] = json.loads(report_data["content"])
        report_data["tickers"] = json.loads(report_data["tickers"])
        report_data["tags"] = json.loads(report_data["tags"])
        
        # Get shares
        cursor.execute("SELECT user_id FROM shares WHERE analysis_id = ?", (report_id,))
        shared_with = [row["user_id"] for row in cursor.fetchall()]
        report_data["shared_with"] = shared_with
        
        # Get comments
        cursor.execute("SELECT * FROM comments WHERE analysis_id = ?", (report_id,))
        comments = [dict(row) for row in cursor.fetchall()]
        report_data["comments"] = comments
        
        conn.close()
        
        return AnalysisReport.from_dict(report_data)
    
    def search(self, 
               author: Optional[str] = None, 
               ticker: Optional[str] = None,
               tag: Optional[str] = None,
               shared_with: Optional[str] = None) -> List[AnalysisReport]:
        """
        Search for analyses matching given criteria.
        
        Args:
            author: Filter by author
            ticker: Filter by related ticker
            tag: Filter by tag
            shared_with: Filter by user it's shared with
            
        Returns:
            List of matching AnalysisReport objects
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT id FROM analyses"
        conditions = []
        params = []
        
        if author:
            conditions.append("author = ?")
            params.append(author)
        
        if ticker:
            conditions.append("tickers LIKE ?")
            params.append(f"%{ticker}%")
        
        if tag:
            conditions.append("tags LIKE ?")
            params.append(f"%{tag}%")
        
        if shared_with:
            query = """
            SELECT a.id FROM analyses a
            JOIN shares s ON a.id = s.analysis_id
            """
            conditions.append("s.user_id = ?")
            params.append(shared_with)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        cursor.execute(query, params)
        report_ids = [row["id"] for row in cursor.fetchall()]
        
        conn.close()
        
        return [self.get_by_id(report_id) for report_id in report_ids]
    
    def delete(self, report_id: str) -> bool:
        """Delete an analysis report by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM shares WHERE analysis_id = ?", (report_id,))
        cursor.execute("DELETE FROM comments WHERE analysis_id = ?", (report_id,))
        cursor.execute("DELETE FROM analyses WHERE id = ?", (report_id,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return deleted


class ShareAnalysisTool(BaseTool):
    name = "share_analysis"
    description = """
    Creates and shares financial analysis reports with other users.
    
    Args:
        title: Title of the analysis report
        author: User creating the report
        content: Dictionary containing analysis data and results
        tickers: List of ticker symbols related to this analysis
        tags: List of tags for categorization
        share_with: List of user IDs to share with (optional)
        
    Returns:
        Details of the saved analysis including its ID
    """
    
    def __init__(self, repository: Optional[AnalysisRepository] = None):
        """Initialize the tool with a repository."""
        super().__init__()
        self.repository = repository or AnalysisRepository()
    
    def _run(self, title: str, author: str, content: Dict[str, Any], 
             tickers: List[str], tags: List[str] = None,
             share_with: List[str] = None) -> Dict[str, Any]:
        try:
            # Create analysis report
            report = AnalysisReport(
                title=title,
                author=author,
                content=content,
                tickers=tickers,
                tags=tags or []
            )
            
            # Share if specified
            if share_with:
                for user_id in share_with:
                    report.share_with(user_id)
            
            # Save to repository
            self.repository.save(report)
            
            return {
                "status": "success",
                "message": f"Analysis '{title}' created and shared with {len(share_with) if share_with else 0} users",
                "analysis_id": report.id,
                "details": {
                    "title": report.title,
                    "author": report.author,
                    "created_at": report.created_at.isoformat(),
                    "tickers": report.tickers,
                    "tags": report.tags,
                    "shared_with": report.shared_with
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error creating analysis: {str(e)}"
            }
    
    async def _arun(self, title: str, author: str, content: Dict[str, Any], 
                    tickers: List[str], tags: List[str] = None,
                    share_with: List[str] = None) -> Dict[str, Any]:
        return self._run(title, author, content, tickers, tags, share_with)


class FindAnalysisTool(BaseTool):
    name = "find_analysis"
    description = """
    Searches for financial analyses that match specific criteria.
    
    Args:
        author: Filter by author (optional)
        ticker: Filter by stock ticker (optional)
        tag: Filter by analysis tag (optional)
        shared_with: Filter by user ID the analysis is shared with (optional)
        
    Returns:
        List of matching analyses with their details
    """
    
    def __init__(self, repository: Optional[AnalysisRepository] = None):
        """Initialize the tool with a repository."""
        super().__init__()
        self.repository = repository or AnalysisRepository()
    
    def _run(self, author: Optional[str] = None, ticker: Optional[str] = None,
             tag: Optional[str] = None, shared_with: Optional[str] = None) -> Dict[str, Any]:
        try:
            if not any([author, ticker, tag, shared_with]):
                return {
                    "status": "error",
                    "message": "At least one search parameter (author, ticker, tag, shared_with) must be provided"
                }
            
            reports = self.repository.search(author, ticker, tag, shared_with)
            
            results = []
            for report in reports:
                results.append({
                    "id": report.id,
                    "title": report.title,
                    "author": report.author,
                    "created_at": report.created_at.isoformat(),
                    "updated_at": report.updated_at.isoformat(),
                    "tickers": report.tickers,
                    "tags": report.tags,
                    "comment_count": len(report.comments),
                    "shared_with_count": len(report.shared_with)
                })
            
            return {
                "status": "success",
                "count": len(results),
                "analyses": results
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error searching analyses: {str(e)}"
            }
    
    async def _arun(self, author: Optional[str] = None, ticker: Optional[str] = None,
                    tag: Optional[str] = None, shared_with: Optional[str] = None) -> Dict[str, Any]:
        return self._run(author, ticker, tag, shared_with)


class GetAnalysisDetailsTool(BaseTool):
    name = "get_analysis_details"
    description = """
    Retrieves the full details of a specific financial analysis by its ID.
    
    Args:
        analysis_id: The ID of the analysis to retrieve
        
    Returns:
        Complete details of the analysis including content and comments
    """
    
    def __init__(self, repository: Optional[AnalysisRepository] = None):
        """Initialize the tool with a repository."""
        super().__init__()
        self.repository = repository or AnalysisRepository()
    
    def _run(self, analysis_id: str) -> Dict[str, Any]:
        try:
            report = self.repository.get_by_id(analysis_id)
            
            if not report:
                return {
                    "status": "error",
                    "message": f"Analysis with ID {analysis_id} not found"
                }
            
            return {
                "status": "success",
                "analysis": {
                    "id": report.id,
                    "title": report.title,
                    "author": report.author,
                    "created_at": report.created_at.isoformat(),
                    "updated_at": report.updated_at.isoformat(),
                    "tickers": report.tickers,
                    "tags": report.tags,
                    "content": report.content,
                    "shared_with": report.shared_with,
                    "comments": report.comments
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error retrieving analysis: {str(e)}"
            }
    
    async def _arun(self, analysis_id: str) -> Dict[str, Any]:
        return self._run(analysis_id)


class CommentOnAnalysisTool(BaseTool):
    name = "comment_on_analysis"
    description = """
    Adds a comment to an existing financial analysis.
    
    Args:
        analysis_id: ID of the analysis to comment on
        author: User making the comment
        text: Content of the comment
        
    Returns:
        Details of the added comment
    """
    
    def __init__(self, repository: Optional[AnalysisRepository] = None):
        """Initialize the tool with a repository."""
        super().__init__()
        self.repository = repository or AnalysisRepository()
    
    def _run(self, analysis_id: str, author: str, text: str) -> Dict[str, Any]:
        try:
            report = self.repository.get_by_id(analysis_id)
            
            if not report:
                return {
                    "status": "error",
                    "message": f"Analysis with ID {analysis_id} not found"
                }
            
            comment = report.add_comment(author, text)
            self.repository.save(report)
            
            return {
                "status": "success",
                "message": f"Comment added to analysis '{report.title}'",
                "comment": {
                    "id": comment["id"],
                    "author": comment["author"],
                    "text": comment["text"],
                    "timestamp": comment["timestamp"].isoformat() if isinstance(comment["timestamp"], datetime) else comment["timestamp"]
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error adding comment: {str(e)}"
            }
    
    async def _arun(self, analysis_id: str, author: str, text: str) -> Dict[str, Any]:
        return self._run(analysis_id, author, text)


class TeamWorkflowTool(BaseTool):
    name = "team_workflow"
    description = """
    Manages team workflows for collaborative financial analysis.
    
    Args:
        action: The workflow action to perform (create_task, assign_task, complete_task, track_progress)
        params: Dictionary with parameters specific to the action
        
    Returns:
        Result of the requested workflow action
    """
    
    def __init__(self, repository: Optional[AnalysisRepository] = None):
        """Initialize the tool with a repository."""
        super().__init__()
        self.repository = repository or AnalysisRepository()
        self._tasks = {}  # Simple in-memory tasks store (would be DB-based in production)
    
    def _run(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if action == "create_task":
                return self._create_task(params)
            elif action == "assign_task":
                return self._assign_task(params)
            elif action == "complete_task":
                return self._complete_task(params)
            elif action == "track_progress":
                return self._track_progress(params)
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {action}. Supported actions: create_task, assign_task, complete_task, track_progress"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error in team workflow: {str(e)}"
            }
    
    def _create_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new analysis task."""
        required_params = ["title", "description", "created_by", "due_date"]
        for param in required_params:
            if param not in params:
                return {
                    "status": "error",
                    "message": f"Missing required parameter: {param}"
                }
        
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "title": params["title"],
            "description": params["description"],
            "created_by": params["created_by"],
            "due_date": params["due_date"],
            "assigned_to": params.get("assigned_to", None),
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "related_tickers": params.get("related_tickers", []),
            "comments": []
        }
        
        self._tasks[task_id] = task
        
        return {
            "status": "success",
            "message": f"Task '{params['title']}' created successfully",
            "task_id": task_id,
            "task": task
        }
    
    def _assign_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assign a task to a user."""
        if "task_id" not in params or "assigned_to" not in params:
            return {
                "status": "error",
                "message": "Missing required parameters: task_id and assigned_to"
            }
        
        task_id = params["task_id"]
        if task_id not in self._tasks:
            return {
                "status": "error",
                "message": f"Task with ID {task_id} not found"
            }
        
        task = self._tasks[task_id]
        task["assigned_to"] = params["assigned_to"]
        task["updated_at"] = datetime.now().isoformat()
        
        return {
            "status": "success",
            "message": f"Task '{task['title']}' assigned to {params['assigned_to']}",
            "task": task
        }
    
    def _complete_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mark a task as completed."""
        if "task_id" not in params:
            return {
                "status": "error",
                "message": "Missing required parameter: task_id"
            }
        
        task_id = params["task_id"]
        if task_id not in self._tasks:
            return {
                "status": "error",
                "message": f"Task with ID {task_id} not found"
            }
        
        task = self._tasks[task_id]
        task["status"] = "completed"
        task["updated_at"] = datetime.now().isoformat()
        task["completed_at"] = datetime.now().isoformat()
        
        # Link to analysis if provided
        if "analysis_id" in params:
            task["related_analysis"] = params["analysis_id"]
        
        return {
            "status": "success",
            "message": f"Task '{task['title']}' marked as completed",
            "task": task
        }
    
    def _track_progress(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Track progress of tasks by user or status."""
        tasks = self._tasks.values()
        
        # Filter by user if specified
        if "user_id" in params:
            tasks = [t for t in tasks if t.get("assigned_to") == params["user_id"] or t.get("created_by") == params["user_id"]]
        
        # Filter by status if specified
        if "status" in params:
            tasks = [t for t in tasks if t.get("status") == params["status"]]
            
        # Filter by ticker if specified
        if "ticker" in params:
            ticker = params["ticker"].upper()
            tasks = [t for t in tasks if ticker in [rt.upper() for rt in t.get("related_tickers", [])]]
        
        # Prepare summary
        summary = {
            "total": len(tasks),
            "by_status": {
                "pending": len([t for t in tasks if t.get("status") == "pending"]),
                "in_progress": len([t for t in tasks if t.get("status") == "in_progress"]),
                "completed": len([t for t in tasks if t.get("status") == "completed"])
            },
            "tasks": sorted(tasks, key=lambda t: t.get("due_date", "9999-12-31"))
        }
        
        return {
            "status": "success",
            "summary": summary
        }
    
    async def _arun(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(action, params)
