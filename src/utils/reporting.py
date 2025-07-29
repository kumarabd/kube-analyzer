"""
Reporting Utility for KubeAnalyzer

This module provides utilities for generating reports from analyzer results
and formatting them in various output formats.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class Report:
    """Class for generating and managing analysis reports."""
    
    def __init__(
        self,
        name: str = "kube-analyzer-report",
        report_id: Optional[str] = None
    ):
        """
        Initialize a new report.
        
        Args:
            name: Name of the report
            report_id: Optional report ID, will generate if not provided
        """
        self.name = name
        self.created_at = datetime.now()
        self.report_id = report_id or self.created_at.strftime("%Y%m%d-%H%M%S")
        self.data = {
            "metadata": {
                "name": name,
                "report_id": self.report_id,
                "timestamp": self.created_at.isoformat(),
                "analyzer_version": "0.1.0"  # TODO: Get from version file
            },
            "summary": {},
            "sections": {}
        }
        logger.info(f"Created new report: {self.report_id}")
    
    def add_summary(self, summary: Dict[str, Any]) -> None:
        """
        Add summary data to the report.
        
        Args:
            summary: Dictionary containing report summary data
        """
        self.data["summary"] = summary
        logger.debug("Added summary to report")
    
    def add_section(self, section_name: str, section_data: Dict[str, Any]) -> None:
        """
        Add a section to the report.
        
        Args:
            section_name: Name of the section
            section_data: Dictionary containing section data
        """
        self.data["sections"][section_name] = section_data
        logger.debug(f"Added section '{section_name}' to report")
    
    def add_findings(self, section: str, findings: List[Dict[str, Any]]) -> None:
        """
        Add findings to a specific section.
        
        Args:
            section: Section name to add findings to
            findings: List of finding dictionaries
        """
        if section not in self.data["sections"]:
            self.data["sections"][section] = {"findings": []}
        
        if "findings" not in self.data["sections"][section]:
            self.data["sections"][section]["findings"] = []
        
        self.data["sections"][section]["findings"].extend(findings)
        logger.debug(f"Added {len(findings)} findings to section '{section}'")
    
    def add_recommendation(self, section: str, recommendation: Dict[str, Any]) -> None:
        """
        Add a recommendation to a specific section.
        
        Args:
            section: Section name to add recommendation to
            recommendation: Dictionary containing recommendation data
        """
        if section not in self.data["sections"]:
            self.data["sections"][section] = {"recommendations": []}
        
        if "recommendations" not in self.data["sections"][section]:
            self.data["sections"][section]["recommendations"] = []
        
        self.data["sections"][section]["recommendations"].append(recommendation)
        logger.debug(f"Added recommendation to section '{section}'")
    
    def get_report(self) -> Dict[str, Any]:
        """
        Get the complete report data.
        
        Returns:
            Complete report data as dictionary
        """
        return self.data
    
    def to_json(self, pretty: bool = True) -> str:
        """
        Convert report to JSON string.
        
        Args:
            pretty: Whether to format the JSON with indentation
            
        Returns:
            JSON string representation of the report
        """
        indent = 2 if pretty else None
        return json.dumps(self.data, indent=indent)
    
    def to_yaml(self) -> str:
        """
        Convert report to YAML string.
        
        Returns:
            YAML string representation of the report
        """
        return yaml.dump(self.data, default_flow_style=False)
    
    def save_to_file(
        self,
        output_format: str = "json",
        output_dir: str = "./reports",
        filename: Optional[str] = None
    ) -> str:
        """
        Save report to a file.
        
        Args:
            output_format: Format to save as (json, yaml)
            output_dir: Directory to save report in
            filename: Optional filename override
            
        Returns:
            Path to the saved report file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = self.created_at.strftime("%Y%m%d-%H%M%S")
            filename = f"{self.name}-{timestamp}.{output_format}"
        elif not filename.endswith(f".{output_format}"):
            filename = f"{filename}.{output_format}"
        
        file_path = os.path.join(output_dir, filename)
        
        # Write report to file
        try:
            with open(file_path, 'w') as f:
                if output_format.lower() == 'json':
                    f.write(self.to_json(pretty=True))
                elif output_format.lower() == 'yaml':
                    f.write(self.to_yaml())
                elif output_format.lower() == 'markdown':
                    f.write(ReportFormatter.format_markdown(self.data))
                elif output_format.lower() == 'html':
                    f.write(ReportFormatter.format_html(self.data))
                else:
                    logger.error(f"Unsupported output format: {output_format}")
                    return ""
            
            logger.info(f"Saved report to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save report to file: {str(e)}")
            return ""


class ReportFormatter:
    """Helper class for formatting report data into different output formats."""
    
    @staticmethod
    def format_markdown(report_data: Dict[str, Any]) -> str:
        """
        Format report data as Markdown.
        
        Args:
            report_data: Report data dictionary
            
        Returns:
            Markdown formatted report
        """
        lines = []
        
        # Add report header
        metadata = report_data.get("metadata", {})
        report_name = metadata.get("name", "Kubernetes Cluster Analysis Report")
        timestamp = metadata.get("timestamp", datetime.now().isoformat())
        
        lines.append(f"# {report_name}")
        lines.append(f"Generated on: {timestamp}")
        lines.append("")
        
        # Add summary section
        lines.append("## Summary")
        summary = report_data.get("summary", {})
        for key, value in summary.items():
            lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        lines.append("")
        
        # Add each section
        sections = report_data.get("sections", {})
        for section_name, section_data in sections.items():
            lines.append(f"## {section_name.replace('_', ' ').title()}")
            
            # Add section overview if it exists
            if "overview" in section_data:
                lines.append(section_data["overview"])
                lines.append("")
            
            # Add findings if they exist
            if "findings" in section_data:
                lines.append("### Findings")
                for finding in section_data["findings"]:
                    # Check if finding is a string instead of a dictionary
                    if isinstance(finding, str):
                        # Create a simple dictionary with the string as description
                        lines.append(f"#### Finding (INFO)")
                        lines.append(finding)
                        lines.append("")
                        continue
                        
                    # Normal dictionary processing
                    try:
                        title = finding.get("title", "Untitled Finding")
                        severity = finding.get("severity", "info").upper()
                        description = finding.get("description", "")
                        
                        lines.append(f"#### {title} ({severity})")
                        lines.append(description)
                        
                        # Add affected resources if they exist
                        if "affected_resources" in finding:
                            lines.append("")
                            lines.append("**Affected Resources:**")
                            for resource in finding["affected_resources"]:
                                lines.append(f"- {resource}")
                    except (AttributeError, TypeError) as e:
                        # Fallback for any other errors
                        logger.error(f"Error formatting finding: {e}, type: {type(finding)}")
                        lines.append(f"#### Finding (ERROR)")
                        lines.append(str(finding))
                    
                    lines.append("")
            
            # Add recommendations if they exist
            if "recommendations" in section_data:
                lines.append("### Recommendations")
                for rec in section_data["recommendations"]:
                    # Check if recommendation is a string instead of a dictionary
                    if isinstance(rec, str):
                        # Create a simple entry with the string as description
                        lines.append(f"#### Recommendation (MEDIUM)")
                        lines.append(rec)
                        lines.append("")
                        continue
                    
                    # Normal dictionary processing
                    try:
                        title = rec.get("title", "Untitled Recommendation")
                        priority = rec.get("priority", "medium").upper()
                        description = rec.get("description", "")
                        
                        lines.append(f"#### {title} ({priority})")
                        lines.append(description)
                        
                        # Add steps if they exist
                        if "steps" in rec:
                            lines.append("")
                            lines.append("**Steps:**")
                            for i, step in enumerate(rec["steps"], 1):
                                lines.append(f"{i}. {step}")
                    except (AttributeError, TypeError) as e:
                        # Fallback for any other errors
                        logger.error(f"Error formatting recommendation: {e}, type: {type(rec)}")
                        lines.append(f"#### Recommendation (ERROR)")
                        lines.append(str(rec))
                        
                    lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_html(report_data: Dict[str, Any]) -> str:
        """
        Format report data as HTML.
        
        Args:
            report_data: Report data dictionary
            
        Returns:
            HTML formatted report
        """
        # Convert to markdown first
        markdown = ReportFormatter.format_markdown(report_data)
        
        # Simple HTML wrapper for the markdown
        # In a real implementation, we'd use a proper markdown to HTML converter
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_data.get("metadata", {}).get("name", "Kubernetes Cluster Analysis")}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1, h2, h3, h4 {{ color: #333; }}
                h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                h2 {{ border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 30px; }}
                code {{ background-color: #f5f5f5; padding: 2px 5px; border-radius: 3px; }}
                .severity-HIGH {{ color: #d73a49; }}
                .severity-MEDIUM {{ color: #e36209; }}
                .severity-LOW {{ color: #b08800; }}
                .priority-HIGH {{ color: #d73a49; }}
                .priority-MEDIUM {{ color: #e36209; }}
                .priority-LOW {{ color: #b08800; }}
            </style>
        </head>
        <body>
            <pre>{markdown}</pre>
        </body>
        </html>
        """
        return html
    
    @staticmethod
    def format_table(
        data: List[Dict[str, Any]], 
        columns: List[str]
    ) -> str:
        """
        Format list of dictionaries as an ASCII table.
        
        Args:
            data: List of dictionaries
            columns: List of columns to include from the dictionaries
            
        Returns:
            ASCII table as string
        """
        if not data:
            return "No data available."
        
        # Calculate column widths
        col_widths = {col: len(col) for col in columns}
        for item in data:
            for col in columns:
                val = str(item.get(col, ""))
                col_widths[col] = max(col_widths[col], len(val))
        
        # Create header
        header = " | ".join(col.ljust(col_widths[col]) for col in columns)
        separator = "-+-".join("-" * col_widths[col] for col in columns)
        
        # Create rows
        rows = []
        for item in data:
            row = " | ".join(
                str(item.get(col, "")).ljust(col_widths[col]) for col in columns
            )
            rows.append(row)
        
        # Combine into table
        table = [header, separator] + rows
        return "\n".join(table)
