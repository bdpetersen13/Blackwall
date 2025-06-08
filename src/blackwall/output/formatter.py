"""
Output Formatting for Blackwall
Handles Different Output Formats and Presentation Styles
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
from colorama import init, Fore, Style
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from blackwall.detectors.base import DetectionResult
from blackwall.config import get_config
from blackwall.utils.logger import get_logger


# Initialize colorama for Windows support
init(autoreset=True)

logger = get_logger(__name__)
console = Console()


class OutputFormat(str, Enum):
    """ Supported Output Formats """
    PLAIN = "plain"
    JSON = "json"
    DETAILED = "detailed"
    MINIMAL = "minimal"


class OutputFormatter:
    """ Format Detection Results for Display"""
    
    def __init__(self, format_type: OutputFormat = OutputFormat.PLAIN):
        self.format_type = format_type
        self.config = get_config()
    
    def format_result(
        self,
        result: DetectionResult,
        verbose: bool = False
    ) -> str:
        """ Format Detection Result Based on Outut """
        if self.format_type == OutputFormat.JSON:
            return self._format_json(result)
        elif self.format_type == OutputFormat.MINIMAL:
            return self._format_minimal(result)
        elif self.format_type == OutputFormat.DETAILED or verbose:
            return self._format_detailed(result)
        else:
            return self._format_plain(result)
    
    def _format_plain(self, result: DetectionResult) -> str:
        """ Format Result in Plain Text """
        # Determine color based on probability
        if result.probability >= 0.7:
            color = Fore.RED
            status = "AI-GENERATED"
        elif result.probability >= 0.3:
            color = Fore.YELLOW
            status = "UNCERTAIN"
        else:
            color = Fore.GREEN
            status = "LIKELY HUMAN"
        
        output = []
        output.append(f"\n{Fore.CYAN}═══ Blackwall Detection Result ═══{Style.RESET_ALL}")
        output.append(f"\n📄 File: {result.file_path.name}")
        output.append(f"🔍 Type: {result.file_type.upper()}")
        output.append(f"📊 Probability: {color}{result.probability:.1%}{Style.RESET_ALL}")
        output.append(f"🎯 Status: {color}{status}{Style.RESET_ALL}")
        output.append(f"⚡ Confidence: {self._format_confidence(result.confidence)}")
        output.append(f"⏱️  Processing Time: {result.processing_time:.2f}s")
        
        # Add notes
        if result.notes:
            output.append(f"\n📝 Notes:")
            for note in result.notes:
                output.append(f"   • {note}")
        
        return "\n".join(output)
    
    def _format_minimal(self, result: DetectionResult) -> str:
        """Format Minimal Result, One-Liner """
        status = "AI" if result.is_ai_generated else "HUMAN"
        return f"{result.file_path.name}: {result.probability:.1%} {status}"
    
    def _format_detailed(self, result: DetectionResult) -> str:
        """ Format Detailed Result with Metadata """
        # Create main panel
        title = f"[bold cyan]Blackwall Detection Report[/bold cyan]"
        
        # Build content
        content = []
        
        # File info section
        file_info = Table(show_header = False, box = None, padding = (0, 1))
        file_info.add_column("Field", style = "dim")
        file_info.add_column("Value")
        
        file_info.add_row("File Path", str(result.file_path))
        file_info.add_row("File Type", result.file_type.upper())
        file_info.add_row("Timestamp", result.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"))
        file_info.add_row("Model Version", result.model_version)
        
        content.append(Panel(file_info, title = "[bold]File Information[/bold]", border_style = "blue"))
        
        # Detection results section
        results_table = Table(show_header = False, box = None, padding = (0, 1))
        results_table.add_column("Metric", style = "dim")
        results_table.add_column("Value")
        
        # Probability with color
        prob_color = self._get_probability_color(result.probability)
        prob_text = Text(f"{result.probability:.1%}", style=prob_color)
        results_table.add_row("AI Probability", prob_text)
        
        # Status
        status = "AI-GENERATED" if result.is_ai_generated else "LIKELY HUMAN"
        status_color = "red" if result.is_ai_generated else "green"
        results_table.add_row("Detection Result", Text(status, style = f"bold {status_color}"))
        
        # Confidence
        conf_text = Text(result.confidence.upper(), style = self._get_confidence_color(result.confidence))
        results_table.add_row("Confidence Level", conf_text)
        
        results_table.add_row("Processing Time", f"{result.processing_time:.3f} seconds")
        
        content.append(Panel(results_table, title = "[bold]Detection Results[/bold]", border_style = "yellow"))
        
        # Metadata section (if verbose)
        if result.metadata:
            self._add_metadata_section(content, result)
        
        # Notes section
        if result.notes:
            notes_text = "\n".join(f"• {note}" for note in result.notes)
            content.append(Panel(notes_text, title="[bold]Analysis Notes[/bold]", border_style = "cyan"))
        
        # Print using rich
        console.print(Panel.fit(
            "\n".join(str(c) for c in content),
            title = title,
            border_style = "bright_blue"
        ))
        
        return ""  # Rich handles the printing
    
    def _format_json(self, result: DetectionResult) -> str:
        """ Format Result as JSON"""
        return json.dumps(result.to_dict(), indent = 2)
    
    def _add_metadata_section(self, content: list, result: DetectionResult) -> None:
        """ Add Metadata Section to Detailed Output """
        metadata = result.metadata
        
        if result.file_type == "text":
            meta_table = Table(show_header = False, box = None, padding = (0, 1))
            meta_table.add_column("Feature", style = "dim")
            meta_table.add_column("Value")
            
            if "text_length" in metadata:
                meta_table.add_row("Text Length", f"{metadata['text_length']:,} characters")
            if "model_probability" in metadata:
                meta_table.add_row("Model Score", f"{metadata['model_probability']:.4f}")
            if "feature_score" in metadata:
                meta_table.add_row("Feature Score", f"{metadata['feature_score']:.4f}")
            
            # Add features
            if "features" in metadata:
                features = metadata["features"]
                if "vocabulary_diversity" in features:
                    meta_table.add_row("Vocabulary Diversity", f"{features['vocabulary_diversity']:.3f}")
                if "word_repetition_rate" in features:
                    meta_table.add_row("Repetition Rate", f"{features['word_repetition_rate']:.3f}")
            
            # Add top indicators
            if "top_indicators" in metadata:
                indicators = "\n".join(f"• {ind}" for ind in metadata["top_indicators"])
                meta_table.add_row("Key Indicators", indicators)
            
            content.append(Panel(meta_table, title = "[bold]Text Analysis Details[/bold]", border_style = "magenta"))
        
        elif result.file_type == "image":
            meta_table = Table(show_header = False, box = None, padding = (0, 1))
            meta_table.add_column("Feature", style = "dim")
            meta_table.add_column("Value")
            
            if "image_size" in metadata:
                meta_table.add_row("Image Size", f"{metadata['image_size'][0]}x{metadata['image_size'][1]}")
            if "format" in metadata:
                meta_table.add_row("Format", metadata["format"])
            if "model_probability" in metadata:
                meta_table.add_row("Model Score", f"{metadata['model_probability']:.4f}")
            if "pattern_score" in metadata:
                meta_table.add_row("Pattern Score", f"{metadata['pattern_score']:.4f}")
            
            # Add anomalies
            if "anomalies" in metadata:
                anomalies = "\n".join(f"• {a}" for a in metadata["anomalies"])
                meta_table.add_row("Detected Anomalies", anomalies)
            
            content.append(Panel(meta_table, title = "[bold]Image Analysis Details[/bold]", border_style = "magenta"))
        
        elif result.file_type == "video":
            meta_table = Table(show_header = False, box = None, padding = (0, 1))
            meta_table.add_column("Feature", style = "dim")
            meta_table.add_column("Value")
            
            if "video_duration" in metadata:
                meta_table.add_row("Duration", f"{metadata['video_duration']:.1f} seconds")
            if "video_resolution" in metadata:
                meta_table.add_row("Resolution", metadata["video_resolution"])
            if "fps" in metadata:
                meta_table.add_row("Frame Rate", f"{metadata['fps']} FPS")
            if "frame_count" in metadata:
                meta_table.add_row("Frames Analyzed", str(metadata["frame_count"]))
            if "ai_frame_count" in metadata:
                meta_table.add_row("AI-Detected Frames", str(metadata["ai_frame_count"]))
            
            content.append(Panel(meta_table, title = "[bold]Video Analysis Details[/bold]", border_style = "magenta"))
    
    def _format_confidence(self, confidence: str) -> str:
        """ Format Confidence Level with Color """
        colors = {
            "high": Fore.GREEN,
            "medium": Fore.YELLOW,
            "low": Fore.RED
        }
        color = colors.get(confidence.lower(), Fore.WHITE)
        return f"{color}{confidence.upper()}{Style.RESET_ALL}"
    
    def _get_probability_color(self, probability: float) -> str:
        """ Get Color for Probability Value """
        if probability >= 0.7:
            return "red"
        elif probability >= 0.3:
            return "yellow"
        else:
            return "green"
    
    def _get_confidence_color(self, confidence: str) -> str:
        """ Get Color for Confidence Level"""
        colors = {
            "high": "bright_green",
            "medium": "yellow",
            "low": "red"
        }
        return colors.get(confidence.lower(), "white")


def create_progress_bar() -> Progress:
    """ Create a Progress Bar for File Processing """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient = True,
        console = console
    )


def print_banner():
    """ Print Application Banner """
    banner = """

    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                                                                   ║
    ║                                                                                                                                   ║
    ║   ░▒▓███████▓▒░░▒▓█▓▒░       ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓█▓▒░          ║
    ║   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░          ║
    ║   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░          ║
    ║   ░▒▓███████▓▒░░▒▓█▓▒░      ░▒▓████████▓▒░▒▓█▓▒░      ░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░      ░▒▓█▓▒░          ║
    ║   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░          ║
    ║   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░          ║
    ║   ░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█████████████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓████████▓▒░   ║
    ║                                                                                                                                   ║
    ║                                                                                                                                   ║
    ║                                               GenAI Detection Tool v0.1.0                                                         ║                                                                                                                    
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    
    """
    console.print(banner, style = "bright_cyan")


def print_error(message: str, error: Optional[Exception] = None):
    """ Print Error Message"""
    error_text = f"[bold red]❌ Error:[/bold red] {message}"
    if error and get_config().verbose:
        error_text += f"\n[dim]Details: {str(error)}[/dim]"
    console.print(error_text)


def print_warning(message: str):
    """Print Warning Message """
    console.print(f"[bold yellow]⚠️  Warning:[/bold yellow] {message}")


def print_info(message: str):
    """ Print Info Message """
    console.print(f"[bold blue]ℹ️  Info:[/bold blue] {message}")


def print_success(message: str):
    """ Print Success Message """
    console.print(f"[bold green]✅ Success:[/bold green] {message}")