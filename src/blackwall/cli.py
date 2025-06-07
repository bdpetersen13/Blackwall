"""
Command-Line Interface for Blackwall
"""
import sys
from pathlib import Path
from typing import Optional, List
import time
import click
from rich.console import Console
from blackwall.config import get_config, Config
from blackwall.detectors import detect_file, get_detector
from blackwall.utils.file_handler import detect_file_type, validate_file
from blackwall.utils.logger import get_logger
from blackwall.utils.exceptions import BlackwallError
from blackwall.output.formatter import OutputFormatter, OutputFormat, print_banner, print_error, print_warning, print_info, print_success, create_progress_bar


logger = get_logger(__name__)
console = Console()


@click.command(name="blackwall")
@click.option(
    "--file", "-f",
    type=click.Path(exists = True, path_type = Path),
    required = True,
    help = "Path to the file to analyze"
)
@click.option(
    "--mode", "-m",
    type=click.Choice(["auto", "text", "image", "video"], case_sensitive = False),
    default = "auto",
    help = "Detection mode (auto-detect by default)"
)
@click.option(
    "--output", "-o",
    type=click.Choice(["plain", "json", "detailed", "minimal"], case_sensitive = False),
    default = "plain",
    help = "Output format"
)
@click.option(
    "--verbose", "-v",
    is_flag = True,
    help = "Enable verbose output"
)
@click.option(
    "--quiet", "-q",
    is_flag = True,
    help = "Suppress non-essential output"
)
@click.option(
    "--no-cache",
    is_flag = True,
    help = "Disable result caching"
)
@click.option(
    "--threshold", "-t",
    type = click.FloatRange(0.0, 1.0),
    default = 0.5,
    help = "Detection threshold (0.0-1.0)"
)
@click.version_option(version=get_config().version, prog_name = "blackwall")
def main(
    file: Path,
    mode: str,
    output: str,
    verbose: bool,
    quiet: bool,
    no_cache: bool,
    threshold: float
) -> int:
    """
    Blackwall - GenAI Detection Tool
    
    Detect AI-generated content in text, images, and videos.
    
    Examples:
    
        blackwall --file document.txt
        
        blackwall --file image.jpg --output json
        
        blackwall --file video.mp4 --mode video --verbose
    """
    try:
        # Show banner unless quiet
        if not quiet and output != "json":
            print_banner()
        
        # Update config with CLI options
        config = get_config()
        config.verbose = verbose
        config.enable_cache = not no_cache
        
        # Validate file
        if not quiet:
            print_info(f"Analyzing: {file.name}")
        
        validation = validate_file(file)
        if not validation["valid"]:
            print_error(f"File validation failed: {', '.join(validation['errors'])}")
            return 1
        
        # Determine file type
        if mode == "auto":
            try:
                file_type = detect_file_type(file)
                if not quiet:
                    print_info(f"Detected file type: {file_type}")
            except Exception as e:
                print_error(f"Failed to detect file type: {str(e)}")
                return 1
        else:
            file_type = mode
        
        # Run detection with progress
        if not quiet and output != "json":
            with create_progress_bar() as progress:
                task = progress.add_task(f"Processing {file_type} file...", total=None)
                result = detect_file(file)
                progress.update(task, completed = True)
        else:
            result = detect_file(file)
        
        # Format and display results
        formatter = OutputFormatter(OutputFormat(output))
        formatted_output = formatter.format_result(result, verbose=verbose)
        
        if formatted_output:  # Empty for rich output (already printed)
            print(formatted_output)
        
        # Exit code based on detection result and threshold
        if result.probability >= threshold:
            if not quiet and output != "json":
                print_warning(f"AI generation detected (probability: {result.probability:.1%} >= threshold: {threshold:.1%})")
            return 2  # Special exit code for AI detected
        else:
            if not quiet and output != "json":
                print_success(f"Content appears human-generated (probability: {result.probability:.1%} < threshold: {threshold:.1%})")
            return 0
    
    except BlackwallError as e:
        print_error(str(e), e)
        logger.error("blackwall_error", error = str(e), details = e.details)
        return 1
    
    except KeyboardInterrupt:
        print_error("Operation cancelled by user")
        return 130
    
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}", e)
        logger.error("unexpected_error", error = str(e), exc_info = True)
        return 1


@click.command(name = "blackwall-batch")
@click.option(
    "--directory", "-d",
    type=click.Path(exists = True, file_okay = False, dir_okay = True, path_type = Path),
    required = True,
    help = "Directory containing files to analyze"
)
@click.option(
    "--pattern", "-p",
    default = "*",
    help = "File pattern to match (e.g., '*.txt', '*.jpg')"
)
@click.option(
    "--recursive", "-r",
    is_flag = True,
    help = "Process files recursively"
)
@click.option(
    "--output", "-o",
    type=click.Choice(["summary", "detailed", "csv", "json"], case_sensitive = False),
    default = "summary",
    help = "Output format for batch results"
)
@click.option(
    "--threshold", "-t",
    type = click.FloatRange(0.0, 1.0),
    default = 0.5,
    help = "Detection threshold (0.0-1.0)"
)
def batch_process(
    directory: Path,
    pattern: str,
    recursive: bool,
    output: str,
    threshold: float
) -> int:
    """
    Process multiple files in batch mode.
    
    Examples:
    
        blackwall-batch --directory ./documents --pattern "*.txt"
        
        blackwall-batch --directory ./images --recursive --output csv
    """
    try:
        print_banner()
        print_info(f"Batch processing directory: {directory}")
        
        # Find files
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
        
        if not files:
            print_warning(f"No files found matching pattern: {pattern}")
            return 0
        
        print_info(f"Found {len(files)} files to process")
        
        # Process files
        results = []
        ai_detected_count = 0
        
        with create_progress_bar() as progress:
            task = progress.add_task("Processing files...", total = len(files))
            
            for file in files:
                try:
                    # Skip if not a file or not supported
                    if not file.is_file():
                        continue
                    
                    # Try to detect and process
                    result = detect_file(file)
                    results.append(result)
                    
                    if result.probability >= threshold:
                        ai_detected_count += 1
                    
                except Exception as e:
                    logger.warning(
                        "batch_file_failed",
                        file = str(file),
                        error = str(e)
                    )
                    results.append(None)
                
                finally:
                    progress.advance(task)
        
        # Display results based on output format
        _display_batch_results(results, output, threshold, ai_detected_count)
        
        return 0
    
    except Exception as e:
        print_error(f"Batch processing failed: {str(e)}")
        return 1


def _display_batch_results(
    results: List[Optional['DetectionResult']],
    output_format: str,
    threshold: float,
    ai_count: int
) -> None:
    """Display Batch Processing Results """
    valid_results = [r for r in results if r is not None]
    
    if output_format == "summary":
        console.print("\n[bold]Batch Processing Summary[/bold]")
        console.print(f"Total files processed: {len(results)}")
        console.print(f"Successful detections: {len(valid_results)}")
        console.print(f"Failed: {len(results) - len(valid_results)}")
        console.print(f"AI-generated files detected: {ai_count} ({ai_count / len(valid_results) * 100:.1f}%)")
        
        # Show top AI-detected files
        if ai_count > 0:
            console.print("\n[bold]Top AI-detected files:[/bold]")
            ai_files = sorted(
                [r for r in valid_results if r.probability >= threshold],
                key = lambda x: x.probability,
                reverse = True
            )[:10]
            
            for result in ai_files:
                console.print(f"  • {result.file_path.name}: {result.probability:.1%}")
    
    elif output_format == "json":
        import json
        output_data = {
            "summary": {
                "total_files": len(results),
                "successful": len(valid_results),
                "failed": len(results) - len(valid_results),
                "ai_detected": ai_count,
                "threshold": threshold
            },
            "results": [r.to_dict() if r else None for r in results]
        }
        print(json.dumps(output_data, indent=2))
    
    elif output_format == "csv":
        # Print CSV header
        print("file_path,file_type,probability,is_ai_generated,confidence,processing_time")
        
        # Print results
        for result in valid_results:
            print(
                f'"{result.file_path}",{result.file_type},{result.probability:.4f},'
                f'{result.is_ai_generated},{result.confidence},{result.processing_time:.3f}'
            )
    
    else:  # detailed
        formatter = OutputFormatter(OutputFormat.DETAILED)
        for result in valid_results:
            if result.probability >= threshold:
                formatter.format_result(result, verbose=True)
                console.print("-" * 80)


if __name__ == "__main__":
    sys.exit(main())