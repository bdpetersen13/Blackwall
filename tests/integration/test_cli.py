"""
Integration Tests for CLI
"""
import pytest
from click.testing import CliRunner
from pathlib import Path
from blackwall.cli import main, batch_process


class TestCLI:
    """ Test CLI Functinality """
    
    def test_help_command(self):
        """ Test Help Output """
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        
        assert result.exit_code == 0
        assert "Blackwall - GenAI Detection Tool" in result.output
        assert "--file" in result.output
        assert "--mode" in result.output
    
    def test_version_command(self):
        """ Test Version Output """
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        
        assert result.exit_code == 0
        assert "0.1.0" in result.output
    
    def test_text_file_detection(self, sample_text_file: Path):
        """ Test Text File Detection via CLI """
        runner = CliRunner()
        result = runner.invoke(main, [
            "--file", str(sample_text_file),
            "--output", "json"
        ])
        
        assert result.exit_code in [0, 2]  # 0 for human, 2 for AI
        assert "probability" in result.output
        assert "file_type" in result.output
        assert '"text"' in result.output
    
    def test_invalid_file(self):
        """ Test Invalid File Handling """
        runner = CliRunner()
        result = runner.invoke(main, [
            "--file", "nonexistent.txt"
        ])
        
        assert result.exit_code == 2  # Click's file not found exit code
    
    def test_output_formats(self, sample_text_file: Path):
        """ Test Different Output Formats """
        runner = CliRunner()
        
        # Test minimal output
        result = runner.invoke(main, [
            "--file", str(sample_text_file),
            "--output", "minimal",
            "--quiet"
        ])
        assert len(result.output.strip().split('\n')) == 1
        
        # Test JSON output
        result = runner.invoke(main, [
            "--file", str(sample_text_file),
            "--output", "json"
        ])
        assert result.output.strip().startswith('{')
        assert result.output.strip().endswith('}')
    
    def test_batch_processing(self, temp_dir: Path):
        """Test batch processing."""
        # Create multiple files
        for i in range(3):
            file_path = temp_dir / f"test_{i}.txt"
            file_path.write_text(f"Test content {i}")
        
        runner = CliRunner()
        result = runner.invoke(batch_process, [
            "--directory", str(temp_dir),
            "--pattern", "*.txt",
            "--output", "summary"
        ])
        
        assert result.exit_code == 0
        assert "Found 3 files" in result.output
        assert "Batch Processing Summary" in result.output