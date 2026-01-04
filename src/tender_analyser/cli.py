"""
Command-line interface for Tender Analyser.
"""

import json
import logging
import click
from pathlib import Path
from typing import Optional

from . import TenderAnalyser, __version__


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@click.group()
@click.version_option(version=__version__)
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """Tender Analyser - Analyze tender documents and extract key information."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    setup_logging(verbose)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(), help='Output file path')
@click.pass_context
def analyze(ctx, file_path, output):
    """Analyze a tender document."""
    try:
        analyser = TenderAnalyser()
        result = analyser.analyze(Path(file_path))
        
        click.echo(f"Analysis complete for: {file_path}")
        click.echo(f"Status: {result.get('status', 'unknown')}")
        
        if output:
            # Save results to output file
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
            click.echo(f"Results saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(), help='Output directory for results')
@click.pass_context
def batch(ctx, directory, output):
    """Analyze all tender documents in a directory."""
    try:
        dir_path = Path(directory)
        files = list(dir_path.glob('**/*.pdf')) + list(dir_path.glob('**/*.xlsx'))
        
        click.echo(f"Found {len(files)} files to analyze")
        
        analyser = TenderAnalyser()
        results = analyser.batch_analyze(files)
        
        click.echo(f"Analyzed {len(results)} files")
        
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_file = output_path / 'batch_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            click.echo(f"Results saved to: {results_file}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


def main():
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == '__main__':
    main()
