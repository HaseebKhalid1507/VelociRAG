#!/usr/bin/env python3

import sys
import tempfile
import shutil
import os
from pathlib import Path

# Add velocirag to path
sys.path.insert(0, '/home/haseeb/velocirag/src')

from velocirag.cli import cli
from click.testing import CliRunner

def test_hybrid_cli():
    """Test the --hybrid flag in CLI."""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test markdown file with large sections
        test_file = temp_path / "test_hybrid.md"
        test_content = """---
title: Hybrid CLI Test
category: testing
---

# Test Document

## Large Section About Programming

This section discusses programming languages in detail. Python is a versatile language used for web development, data science, and automation. It has a simple syntax that makes it beginner-friendly. JavaScript is another popular language that runs in browsers and on servers. It's essential for web development and has a large ecosystem of libraries and frameworks. Both languages have their strengths and are widely used in the industry.

## Large Section About Cooking

This section covers culinary arts and cooking techniques. Professional cooking requires understanding of heat, timing, and ingredient interactions. Different cooking methods like sautéing, braising, and roasting produce different flavors and textures. French cuisine emphasizes technique and precise execution. Italian cooking focuses on high-quality ingredients and simplicity. Asian cuisines bring unique flavor profiles with their use of spices and fermented ingredients.
"""
        
        test_file.write_text(test_content)
        
        # Create database directory
        db_dir = temp_path / ".velocirag"
        
        # Test CLI runner
        runner = CliRunner()
        
        print("Testing --hybrid flag...")
        
        # Test hybrid indexing
        result = runner.invoke(cli, [
            'index', str(temp_path),
            '--db', str(db_dir),
            '--hybrid',
            '--threshold', '20.0',
            '--large-section', '200'
        ])
        
        print(f"Exit code: {result.exit_code}")
        print(f"Output: {result.output}")
        
        if result.exit_code == 0:
            print("✅ Hybrid CLI integration successful!")
            
            # Check if database was created
            store_db = db_dir / "store.db"
            if store_db.exists():
                print("✅ Database created successfully!")
                
                # Try a search
                search_result = runner.invoke(cli, [
                    'search', 'programming',
                    '--db', str(db_dir),
                    '--limit', '3'
                ])
                
                print(f"Search exit code: {search_result.exit_code}")
                print(f"Search output: {search_result.output}")
                
                if search_result.exit_code == 0:
                    print("✅ Search working with hybrid chunks!")
                else:
                    print("❌ Search failed")
            else:
                print("❌ Database not created")
        else:
            print("❌ Hybrid CLI integration failed")
            print(f"Error: {result.output}")
        
        print("\nTesting --hybrid with --semantic (should fail)...")
        
        # Test that hybrid and semantic flags conflict
        conflict_result = runner.invoke(cli, [
            'index', str(temp_path),
            '--db', str(db_dir),
            '--hybrid',
            '--semantic'
        ])
        
        print(f"Conflict test exit code: {conflict_result.exit_code}")
        print(f"Conflict test output: {conflict_result.output}")
        
        if conflict_result.exit_code != 0 and "Cannot use both" in conflict_result.output:
            print("✅ Conflict detection working correctly!")
        else:
            print("❌ Conflict detection not working")

if __name__ == "__main__":
    test_hybrid_cli()