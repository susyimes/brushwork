"""Command-line entry to brushwork.  Run `python brush.py --help`, or see skill/SKILL.md."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brushwork.cli import main

if __name__ == '__main__':
    sys.exit(main())
