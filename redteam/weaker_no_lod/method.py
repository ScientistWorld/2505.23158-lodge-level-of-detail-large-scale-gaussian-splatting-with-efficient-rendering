from pathlib import Path
import shutil
from baseline.full_representation import main
from redteam.common import NAMES
main()
for name in NAMES:
    shutil.copytree(Path('checkpoints/full_representation'), Path('checkpoints')/name, dirs_exist_ok=True)
