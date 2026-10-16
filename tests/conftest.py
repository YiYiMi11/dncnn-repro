"""pytest 共享配置：把仓库根目录加进 sys.path，这样 `from src.model import ...` 到处都能用。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))