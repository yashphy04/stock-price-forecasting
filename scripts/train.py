import sys, pathlib
# add project root to path
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.train import train

if __name__ == '__main__':
    train()
