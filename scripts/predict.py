import sys, pathlib
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.predict import predict_next

if __name__ == '__main__':
    predict_next()
