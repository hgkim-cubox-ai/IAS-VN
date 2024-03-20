import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import is_image_file


def main():
    model_path = 'detector.onnx'


if __name__ == '__main__':
    main()
    print('Done')