import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from backbone_code.idcard_segment import IDCardSegment
from utils import is_image_file


def main():
    segmentor = IDCardSegment('preprocessing/detector.onnx', 0.8, 0.5, 'cuda')
    
    print('')


if __name__ == '__main__':
    main()
    print('Done')

    '''
    conda env ias
    run in preprocessing folder
    '''