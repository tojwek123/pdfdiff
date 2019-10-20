import cv2
import numpy as np
import subprocess
import os
import argparse
import shutil
import itertools

PDFTOPPM_PATH = 'poppler-0.68.0/bin/pdftoppm.exe'
TEMP_DIR_PATH = '.tmp'

class PdfConversionError(Exception):
    pass

def pdf_to_png(pdf_filename, output_path, output_basename):
    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass
  
    proc = subprocess.Popen([PDFTOPPM_PATH, '-r', '100', '-png', pdf_filename, os.path.join(output_path, output_basename)])
    result = proc.wait()
    
    if result != 0:
        raise PdfConversionError('{} returned {}'.format(PDFTOPPM_PATH, result))
      
def main():
    parser = argparse.ArgumentParser(description='Compare two PDF-s')
    parser.add_argument('file_a', metavar='file_a', type=str, help='First file to compare')
    parser.add_argument('file_b', metavar='file_b', type=str, help='Second file to compare')
    parser.add_argument('output_path', metavar='output_path', type=str, help='Path where series of PNG files will be saved as a result of comparison')
    
    args = parser.parse_args()
    
    try:
        os.makedirs(args.output_path)
    except FileExistsError:
        pass
        
    a_temp_dir_path = os.path.join(TEMP_DIR_PATH, 'a')
    b_temp_dir_path = os.path.join(TEMP_DIR_PATH, 'b')

    print('Converting "{}" to png...'.format(args.file_a))
    pdf_to_png(args.file_a, a_temp_dir_path, 'a')
    print('Converting "{}" to png...'.format(args.file_b))
    pdf_to_png(args.file_b, b_temp_dir_path, 'b')
    
    a_filenames = os.listdir(a_temp_dir_path)
    b_filenames = os.listdir(b_temp_dir_path)
    
    a_filenames.sort()
    b_filenames.sort()
    
    for file_no, (a_filename, b_filename) in enumerate(itertools.zip_longest(a_filenames, b_filenames)):              
        has_difference = True
        
        if a_filename is not None:
            a_path = os.path.join(a_temp_dir_path, a_filename)
            a_im = cv2.imread(a_path)
        if b_filename is not None:
            b_path = os.path.join(b_temp_dir_path, b_filename)  
            b_im = cv2.imread(b_path)
            
        if a_filename is None:
            a_im = np.zeros(b_im.shape, dtype=np.uint8)
        elif b_filename is None:
            b_im = np.zeros(a_im.shape, dtype=np.uint8)
        
        if a_filename is not None and b_filename is not None:
            print('Comparing: "{}" vs. "{}"...'.format(a_filename, b_filename))
            
            diff = cv2.absdiff(a_im, b_im)
            ret, mask = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=3)
            
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
            
            if 0 == len(contours):
                has_difference = False
            else:
                for i, contour in enumerate(contours):
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
                    top = min(approx, key=lambda i: i[0][1])[0][1]
                    bottom = max(approx, key=lambda i: i[0][1])[0][1]
                    left = min(approx, key=lambda i: i[0][0])[0][0]
                    right = max(approx, key=lambda i: i[0][0])[0][0]
                    
                    cv2.rectangle(a_im, (left, top), (right, bottom), (0,0,255), 2)
                    cv2.rectangle(b_im, (left, top), (right, bottom), (0,0,255), 2)
            
        if has_difference:
            compare = np.concatenate((a_im, b_im), axis=1)
            cv2.line(compare, (compare.shape[1]//2, 0), (compare.shape[1]//2, compare.shape[0]), (0,0,0), 1) 
            
            dest_filename = 'compare-{}.png'.format(file_no + 1)
            dest_path = os.path.join(args.output_path, dest_filename)
            
            cv2.imwrite(dest_path, compare)
        
    shutil.rmtree(TEMP_DIR_PATH)
    print('Done')
    
if __name__ == '__main__':
    main()