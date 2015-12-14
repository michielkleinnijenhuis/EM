#!/usr/bin/env python

"""
Register the slices in a stack.
"""

import sys
import argparse
from ij import IJ
from register_virtual_stack import Register_Virtual_Stack_MT

def main(argv):
    
#     parser = argparse.ArgumentParser(description='...')
#     
#     parser.add_argument('sourcedir', help='...')
#     parser.add_argument('targetdir', help='...')
#     parser.add_argument('transformdir', help='...')
#     parser.add_argument('refname', help='...')
#     
#     args = parser.parse_args()
#     
#     source_dir = args.sourcedir
#     target_dir = args.targetdir
#     transf_dir = args.transformdir
#     reference_name = args.refname
    source_dir = 'SOURCE_DIR/'
    target_dir = 'TARGET_DIR/'
    transf_dir = 'TARGET_DIR/trans/'
    reference_name = 'REFNAME'
    
    use_shrinking_constraint = 0
    p = Register_Virtual_Stack_MT.Param()
    p.sift.maxOctaveSize = 1024
    p.minInlierRatio = 0.05
    Register_Virtual_Stack_MT.exec(source_dir, target_dir, transf_dir, reference_name, p, use_shrinking_constraint)

if __name__ == "__main__":
    main(sys.argv)
