#!/usr/bin/env python

"""
Register the slices in a stack.
"""

import sys
from ij import IJ
from register_virtual_stack import Register_Virtual_Stack_MT

def main(argv):
    
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
    main(sys.argv[1:])
