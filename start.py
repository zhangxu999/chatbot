#!/home/xu/anaconda3/bin/python
# -*- coding: utf-8 -*-
import re
import sys

from rasa.__main__ import main

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.argv.append('train')
    sys.argv.append('--force')
    # sys.argv.append('run')
    # sys.argv.append('-p 6006 ')
    # sys.argv.append('--enable-api')
    # sys.argv.append('-v')
    sys.exit(main())
