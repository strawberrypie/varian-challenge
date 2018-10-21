#!/usr/bin/env python3

from varian.services.service import app
import logging
import sys

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    app.run(debug=True)

