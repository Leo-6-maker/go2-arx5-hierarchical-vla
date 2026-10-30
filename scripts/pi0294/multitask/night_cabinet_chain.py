"""Only the two original, previously unrun cabinet conditions; no retries."""
import sys
from .night_book_chain import main

if __name__=='__main__':main(sys.argv[1],task_kind='ROBOCASA_CABINET')
