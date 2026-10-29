"""One explicitly registered cabinet child through the existing source worker."""
import sys
from .night_book_worker import run


if __name__=='__main__':sys.exit(run(sys.argv[1],sys.argv[2],task_kind='ROBOCASA_CABINET'))
