"""Four fixed task selectors on one existing finite owner/account/lease."""
import sys
from .night_book_chain import main


if __name__=='__main__':main(sys.argv[1],task_kind='ALL_PINNED')
