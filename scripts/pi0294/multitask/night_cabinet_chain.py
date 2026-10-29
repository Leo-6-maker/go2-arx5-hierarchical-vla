"""Cabinet task selector for the existing finite source owner, no new scheduler."""
import sys
from .night_book_chain import main


if __name__=='__main__':main(sys.argv[1],task_kind='ROBOCASA_CABINET')
