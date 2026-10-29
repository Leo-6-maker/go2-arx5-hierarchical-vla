"""Fixed RoboCasa raw-record audit through the existing source audit core."""
from .night_book_audit import audit as _audit


def audit(folder):return _audit(folder,task_kind='ROBOCASA_CABINET')
