"""PI0197 model-worker dependency path; native Python remains unchanged."""

import os
import sys


if sys.version_info >= (3, 12):
    _groot_site = "/runtime/groot-site"
    if os.path.isdir(_groot_site) and _groot_site not in sys.path:
        sys.path.insert(0, _groot_site)
