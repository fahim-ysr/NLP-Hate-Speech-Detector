import sys
from logger import logging
from exception import CustomException

try:
    curr = 6/"7"

except Exception as e:
    raise CustomException(e, sys) from e