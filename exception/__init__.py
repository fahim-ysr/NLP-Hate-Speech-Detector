# Importing required modules
import os
import sys


def error_message_details(error, error_detail:sys):
    """
    Gives tailored error message for exceptions
    """
    _, _, exc_tb = error_detail.exc_info()
    file = exc_tb.tb_frame.f_code.co_filename
    message = f"Error occured python script name [{file}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"

    return message


class CustomException(Exception):
    # Inherits exception and gives tailored error message with details

    def __init__(self, message, details):
        super().__init__(message)
        self.message = error_message_details(message, error_detail = details)

    def __str__(self):
        return self.message