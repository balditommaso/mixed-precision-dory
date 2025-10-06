from qonnx.transformation.base import Transformation
from datetime import datetime
from warnings import warn
from typing import *


class BaseTrasformation(Transformation):
    """
    Convert Quant node of the parameters, such as weight and bias, to 
    thir quantized repperesentation and redirect the new tensor as input 
    of the operation node and store useful information.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        
    def warning_message(self, msg: str) -> str:
        if self.verbose:
            now = datetime.now()
            warn(f"{now} - WARNING [{self.__class__.__name__}] : {msg}")
            
            
    def info_message(self, msg: str) -> str:
        if self.verbose:
            now = datetime.now()
            print(f"{now} - INFO [{self.__class__.__name__}] : {msg}")
            
            
    def debug_message(self, msg: str) -> str:
        if self.verbose:
            now = datetime.now()
            print(f"{now} - DEBUG [{self.__class__.__name__}] : {msg}")
            
            
    def error_message(self, msg: str, e: ValueError) -> str:
        now = datetime.now()
        msg = f"{now} - ERROR [{self.__class__.__name__}] : {msg}"
        raise e(msg)
            
    

            