class NumberOfDeviceError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
    
    def __str__(self,num_gpu,device) -> str:
        if num_gpu == 0 and device !=-1:
            return repr(f"The number of GPUs does not meet the relationship! No GPU detected, but received a GPU parameter{device}")
        if num_gpu > 0 and device > num_gpu:
            return repr(f"The expected GPU does not exist, there are a total of {num_gpu} GPUs, but parameter {device} was received")
        if num_gpu < -1:
            return repr(f"No such device can't be used!")


