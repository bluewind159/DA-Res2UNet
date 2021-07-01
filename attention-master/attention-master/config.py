# configuration file
import math

EPOCH = 200
BATCH = 128

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

def learning_rate1(init, epoch):
    optim_factor = 0
    if(epoch > 170):
        optim_factor = 5
    elif(epoch > 130):
        optim_factor = 4
    elif(epoch > 80):
        optim_factor = 3
    elif(epoch > 60):
        optim_factor = 2
    elif(epoch > 40):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
