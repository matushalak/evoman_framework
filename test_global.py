import random

global env



def print_env():
    print(env)


def runner():
    
    update_global()

def update_global():

    global env 

    env = random.uniform(80, 95)

    print_env()




if __name__ == '__main__':
    for i in range(10):
        runner()