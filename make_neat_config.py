import os
def make_config(p_add_connection:float, 
                p_remove_connection:float, 
                p_add_node:float,
                p_remove_node:float,
                N_starting_hidden_neurons:int):

    os.makedirs("configs", exist_ok=True)
    # how many config files do we have already
    n_configs = sum([1 if 'neat_config' in f else 0 for f in os.listdir('configs')])    
    
    # this will be the Nth config file
    file_name = f'configs/neat_config{n_configs+1}.txt'

    keywords = {'conn_add_prob':p_add_connection,
                'conn_delete_prob':p_remove_connection,
                'node_add_prob':p_add_node,
                'node_delete_prob':p_remove_node,
                'num_hidden':N_starting_hidden_neurons}

    # open default config
    with open('neat_config.txt', 'r') as default:
        with open(file_name, 'w') as new:
            for line in default:
                if any(k in line for k in keywords) == True:
                    parameter = next(k for k in keywords if k in line)
                    start, end = line.split('=')
                    # update with our parameter value
                    end = str(keywords[parameter])
                    new_line = start + '= ' + end +'\n'
                    new.write(new_line)
                # copy all other lines
                else:
                    new.write(line)

if __name__ == '__main__':
    # test case
    make_config(0.9,
                0.977,
                0.3439,
                0.97,
                37)



