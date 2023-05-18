import pandas as pd
import numpy as np
import math
import datetime

from ALG.utils import swap, get_next_candidate


def gfair_underranking(data_id, id_2_group, NUM_GROUPS, p, delta, K, rev):


    data_id = data_id[:2000]
    NUM_ELEMENTS = len(data_id)
    ALPHAS = []
    BETAS = []
    for j in range(NUM_GROUPS):
        ALPHAS.append(p[j]+delta)
        BETAS.append(p[j]-delta)
    
    
    EPSILON = 2
        
    
    BLOCK_SIZE = math.floor(EPSILON * K * 0.5) 
    U = [math.floor(i * BLOCK_SIZE) for i in ALPHAS]
    L = [math.ceil(i * BLOCK_SIZE) for i in BETAS]
    
    UPPER_BOUND = max(1, min( math.floor(min(U)) , BLOCK_SIZE - ( sum(L)-min(L) ) ) )
    NUM_BLOCKS = math.ceil(NUM_ELEMENTS/UPPER_BOUND)
    
    # print("delta: ",delta)
    # print("Num elements: ", NUM_ELEMENTS)
    # print("Num groups: ", NUM_GROUPS)
    # print("Alphas: ", ALPHAS)
    # print("Betas: ", BETAS)
    # print("U: ", U)
    # print("L: ", L)
    # print("Eps: ", EPSILON)
    # print("K: ", K)
    # print("Block size: ", BLOCK_SIZE)
    # print("Upper bound: ", UPPER_BOUND)
    # print("No of blocks: ", NUM_BLOCKS)

    target_data = []
    counter = np.zeros((NUM_BLOCKS, NUM_GROUPS))
    num_group_items = np.zeros((NUM_GROUPS))
    for block_num in range(NUM_BLOCKS):
        target_block = [None] * BLOCK_SIZE
        for rank in range(UPPER_BOUND):
            if (block_num)*UPPER_BOUND + rank > len(data_id) - 1 :
                break
            item = data_id[ (block_num)*UPPER_BOUND + rank]
            target_block[rank] = item
            counter[block_num][id_2_group[item]] += 1 
            num_group_items[id_2_group[item]] += 1 
        target_data.extend(target_block)
    # Loop across blocks
    for block_num in range(NUM_BLOCKS):
        START_ID = (block_num)*BLOCK_SIZE
        END_ID = (block_num)*BLOCK_SIZE + BLOCK_SIZE - 1

        
        # Looping inside each block
        for curr_id in range(START_ID, END_ID+1):
            # if curr_id == 500:
                # print(datetime.datetime.now().time())
            if (curr_id < len(target_data)) and (target_data[curr_id] is None):
                
                candidate_id = curr_id + 1
                
                
                # Try and fill the current None value from the remaining ids as per fairness
                while (candidate_id < len(target_data)): 
                    # Search for next non-empty ID
                    candidate_id = get_next_candidate(target_data, start = candidate_id) 
                    
                    candidate_block_num = math.floor(candidate_id/BLOCK_SIZE) 
                    
                    # if next candidate satisfies all the required conditions
                    if candidate_id == -1:
                        break
                    group_id = id_2_group[target_data[candidate_id]]
                    if (counter[block_num][group_id] < L[group_id]) or ( (BLOCK_SIZE - sum(counter[block_num])) > (sum(np.maximum(0,L-counter[block_num])))  and counter[block_num][group_id] < U[group_id]):
                        
                        swap(target_data, curr_id, candidate_id) # swap
                        counter[block_num][group_id] += 1
                        counter[candidate_block_num][group_id] -= 1
                        
                        break

                    candidate_id += 1 
            
        
    
    final_rank = [] 
    for item in target_data:
        if item is not None:
            final_rank.append(item)
    # print(len(final_rank))
    return final_rank

