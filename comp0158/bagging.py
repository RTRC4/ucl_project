
import numpy as np




def find_block_length(num_qrters):
    """Function to determine the optimal block length.
    
    Parameters
    - num_points: A single numeric value giving the number of datapoints
    
    Return
    - block_length: A single numeric value giving the length of the block
    - num_blocks: A single numeric value giving the number of blocks"""
    
    pos_block_lengths=np.arange(10,15)
    pos_num_blocks=num_qrters/pos_block_lengths
    dis_int=np.abs(pos_num_blocks-np.round(pos_num_blocks))
    num_blocks=int(pos_num_blocks[np.argmin(dis_int)])
    block_length=pos_block_lengths[np.argmin(dis_int)]
    
    return block_length, num_blocks
    


def block_bootstrapping(num_points, sample_size = 5, overlap = True):
    """Function to apply block bootstrapping. 
    
    Parameters
    - num_points: A single numeric value giving the number of datapoints
    
    Return
    - sampled_idx: A numpy array containing the index positions of the new
                   bootstrapped permutation."""
    
    # Find the number of quarters
    num_qrters=num_points/3
    
    # Given the number of quarters, find the block length
    block_length, num_blocks=find_block_length(num_qrters=num_qrters)
    block_length*=3
    
    # Find the index positions to be used to find the blocks
    index_pos=np.arange(0,num_points)
    index_pos=index_pos[(num_points-block_length*num_blocks):]
    
    # Create a list of index positions for the values of each block
    blocks=[index_pos[i*block_length:i*block_length+block_length] for i in range(0,num_blocks)]
    
    # Randomly select the blocks
    rand_blocks_pos=np.random.choice(np.arange(0,num_blocks),num_blocks)
    
    # Join the blocks together to form the new permutation
    chosen_blocks=[blocks[i] for i in rand_blocks_pos]
    sampled_idx=np.array([x for l in chosen_blocks for x in l])
    
    return sampled_idx



def quarterly_average(df,num_months=3):
    """Function to calculate the quarterly average across a monthly numpy array.
    This quarterly average is calculated by calendar quarter, with the values for
    the months of the given quarter replaced by the quarter average for each column.
    Note, we can also replace the monthly values by the quarterly average calculated
    using just the first 2 months, or the first month.
    
    Parameters
    - df: A numpy array containing the data that we want to calculate the quarterly
          average.
    - num_months: A single numeric value determining the number of months of the
                  quarter that we are calculating the average over. If 3, then
                  we are calculating the average across the 3 months of the quarter.
                  If 2, we calculate the average based on the first 2 months of
                  the quarter. If 1, we replace the monthly values in a given 
                  quarter by the value of the first month of the quarter.
                  
    Return
    - qrt_avg: A numpy array with the values converted to quarterly average."""
    
    num_months=3-num_months
    num_qrters=int(len(df)/3)
    qrt_avg_lt=list()
    for i in range(num_qrters):
        temp_avg=np.mean(df[i*3:(i+1)*3-num_months,:],axis=0)
        temp_avg=np.reshape(temp_avg,(1,len(temp_avg)))
        temp_avg_m=np.repeat(temp_avg,3,0)
        qrt_avg_lt.append(temp_avg_m)
        
    qrt_avg=np.concatenate(qrt_avg_lt,axis=0)
    return(qrt_avg)


