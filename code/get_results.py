import numpy as np
import pickle


timesteps = int(1e6)
rk = []

for cc in range(timesteps):
    
    if cc%100 == 0:
        with open('simulation_4/r_{}.pkl'.format(cc), 'rb') as fp:
            r = pickle.load(fp)

        i_size = max([x[0] for x in r.keys()])
        j_size = max([x[1] for x in r.keys()])
        k_size = max([x[2] for x in r.keys()])


        print(i_size, j_size, k_size) 

        r_ijk = np.zeros(shape=(i_size+1, j_size+1, k_size+1))

        for key in r.keys():
            r_ijk[key] = r[key]

        # Transpose the array to bring the first two axes to the end
        array_transposed = np.transpose(r_ijk, (2, 0, 1))
        # Apply np.triu along the last two axes
        upper_triangle_transposed = np.triu(array_transposed)
        # Transpose the result back to the original shape
        upper_triangle = np.transpose(upper_triangle_transposed, (1, 2, 0))


        normalized_rijk = upper_triangle/np.sum(upper_triangle)

        rk.append(np.sum(normalized_rijk, axis = (0,1)))
        
rk = np.array(rk)
np.save("rk_4.npy", rk)