import numpy as np
import pickle


# +
realworld_Hgraphs = ["coauth-dblp", "coauth-mag-geology", "coauth-mag-history", "dawn", "disgenenet",
                     "diseasome", "kaggle-whats-cooking", "ndc-classes", "ndc-substances",
                     "tags-ask-ubuntu", "tags-math-sx" , "tags-stack-overflow", "threads-ask-ubuntu", 
                     "threads-math-sx", "threads-stack-overflow",
                     "congress-bills", "contact-high-school", "contact-primary-school",
                     "email-enron", "email-eu", "hospital-lyon", "hypertext-conference", 
                     "invs13", "invs15",  "malawi-village", "science-gallery", "sfhh-conference"]



num_nodes = ["1,930,378", "1,261,129", "1,034,876", "2,558", "12,368", "516", "6,714", "1,161", "5,556",
            "3,029", "1,629", "49,998", "125,602", "176,445", "2,675,969",
            "1,718", "327", "242", "148", "1,005", "75", "113", "92", "232", "86", "10,972", "403"]

num_edges = ["3,700,681", "1,590,335", "1,812,511", "2,272,433", "2,261", "903", "39,774", "49,726", "112,405",
            "271,233", "822,059", "14,458,875", "192,947", "719,792", "11,305,356",
            "282,049", "172,035", "106,879", "10,885", "235,263", "27,834", "19,036",
            "9,644", "73,822", "99,942", "338,765", "54,305"]

num_edges = [int(x.replace(',', '')) for x in num_edges]
timestamp_dict = dict(zip(realworld_Hgraphs, num_edges))

Gtypes = ["orig", "PA", "ER", "PA_exact", "ER_exact"]


# +
def save_output(dataset, timesteps, Gtype):
    
    rk = []

    for cc in range(timesteps):

        if cc%100 == 0 and cc!=0:
            with open('./simulation/'+ dataset + '_' + Gtype + '_{}.pkl'.format(cc), 'rb') as fp:
                r = pickle.load(fp)
                
            i_size = max([x[0] for x in r.keys()])
            j_size = max([x[1] for x in r.keys()])
            k_size = max([x[2] for x in r.keys()])


            #print(i_size, j_size, k_size) 

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
    np.save('real_simulation/' + dataset + '_' + Gtype + ".npy", rk)   



def get_results(dataset):
    
    timesteps = timestamp_dict[dataset]
    
    for g in Gtypes:
        save_output(dataset, timesteps, g)


for data in realworld_Hgraphs:
    try:
        get_results(data)
    except:
        print(data, "not finished")
