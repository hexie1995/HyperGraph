import xgi 
import time 
import os 
import importlib 

os.chdir("code")
m = importlib.import_module(".model", "src")

def data_load_test(data_set):
    
    print(data_set)
    print("-"*60)
    start = time.time()
    H = xgi.load_xgi_data(data_set)
    end = time.time()
    
    xgi_load_time = end - start
    
    start = time.time()
    H = H.copy()
    end = time.time()
    
    copy_time = end - start
    
    start = time.time()
    GH = m.GrowingHypergraph(H)
    end = time.time()
    
    conversion_time = end - start
    
    
    print(f"XGI loaded in {xgi_load_time:.2f} seconds.\nCopied in {copy_time:.2f} seconds.\nConverted to GrowingHypergraph in {conversion_time:.2f} seconds.")
    
    
if __name__ == "__main__":
    data_sets = ["email-enron", "email-eu", "science-gallery", "congress-bills", "threads-math-sx", "coauth-mag-history", "coauth-dblp", "threads-stack-overflow"]
    
    print("\n")
    for data_set in data_sets: 
        data_load_test(data_set)
        print()
    
    