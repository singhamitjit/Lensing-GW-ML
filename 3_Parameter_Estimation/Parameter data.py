import numpy as np

n_specs = 4000

parameters = []

for i in range(n_specs):
    f = open('C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/Internship (Summer 2018)/Code/1. Lensed and Unlensed Model/Images/Unlensed/Unlensed_'+str(i+1)+'.txt')
    p = []
    for line in f:
        l = line.split(' ')
        for elem in l:
            try:
                p.append(float(elem))
            except ValueError:
                pass
    chirpm = ((p[0]*p[1])**(3/5))/((p[0]+p[1])**(1/5))
    p = p[2:]
    para = [chirpm]
    para.extend(p)
    parameters.append(para)

parameters = np.array(parameters)
print(parameters)


rank = np.arange(n_specs) + 1
np.savez('C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/Internship (Summer 2018)/Code/3. Parameter estimation/parameters_unlensed', parameters = parameters, rank=rank)
 


from IPython import get_ipython;   
get_ipython().magic('reset -sf')

import numpy as np

n_specs = 4000
print('lensed')
parameters = []

for i in range(n_specs):
    f = open('C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/Internship (Summer 2018)/Code/1. Lensed and Unlensed Model/Images/Lensed/Lensed_'+str(i+1)+'.txt')
    p = []
    for line in f:
        l = line.split(' ')
        for elem in l:
            try:
                p.append(float(elem))
            except ValueError:
                pass
    chirpm = ((p[0]*p[1])**(3/5))/((p[0]+p[1])**(1/5))
    p = p[2:]
    para = [chirpm]
    para.extend(p)
    parameters.append(para)

parameters = np.array(parameters)
print(parameters)


rank = np.arange(n_specs) + 1
np.savez('C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/Internship (Summer 2018)/Code/3. Parameter estimation/parameters_lensed', parameters = parameters, rank=rank)


