
import numpy as np
import pandas as pd
import dataframe_image as dfi
import pickle


class tc():
    def __init__(self, filename, vg_start, vg_end, step, shift, vds, linear_fit_steps):
        # linear_fit_steps: list
        # shift: list
        self.filename = filename
        self.load(filename, vg_start, vg_end, step, shift, vds, linear_fit_steps, type)
        self.to_pandas(filename)
        
        
    def load(self, filename,vg_start, vg_end, step, shift, vds, linear_fit_steps, type):
        vg_min = min(vg_start, vg_end)
        vg_max = max(vg_start, vg_end)
        
        raw_data = pd.read_csv(filename + ".csv", header=None).to_numpy().T
        raw_data[1] = raw_data[1]/vds
        
        if raw_data[0][0] > raw_data[0][step]:
            data_dn = np.flip(raw_data[:,:step], axis=1)
            data_up = raw_data[:,step:]
        else:
            data_up = raw_data[:,:step]
            data_dn = np.flip(raw_data[:,step:], axis=1)
        
        # self.raw_data = raw_data
        # self.data_up = data_up
        # self.data_dn = data_dn
        
        self.up = tc_single(data_up, vg_min, vg_max, step, shift, linear_fit_steps)
        self.dn = tc_single(data_dn, vg_min, vg_max, step, shift, linear_fit_steps)

    def to_pandas(self, filename):
        df_up_n = self.up.n.params
        df_up_p = self.up.p.params
        df_dn_n = self.dn.n.params
        df_dn_p = self.dn.p.params
        self.params = pd.concat([df_dn_n,df_dn_p,df_up_n,df_up_p], keys=['dn_N', 'dn_P',"up_N",'up_P'])
        # self.params.to_html(filename + '_params.html')
        dfi.export(self.params.round(3), filename + '_params.png')
    def __str__(self):
        rep = self.params.round(3)
        return str(rep)
        
class tc_single():
    def __init__(self, data, vg_min, vg_max, step, shift, linear_fit_steps):
        p_data = np.zeros(data.shape)
        p_data[0] = np.flip(-data[0])
        p_data[1] = np.flip(data[1])
        self.n = tc_single_type(data, vg_min, vg_max, step, shift, linear_fit_steps)
        self.p = tc_single_type(p_data, -vg_max, -vg_min, step, shift, linear_fit_steps)
        
class tc_single_type():
    def __init__(self, data, vg_min, vg_max, step, shift, linear_fit_steps):
        self.step_size = (vg_max - vg_min)/(step-1)
        self.vg_min = vg_min
        self.vg_max= vg_max
        self.step = step
        self.shift = np.array(shift) * self.step_size
        self.fit_steps = np.array(linear_fit_steps) 
        self.data = data
        
        # self.data is a nparray, of shape (2, step)
        self.ids = np.zeros(len(linear_fit_steps))
        self.mu_max = np.zeros(len(linear_fit_steps))
        self.vt = np.zeros(len(linear_fit_steps))
        self.ss = np.zeros(len(linear_fit_steps))
        self.vt_fit_data = []
        self.fit()
        self.mu = np.zeros((len(linear_fit_steps), len(shift)))
        self.shift_fit_data = [[] for _ in range(len(self.fit_steps))]
        self.shift_fit()
        
        self.to_pandas()
        
    def lin_reg(self, data):
        slope, intercept =  np.polyfit(data[0],data[1],1)
        mu = np.abs(slope/1.2E-8)
        vt = -intercept/slope
        ids = np.mean(data[1])
        ss, _ =  np.polyfit(np.log(np.abs(data[1])),data[0],1)
        return mu, vt, ids, ss, slope
    
    def fit(self):
        for i,step in enumerate(self.fit_steps):
            lin_data = self.data[:,-step:]
            self.mu_max[i], self.vt[i], self.ids[i], self.ss[i], slope = self.lin_reg(lin_data)
            vt_fit_data = np.zeros(self.data.shape)
            vt_fit_data[0] = self.data[0]
            vt_fit_data[1] = slope * (self.data[0] - self.vt[i])
            mask = (vt_fit_data[1] >=0)
            self.vt_fit_data.append(vt_fit_data[:,mask])
    
    def shift_fit(self):
        fit_steps_resized = self.fit_steps * self.step_size
        for i,step in enumerate(fit_steps_resized):
            for j,sf in enumerate(self.shift):
                # run linear fit data in self.data[:,mask],
                # where mask is ideally the intersection of:
                # (self.data[0] >= self.vt[i]+ sf) & (self.data[0] <= self.vt[i] + sf + step)
                # and vg_min, vg_max
                # only do linear reg when mask is non empty
                mask_min = max(self.vg_min,self.vt[i]+ sf)
                mask_max = min(self.vg_max,self.vt[i] + sf + step)
                if mask_max <= mask_min:
                    self.mu[i,j] = np.nan
                    self.shift_fit_data[i].append(np.array([]))
                else:
                    mask = (self.data[0] >= mask_min) & (self.data[0] <= mask_max)
                    lin_data = self.data[:,mask]
                    if len(lin_data[0]) <= 1:
                        self.mu[i,j] = np.nan
                        self.shift_fit_data[i].append(np.array([]))
                    else:
                        self.mu[i,j], _, _, _, _ = self.lin_reg(lin_data)
                        self.shift_fit_data[i].append(lin_data)
                    
    def to_pandas(self):
        df = pd.DataFrame(columns=["Step_size_"+str(round(step, 2)) for step in self.step_size * self.fit_steps])
        df.loc['Vt'] = self.vt
        df.loc['mu_max'] = self.mu_max 
        for i,mu in enumerate(self.mu.T):
            df.loc['mu_shift_'+str(round(self.shift[i],2)) + '_V'] = mu
        df.loc['ss'] = self.ss
        self.params = df

def save(result:tc):
    file = open(result.filename + '_pickle.pkl',"wb")
    pickle.dump(result, file)
    file.close()
    


# # %%
# input_params = {
#     'filename':'test', 
#     'vg_start':50,
#     'vg_end':-50,
#     'shift':[15,20,30],
#     'linear_fit_steps':[5,10,12,15,17],
#     'step':101,
#     'vds':0.1
# }

# # %%
# test = tc(**input_params)

# # %%
# test.params

# # %%
# dfi.export(test.params.round(3), 'df_styled.png')

# # %%
# # file = open("test_pickle","wb")
# # pickle.dump(test, file)
# # file.close()

# # file = open("test_pickle", "rb")
# # test_read = pickle.load(file)
# # file.close()

# # %%
# file = open("test_pickle", "rb")
# test_read = pickle.load(file)
# file.close()

# # %%


# # %%



