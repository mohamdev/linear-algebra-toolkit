from scipy import signal

def butterworth_filter(data, cutoff_frequency, order=5, sampling_frequency=60):
    nyquist = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    return filtered_data

def low_pass_filter_data(data,nbutter=5):
    '''This function filters and elaborates data used in the identification process. 
    It is based on a return of experience  of Prof Maxime Gautier (LS2N, Nantes, France)'''
    
    b, a = signal.butter(nbutter, 0.01*5 / 2, "low")
   
    #data = signal.medfilt(data, 3)
    data= signal.filtfilt(
            b, a, data, axis=0, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1) )
    
    
    # suppress end segments of samples due to the border effect
    # nbord = 5 * nbutter
    # data = np.delete(data, np.s_[0:nbord], axis=0)
    # data = np.delete(data, np.s_[(data.shape[0] - nbord): data.shape[0]], axis=0)
     
    return data