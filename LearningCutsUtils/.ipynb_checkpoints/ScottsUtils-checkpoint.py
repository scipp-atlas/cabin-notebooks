import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import torch

def featSort(d):
    print('.')
    # takes awk array and converts to numpy and sorts features
    datalist=['ph.pt', 'ph.eta', 'ph.rhad', 'ph.rhad1', 'ph.reta', 'ph.rphi', 'ph.weta2', 'ph.eratio', 'ph.deltae', 'ph.wstot', 'ph.fside', 'ph.w1', 'ph.truth_pdgId', 'ph.truth_type', 'ph.convFlag']
    # for i in datalist:
    #     print(f'{i:7f},{len(data[i])},{type(data[i])}')
    mask = (d['ph.pt'] >= 15000) & (d['ph.pt'] < 20000) & (abs(d['ph.eta']) < 0.8) & (d['ph.wstot'] >= 0) & (d['ph.w1'] >= 0)
    print('.')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
    
    myrange_pt = (15,20)
    nbins_pt=20
    binwidth_pt=(myrange_pt[1]-myrange_pt[0])/nbins_pt
    ax1.hist(((d['ph.pt'])/1000)[mask],bins=nbins_pt,range=myrange_pt, color = 'grey')
    ax1.set_xlabel("Photon $p_{T}$ [GeV]")
    ax1.set_ylabel(f"Events/({binwidth_pt} GeV)")
    
    nbins_eta=20
    myrange_eta=(-4,4)
    binwidth_eta=(myrange_eta[1]-myrange_eta[0])/nbins_eta
    ax2.hist(d["ph.eta"][mask],bins=nbins_eta,range=myrange_eta, color = 'grey')
    ax2.set_xlabel("Psuedorapidity $\eta$")
    ax2.set_ylabel(f"Events/({binwidth_eta})")
    plt.tight_layout()
    numpy_data = {}
    
    for i in datalist:
        numpy_array = ak.to_numpy(d[i][mask])
        numpy_data[i] = numpy_array
        print(i,len(numpy_array),type(numpy_array))
        
    return numpy_data

def trueSort(d,dlist):
    # print(d['ph.truth_pdgId'],d['ph.truth_type'])
    loosetrues = []
    for i in range(len(d['ph.pt'])):
        if d['ph.truth_pdgId'][i] == 22 and d['ph.truth_type'][i]==15 or d['ph.truth_type'][i]==13 or d['ph.truth_type'][i]==14:
            true = 1
        else:
            true = 0
        loosetrues.append(true)
    loosetrues = torch.Tensor(loosetrues)
    # print(len(loosetrues))
    truedata = {}
    for i in range(0,11):
        truelist = []
        for j in range(0,len(d['ph.pt'])):
            if loosetrues[j] == 1:
                truelist.append(d[dlist[i]][j])
        # truelist = torch.Tensor(truelist)
        # print(i+1,dlist[i],len(truelist),type(truelist))
        truedata[i] = truelist
    falsedata = {}
    for i in range(0,11):
        falselist = []
        for j in range(0,len(d['ph.pt'])):
            if loosetrues[j] == 0:
                falselist.append(d[dlist[i]][j])
        # falselist = torch.Tensor(truelist)
        # print(i+1,dlist[i],len(falselist),type(falselist))
        falsedata[i] = falselist
    print('We are working with:')
    print(f'{len(truelist)} Signal Events')
    print(f'{len(falselist)} Backround Events')
    print(f'Signal:Backround = {len(truelist)/len(falselist)}')
    return truedata, falsedata

def plotTVF(t,f):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
    ranges = [(-0.05,0.25),(0.2,1.1),(0.2,1.1),(0,0.025),(0,1),(0,8000),(0,15),(0,1.),(0,1.)]
    latex_string = ['$R_{{had}}$','$R_{{\eta}}$','$R_{{\phi}}$','$w_{{\eta^{{2}}}}$','$E_{{ratio}}$','$\Delta E [MeV]$','$w^{{tot}}_{{\eta 1}}$','$F_{{side}}$','$w_{{\eta^{{3}}_{{\eta 1}}}}$']
    nbins = 20
    for i, ax in enumerate(axes.flatten()):
        ax.hist(f[i+2],density=True, range = ranges[i],bins=nbins,histtype = 'stepfilled',color='grey',alpha = 0.9, edgecolor='black', label = 'Backround')
        ax.hist(t[i+2], density=True, range = ranges[i],bins=nbins,histtype = 'stepfilled',color='white',alpha = 0.75, edgecolor='black', label = 'Signal')
        ax.set_xlabel(latex_string[i],loc = 'right')
        ax.set_ylabel(f"1/N dN/d({latex_string[i]})",loc = 'top')
        if i == 0:
            ax.legend()
            # ax.set_title('**ATLAS**\nPrelimiary\nData',y=0.85)
        ax.set_yscale('log')
    fig.tight_layout()

def plotFeats(d):
    datalist = ['ph.pt','ph.eta','ph.rhad1','ph.reta','ph.rphi','ph.weta2','ph.eratio','ph.deltae','ph.wstot','ph.fside','ph.w1','ph.truth_pdgId','ph.truth_type','ph.convFlag']
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 8))
    ranges = [(-0.05,0.25),(0.2,1.1),(0.2,1.1),(0,0.025),(0,1),(0,8000),(0,15),(0,1.),(0,1.)]
    latex_string = ['$R_{{had}}$','$R_{{\eta}}$','$R_{{\phi}}$','$w_{{\eta^{{2}}}}$','$E_{{ratio}}$','$\Delta E [MeV]$','$w^{{tot}}_{{\eta 1}}$','$F_{{side}}$','$w_{{\eta^{{3}}_{{\eta 1}}}}$']
    nbins = 20
    for i, ax in enumerate(axes.flatten()):
        ax.hist(d[datalist[i+2]], density=True, range = ranges[i],bins=nbins,histtype = 'stepfilled',color='grey',alpha = 0.75, edgecolor='black')
        ax.set_xlabel(latex_string[i],loc = 'right')
        ax.set_ylabel(f"1/N dN/d({latex_string[i]})",loc = 'top')
        ax.set_yscale('log')
    fig.tight_layout()

def test_train_split(d,dlist,ratio):
    loosetrues = []
    for i in range(len(d['ph.pt'])):
        if d['ph.truth_pdgId'][i] == 22 and d['ph.truth_type'][i]==15 or d['ph.truth_type'][i]==13 or d['ph.truth_type'][i]==14:
            true = 1
        else:
            true = 0
        loosetrues.append(true)
    loosetrues = torch.Tensor(loosetrues)
    x_train_list = []

    for i in range(2,11):
        x_train_array = torch.from_numpy(d[dlist[i]])
        x_train_list.append(x_train_array)
        # print(i-1,dlist[i], len(x_train_array), type(x_train_array))
    
    x_train_tensor = torch.stack(x_train_list)
    
    # print(x_train_tensor.shape,x_train_tensor.dtype,type(x_train_tensor))
    
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(x_train_tensor.T, loosetrues, test_size=ratio, random_state=42)
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    x_train_tensor = torch.tensor(X_train, dtype=torch.float).detach()
    y_train_tensor=y_train
    
    x_test_tensor = torch.tensor(X_test, dtype=torch.float).detach()
    y_test_tensor=y_test
    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor
