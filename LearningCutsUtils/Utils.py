import numpy as np
import matplotlib.pyplot as plt
import torch

# Utility function for seeing what happens when taking gradient of loss function.
def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])
                

def ListToGraph(l,bins,color,low=0,high=1,weight=0):
    counts,bin_edges = np.histogram(l,bins,range=(low,high))
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    err = np.sqrt(counts)
    if weight==0:
        weight=1./(sum(counts)*(bin_edges[1]-bin_edges[0]))
    counts = weight*counts
    err = weight*err
    return plt.errorbar(bin_centres, counts, yerr=err, fmt='o',color=color)




def make_ROC_curve(y_test, y_pred_test):
    fpr, tpr, _ = roc_curve(y_test, y_pred_test.numpy())
    roc_auc = roc_auc_score(y_test, y_pred_test.numpy())
    
    plt.figure(1)
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", label="DNN (area = {:.3f})".format(roc_auc))
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    

def plot_classifier_output(y_train, y_pred_train,y_test, y_pred_test):
    signal_train=[]
    signal_test =[]
    backgr_train=[]
    backgr_test =[]
    for y,y_p in zip(y_train,y_pred_train):
        if y==1: signal_train.append(float(y_p))
        else:    backgr_train.append(float(y_p))
    for y,y_p in zip(y_test,y_pred_test):
        if y==1: signal_test.append(float(y_p))
        else:    backgr_test.append(float(y_p))
    
    nbins=20
    signal_train_hist=plt.hist(signal_train,nbins,density=True,range=(0,1),histtype='stepfilled',alpha=0.5,color='red')
    backgr_train_hist=plt.hist(backgr_train,nbins,density=True,range=(0,1),histtype='stepfilled',alpha=0.5,color='blue')
    signal_test=ListToGraph(signal_test,nbins,"red")
    backgr_test=ListToGraph(backgr_test,nbins,"blue")
    plt.yscale("log")
    plt.show()


def plotlosses(losses, test_losses):
    plt.plot([l.totalloss().detach().numpy() for l in losses], '.', label="Train")
    plt.plot([l.totalloss().detach().numpy() for l in test_losses], '.', label="Test")
    plt.plot([l.efficloss.detach().numpy() for l in losses], '.', label="Train: effic")
    plt.plot([l.backgloss.detach().numpy() for l in losses], '.', label="Train: backg")
    plt.plot([l.cutszloss.detach().numpy() for l in losses], '.', label="Train: cutsz")
    if type(losses[0].monotloss) is not int:
        # this particular term can get very small, just cut it off for super small values
        plt.plot([max(l.monotloss.detach().numpy(),1e-12) for l in losses], '.', label="Train: smooth")
    plt.legend()
    plt.xlabel('Training Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.yscale('log');


def ploteffics(losses, targeteffics):
    if type(losses[0].signaleffic) is list:
        for e in range(len(losses[0].signaleffic)):
            p = plt.plot([l.signaleffic[e].detach().numpy() for l in losses], '.', label=f"Effic: {100.*targeteffics[e]:3.1f}%")
            p = plt.plot([l.backgreffic[e].detach().numpy() for l in losses], '-', color=plt.gca().lines[-1].get_color(), label="BG effic")
    else:
        plt.plot([l.signaleffic.detach().numpy() for l in losses], '.', label=f"Effic: {100.*targeteffics[0]:3.1f}%")
        plt.plot([l.backgreffic.detach().numpy() for l in losses], '.', color=plt.gca().lines[-1].get_color(), label="BG effic")
    plt.legend()
    plt.xlabel('Training Epoch')
    plt.ylabel('Signal Efficiency')
    #plt.yscale('log')


def check_effic(x_test_tensor, y_test, net, printout=True):
    num_pass_test=0.
    num_bg_pass_test=0.
    test_outputs = net.apply_cuts(x_test_tensor).detach().cpu()
    m=test_outputs.shape[1]
    trues=torch.tensor(m*[True])
    for i in range(len(test_outputs)):
    
        tt=torch.zeros(m)
        t=torch.gt(test_outputs[i],tt)
    
        if torch.equal(t,trues) and y_test[i]==1.:
            num_pass_test+=1
        elif torch.equal(t,trues) and y_test[i]!=1.:
            num_bg_pass_test+=1.
        
    
    effic_test    = num_pass_test    / np.sum(y_test)
    bg_effic_test = num_bg_pass_test / np.sum(1.-y_test)

    if printout:
        print(f"Signal     efficiency with net outputs: {100*effic_test:4.1f}%")
        print(f"Background efficiency with net outputs: {100*bg_effic_test:8.5f}%")
    else:
        # do we want to return anything here?
        return effic_test,bg_effic_test


def plotcuts(net):
    fig = plt.figure(figsize=(20,5))
    fig.tight_layout()
    targeteffics=net.effics
    m=net.features
    
    scaled_cuts=[len(targeteffics)*[0] for i in range(m)]
    for n in range(len(targeteffics)):
        cuts=net.nets[n].get_cuts().detach().numpy()
        for f in range(m):
            cutval=cuts[f]
            scaled_cuts[f][n]=cutval
    for b in range(m):
        ax=fig.add_subplot(2,5,1+b)
        plt.subplots_adjust(hspace=0.6,wspace=0.5)
        ax.plot(targeteffics,scaled_cuts[b])
        ax.set_xlabel(f"Target Efficiency")
        ax.set_ylabel("Cut value")
        ax.set_title(f"Feature {b}")
        ax.set_ylim([-10,10])


def plotfeatures(net,x_signal,x_backgr,sc):
    # Distributions after scaling
    targeteffics=net.effics
    m=net.features
    for n in range(len(targeteffics)):
        print(f"Target efficiency: {targeteffics[n]*100}%")
        fig = plt.figure(figsize=(20,5))
        fig.tight_layout()
        nbins=50
        
        weights=net.nets[n].weight.detach().numpy()
        scaled_cuts=net.nets[n].get_cuts().detach().numpy()
        
        x_signal_scaled=sc.transform(x_signal)
        x_backgr_scaled=sc.transform(x_backgr)
        
        for b in range(m):
            ax=fig.add_subplot(2,5,1+b)
            plt.subplots_adjust(hspace=0.3,wspace=0.5)
            plt.yscale('log')
            ax.hist(x_signal_scaled[:,b],nbins,density=True,histtype='stepfilled',alpha=0.5,color='red')
            ax.hist(x_backgr_scaled[:,b],nbins,density=True,histtype='stepfilled',alpha=0.5,color='blue')
            ax.set_xlabel(f"Feature {b}")
            ax.set_ylabel("Events/Bin")
            if weights[b] < 0:
                ax.axvline(x = scaled_cuts[b], color='g') # cut is "less than"
            else:
                ax.axvline(x = scaled_cuts[b], color='y') # cut is "greater than"
        plt.show()
