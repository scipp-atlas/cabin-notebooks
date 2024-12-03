import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

import torch
import torch.nn
import torch.optim
import torch.utils.data

import time
import math

from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F


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

class OneToOneLinear(torch.nn.Module):

    __constants__ = ['features']
    features: int
    weight: Tensor

    def __init__(self, 
                 features: int,
                 scalefactor=None, 
                 weights=None, 
                 postroot=None,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.features = features
        self.trainable_weights=(weights==None or len(weights)!=features)
        if self.trainable_weights:
            if weights==None:
                print("This network will learn weights and biases, since no weights were supplied.")
            elif len(weights)!=features:
                print(f"This network will learn weights and biases, since num weights ({len(weights)}) does not match num features ({features}).")
            self.weight = Parameter(torch.empty(features, **factory_kwargs))
        else:
            self.register_buffer('weight', torch.tensor(weights))
        self.bias = Parameter(torch.empty(features, **factory_kwargs))
        self.activation_scale_factor = scalefactor
        self.post_product_root = postroot if postroot != None else 1.0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound=1./math.sqrt(self.features)
        if self.trainable_weights:
            torch.nn.init.uniform_(self.weight, -bound, bound)
        torch.nn.init.zeros_(self.bias)

    def get_cuts(self) -> Tensor:
        return -self.bias/self.weight

    def apply_cuts(self, input: Tensor) -> Tensor:
        # need to turn the "weights" vector into a matrix with the vector 
        # elements on the diagonal, and zeroes everywhere else.
        targets = torch.matmul(input,torch.diag(self.weight))+self.bias
        return targets

    def forward(self, input: Tensor) -> Tensor:
        # apply the cuts
        targets = self.apply_cuts(input)

        # activation function
        targets = torch.sigmoid(self.activation_scale_factor*targets)
     
        # optionally take the root of the targets
        # had thought this might be a good way to reduce bias introduced
        # by taking the product of so many sigmoids, but this destroys
        # the good agreement we see between the "actual" efficiency
        # and the efficiency we calculate in the loss function.  
        # so, for now, make sure post_product_root remains 1.0.
        targets=targets**(1./self.post_product_root)

        # this is differentiable, unlike using torch.all(torch.gt()) or something else that yields booleans.
        # will converge to 1 for things that pass all cuts, and to zero for things that fail any single cut.
        targets=torch.prod(targets,dim=1)
        
        return targets

    def extra_repr(self) -> str:
        return f'in_features={self.features}, bias={self.bias}'


class lossvars():

    def __init__(self):
        self.efficloss = 0
        self.backgloss = 0
        self.cutszloss = 0
        self.monotloss = 0
        self.signaleffic = 0
        self.backgreffic = 0
    
    def totalloss(self):
        return self.efficloss + self.backgloss + self.cutszloss + self.monotloss

    def __add__(self,other):
        third=lossvars()
        third.efficloss = self.efficloss + other.efficloss
        third.backgloss = self.backgloss + other.backgloss
        third.cutszloss = self.cutszloss + other.cutszloss
        third.monotloss = self.monotloss + other.monotloss

        if type(self.signaleffic) is list:
            third.signaleffic = self.signaleffic
            third.signaleffic.append(other.signaleffic)
        else:
            third.signaleffic = []
            third.signaleffic.append(self.signaleffic)
            third.signaleffic.append(other.signaleffic)
        if type(self.backgreffic) is list:
            third.backgreffic = self.backgreffic
            third.backgreffic.append(other.backgreffic)
        else:
            third.backgreffic = []
            third.backgreffic.append(self.backgreffic)
            third.backgreffic.append(other.backgreffic)
        return third

def loss_fn (y_pred, y_true, features, net, 
             target_signal_efficiency=0.8,
             alpha=1., beta=1., gamma=0.001,
             debug=False):

    loss = lossvars()
    
    # signal efficiency: (selected events that are true signal) / (number of true signal)
    signal_results = y_pred * y_true
    loss.signaleffic = torch.sum(signal_results)/torch.sum(y_true)

    # background efficiency: (selected events that are true background) / (number of true background)
    background_results = y_pred * (1.-y_true)
    loss.backgreffic = torch.sum(background_results)/(torch.sum(1.-y_true))

    cuts=net.get_cuts()
    
    # * force signal efficiency to converge to a target value
    # * force background efficiency to small values at target efficiency value.
    # * also prefer to have the cuts be close to zero, so they're not off at some crazy 
    #   value even if we prefer for the cut to not have much impact on the efficiency 
    #   or rejection.
    #
    # should modify the efficiency target requirement here, to make this more 
    # like consistency with e.g. a gaussian distribution rather than just a penalty 
    # calculated from r^2 distance.
    #
    # for both we should prefer to do something like "sum(square())" or something.
    loss.efficloss = alpha*torch.square(target_signal_efficiency-loss.signaleffic)
    loss.backgloss = beta*loss.backgreffic
    loss.cutszloss = gamma*torch.sum(torch.square(cuts))/features

    if debug:
        print(f"Inspecting efficiency loss: alpha={alpha}, target={target_signal_efficiency:4.3f}, subnet_effic={loss.signaleffic:5.4f}, subnet_backg={loss.backgreffic:5.4f}, efficloss={loss.efficloss:4.3e}, backgloss={loss.backgloss:4.3e}")
    
    # sanity check in case we ever need it, should work
    #loss=bce_loss_fn(outputs_to_labels(y_pred,features),y_true)
    
    return loss


class EfficiencyScanNetwork(torch.nn.Module):
    def __init__(self,features,effics,weights=None,activationscale=2.,postroot=1.):
        super().__init__()
        self.features = features
        self.effics = effics
        self.weights = weights
        self.activation_scale_factor=activationscale
        self.post_product_root=postroot
        self.nets = torch.nn.ModuleList([OneToOneLinear(features, self.activation_scale_factor, self.weights, self.post_product_root) for i in range(len(self.effics))])

    def forward(self, x):
        outputs=torch.stack(tuple(self.nets[i](x) for i in range(len(self.effics))))
        return outputs

    def to(self, device):
        super().to(device)
        for n in self.nets:
            n.to(device)


def effic_loss_fn(y_pred, y_true, features, net,
                  alpha=1., beta=1., gamma=0.001, epsilon=0.001,
                  debug=False):

    # probably a better way to do this, but works for now
    sumefficlosses=None    
    for i in range(len(net.effics)):
        effic=net.effics[i]
        efficnet = net.nets[i]
        l=loss_fn(y_pred[i], y_true, features, 
                  efficnet, effic,
                  alpha, beta, gamma, debug)
        if sumefficlosses==None:
            sumefficlosses=l
        else:
            #sumefficlosses=torch.add(sumefficlosses,l)
            sumefficlosses = sumefficlosses + l

    loss=sumefficlosses
    # now set up global penalty for cuts that vary net by net.
    # some options:
    # a. penalize a large range of cut values
    # b. penalize large changes between nearest neighbors
    # c. test for non-monotonicity?
    #
    # go for b for now.
    #

    # For a fancier way to force monotonic behavior, see e.g. 
    # https://pypi.org/project/monotonicnetworks/
    #
    # Note that this also has issues since sortedeffics won't necessarily have the same
    # index mapping as 'nets'....  so lots of potential problems here.
    #
    sortedeffics=sorted(net.effics)

    if len(sortedeffics)>=3:
        featureloss = None
        for i in range(1,len(sortedeffics)-1):
            cuts_i   = net.nets[i  ].get_cuts()
            cuts_im1 = net.nets[i-1].get_cuts()
            cuts_ip1 = net.nets[i+1].get_cuts()

            # calculate distance between cuts.  
            # would be better to implement this as some kind of distance away from the region 
            # between the two other cuts.
            #
            # maybe some kind of dot product?  think about Ising model.
            #
            # maybe we just do this for the full set of biases, to see how many transitions there are?  no need for a loop?
            #
            # otherwise just implement as a switch that calculates a distance if outside of the range of the two cuts, zero otherwise
            fl = None

            # ------------------------------------------------------------------
            # This method just forces cut i to be in between cut i+1 and cut i-1. 
            #
            # add some small term so that when cutrange=0 the loss doesn't become undefined  
            cutrange           =  cuts_ip1-cuts_im1
            mean               = (cuts_ip1+cuts_im1)/2.
            distance_from_mean = (cuts_i  -mean)
            
            # add some offset to denominator to avoid case where cutrange=0.
            # playing with the exponent doesn't change behavior much.
            # it's important that this term not become too large, otherwise
            # the training won't converge.  just a modest penalty for moving
            # away from the mean should do the trick.
            exponent=2.  # if this changes, e.g. to 4, then epsilon will also need to increase
            fl=(distance_from_mean**exponent)/((cutrange**exponent)+0.1)
            # ------------------------------------------------------------------
            
            # ------------------------------------------------------------------
            ## can also do it this way, which just forces all sequential cuts to be similar.
            #fl = torch.pow(cuts_i-cuts_im1,2) + torch.pow(cuts_i-cuts_ip1,2) + torch.pow(cuts_im1-cuts_ip1,2)
            # ------------------------------------------------------------------
          
            if featureloss == None:
                featureloss = fl
            else:
                featureloss = featureloss + fl

        # need to sum all the contributions to this component of the loss from the different features.
        sumfeaturelosses = torch.sum(featureloss)/(len(sortedeffics)-2)/features
        loss.monotloss = epsilon*sumfeaturelosses

    return loss

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