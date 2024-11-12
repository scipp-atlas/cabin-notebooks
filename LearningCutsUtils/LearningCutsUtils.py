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
                 device=None, 
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.features = features
        #print(weights)
        self.trainable_weights=(weights==None or len(weights)!=features)
        if self.trainable_weights:
            if weights==None:
                print("This network will learn weights and biases, since no weights were supplied.")
            elif len(weights)!=features:
                print(f"This network will learn weights and biases, since num weights ({len(weights)}) does not match num features ({features}).")
            self.weight = Parameter(torch.empty(features, **factory_kwargs))
        else:
            self.weight = torch.tensor(weights)
        self.bias = Parameter(torch.empty(features, **factory_kwargs))
        self.activation_scale_factor = scalefactor
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound=1./math.sqrt(self.features)
        if self.trainable_weights:
            torch.nn.init.uniform_(self.weight, -bound, bound)
        #torch.nn.init.uniform_(self.bias, -bound, bound)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:

        # need to turn the "weights" vector into a matrix with the vector 
        # elements on the diagonal, and zeroes everywhere else.
        targets = torch.matmul(input,torch.diag(self.weight))+self.bias
        targets = torch.sigmoid(self.activation_scale_factor*targets)
        return targets

    def extra_repr(self) -> str:
        return f'in_features={self.features}, bias={self.bias}'


# net_outputs in this case is a list of output scores for each event, one score per input features.
# need to try to push these towards 1 for signal, and 0 for background.
def outputs_to_labels(net_outputs, features):
    # an infinity norm.  this gives good efficiency, but basically gives up on a number
    # of cuts for reasons that are unclear to me, setting cut values way below or above
    # both signal and background distributions.  Can use either min or max, give very
    # similar results.
    #
    # targets = torch.min(torch.abs(net_outputs),1,keepdim=True).values.squeeze(1)

    # do the silly thing and sum over all the outputs.  each output weighted the same,
    # misclassification contributes to a larger loss value.
    # This gives an excellent summary score that has near perfect discrimination,
    # but does a terrible job of creating a set of cuts that has good efficiency, at
    # least when combined with a BCE loss function.  Each individual cut is snug up
    # against where signal and background distributions cross, almost perfectly.  it
    # looks pretty, but not useful combined with BCE.
    #
    targets = torch.sum(net_outputs,1)/features
    
    # there's no variation on this that seems to work well. 
    # loss function is capped within a range less than [0,1], or get errors.
    #
    # targets = torch.sqrt(torch.sum(torch.square(net_outputs,1))/features

    # this doesn't seem to work at all.  not sure why.
    #print(net_outputs[0])
    #print(torch.gt(net_outputs[0],0.5))
    #print(torch.all(torch.gt(net_outputs[0],0.5)))
    # targets = torch.autograd.Variable(torch.all(torch.gt(net_outputs,0.5),dim=1).float(), requires_grad=True)
    
    #print(targets)
    return targets

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
    
    # this is differentiable, unlike using torch.all(torch.gt()) or something else that yields booleans.
    # will converge to 1 for things that pass all cuts, and to zero for things that fail any single cut.
    all_results=torch.prod(y_pred,dim=1)

    # rescale the signal and background efficiencies to take into account the fact that all weights will be <1.
    sum_of_weights=torch.sum(all_results)
    scale_factor=len(y_pred)/sum_of_weights
    # is this right?  pick up here.
    
    # signal efficiency: (selected events that are true signal) / (number of true signal)
    signal_results = all_results * y_true
    loss.signaleffic = torch.sum(signal_results)/torch.sum(y_true)

    # background efficiency: (selected events that are true background) / (number of true background)
    background_results = all_results * (1.-y_true)
    loss.backgreffic = torch.sum(background_results)/(torch.sum(1.-y_true))

    cuts=-net.bias/net.weight
    
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
        print(f"Inspecting efficiency loss: alpha={alpha}, target={target_signal_efficiency}, subnet_effic={loss.signaleffic}, efficloss={loss.efficloss}")
    
    # sanity check in case we ever need it, should work
    #loss=bce_loss_fn(outputs_to_labels(y_pred,features),y_true)
    
    return loss


class EfficiencyScanNetwork(torch.nn.Module):
    def __init__(self,features,effics,weights=None,activationscale=2.):
        super().__init__()
        self.features = features
        self.effics = effics
        self.weights = weights
        self.activation_scale_factor=activationscale
        self.nets = torch.nn.ModuleList([OneToOneLinear(features, self.activation_scale_factor, self.weights) for i in range(len(self.effics))])

    def forward(self, x):
        outputs=torch.stack(tuple(self.nets[i](x) for i in range(len(self.effics))))
        return outputs


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

    #
    # The below implementation isn't quite right, and probably isn't differentiable.
    # Need to find some way to constrain the variation between adjacent cuts in
    # a differentiable way, similar to how the torch.all(torch.gt()) thing was 
    # approximated by torch.prod().
    #
    # Could do something like taking the mean of the cut values, subtracting that off,
    # and...  checking something?
    #
    # see e.g. https://pypi.org/project/monotonicnetworks/ for a more complicated treatment
    #
    # Note that this also has issues since sortedeffics won't necessarily have the same
    # index mapping as 'nets'....  so lots of potential problems here.
    #
    #
    sortedeffics=sorted(net.effics)

    if len(sortedeffics)>=3:
        def getcuts(subnet):
            return -subnet.bias/subnet.weight
        featureloss = None
        for i in range(1,len(sortedeffics)-1):
            cuts_i   = getcuts(net.nets[i  ])
            cuts_im1 = getcuts(net.nets[i-1])
            cuts_ip1 = getcuts(net.nets[i+1])

            # calculate distance between cuts.  
            fl = torch.pow(cuts_i-cuts_im1,2) + torch.pow(cuts_i-cuts_ip1,2) + torch.pow(cuts_im1-cuts_ip1,2)

            # don't think we need this anymore, if we're fixing the lt vs gt interpretation of cuts by fixing weights
            # when initializing the efficiency scan network:
            #
            ## also need some term that penalizes weights that change sign.  going from positive to negative
            ## weights will change the interpretation from "less than" to "greater than" or vice versa.
            #fl = fl + \
            #(features - torch.sum(torch.tanh(2*net.nets[i].weight)*torch.tanh(net.nets[i-1].weight))) + \
            #(features - torch.sum(torch.tanh(2*net.nets[i].weight)*torch.tanh(net.nets[i+1].weight)))
            
            if featureloss == None:
                featureloss = fl
            else:
                featureloss = featureloss + fl
        sumfeaturelosses = torch.sum(featureloss)/(3.*(len(sortedeffics)-2))/features
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
        plt.plot([l.monotloss.detach().numpy() for l in losses], '.', label="Train: smooth")
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
    test_outputs = net(x_test_tensor).detach().cpu()
    m=test_outputs.shape[1]
    trues=torch.tensor(m*[True])
    for i in range(len(test_outputs)):
    
        tt=torch.zeros(m)+0.5
        t=torch.gt(test_outputs[i],tt)
    
        if torch.equal(t,trues) and y_test[i]==1.:
            num_pass_test+=1
        elif torch.equal(t,trues) and y_test[i]!=1.:
            num_bg_pass_test+=1.
        
    
    effic_test    = num_pass_test    / np.sum(y_test)
    bg_effic_test = num_bg_pass_test / np.sum(1.-y_test)

    if printout:
        print(f"Signal Efficiency with net outputs: {100*effic_test:4.1f}%")
        print(f"Background Efficiency with net outputs: {100*bg_effic_test:6.5f}%")
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
        biases=net.nets[n].bias.detach().numpy()
        weights=net.nets[n].weight.detach().numpy()
        cuts=(-biases/weights)
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
        
        biases=net.nets[n].bias.detach().numpy()
        weights=net.nets[n].weight.detach().numpy()
        scaled_cuts=-biases/weights
        #print(f"Cuts are: {scaled_cuts}")
        
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