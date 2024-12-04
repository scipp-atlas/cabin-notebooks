import torch


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
