import ROOT
from ROOT import TMVA
import os
import sys
import uproot

# Using LCU functions; load_random_data & load_SUSY_data
import LearningCutsUtils.Utils as LCU

# decide which dataset to run
datatag='random'
#datatag='SUSY'

try:
   os.mkdir(f"tmva/{datatag}")
except FileExistsError as err:
   pass

# =========================================================================================================================
# Global setup

TMVA.Tools.Instance()  # Initialize the TMVA environment
TMVA.PyMethodBase.PyInitialize()  # Initialize the Python interface for TMVA

output = ROOT.TFile.Open(f'tmva/{datatag}/TMVA_Output.root', 'RECREATE')  # Create output file
factory = TMVA.Factory('TMVACuts', output, '!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification')  # Create TMVA factory for cuts classification
# =========================================================================================================================


# =========================================================================================================================
# Data preparation

dataloader = TMVA.DataLoader(f"tmva/{datatag}")  # Create a data loader object

def write_ttrees(datatag="random"):
    x_signal, y_signal, x_backgr, y_backgr, branchFeatures = LCU.load_random_data() if datatag=="random" else LCU.load_SUSY_data()

    branches=x_signal.shape[1]
    sig_branch_dict = {}
    bkg_branch_dict = {}
    for i in range(branches):
        sig_branch_dict[branchFeatures[i].replace(' ','_')] = x_signal[:,i]
        bkg_branch_dict[branchFeatures[i].replace(' ','_')] = x_backgr[:,i]

    with uproot.recreate(f"tmva/{datatag}/data.root") as f:
        f["treeS"] = sig_branch_dict
        f["treeB"] = bkg_branch_dict

if not os.path.exists(f"tmva/{datatag}/data.root"):
    write_ttrees(datatag)

# Load your signal and background trees:
data_file = ROOT.TFile(f'tmva/{datatag}/data.root',"RO")
signal_tree = data_file.Get('treeS')  # Replace with your signal tree
background_tree = data_file.Get('treeB')  # Replace with your background tree

# Assuming you have a ROOT file named 'data.root' with TTree 'tree' containing signal and background data
# Add your discriminating variables:
for i in range(signal_tree.GetNbranches()): # extract this number automatically somehow
    print(signal_tree.GetListOfBranches()[i].GetName())
    dataloader.AddVariable(signal_tree.GetListOfBranches()[i].GetName())

dataloader.AddSignalTree(signal_tree, 1.0)  # Add signal tree with weight 1.0
dataloader.AddBackgroundTree(background_tree, 1.0)  # Add background tree with weight 1.0

dataloader.PrepareTrainingAndTestTree(
    ROOT.TCut(''),  # You can apply cuts here if needed
    f'nTrain_Signal={int(0.8*signal_tree.GetEntriesFast())}:nTrain_Background={int(0.8*background_tree.GetEntriesFast())}:SplitMode=Random:NormMode=NumEvents:!V'
)  # Split data for training and testing
# =========================================================================================================================


# =========================================================================================================================
# Options for Cuts optimization

# -------------------------------------------------------------------------------------------------------------------------
# MC sampling is fast.  Marginally worse for smoothness of cuts, but gives crazy (bad) ROC curves.  Do not use, except
# for debugging purposes.
SampleSize=100000
MCoptions = f'!H:!V:FitMethod=MC:EffSel:VarProp=FSmart:SampleSize={SampleSize}'

factory.BookMethod(
    dataloader,
    TMVA.Types.kCuts,  # Specify the Cuts method
    'Cuts_MC',  # Method name
    MCoptions
) 
# -------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------------
# Earl does popsize=500, 20 steps, and 6 cycles.  
Popsize=500
Steps=20
Cycles=6
GAoptions_Earl = f'!H:!V:FitMethod=GA:EffSel:VarProp=FSmart:CreateMVAPdfs:Popsize={Popsize}:Steps={Steps}:Cycles={Cycles}' 

factory.BookMethod(
    dataloader,
    TMVA.Types.kCuts,  # Specify the Cuts method
    'Cuts_GA_Earl',  # Method name
    GAoptions_Earl
) 
# -------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------------
# Training time scales linearly with product of steps, cycles, and popsize.
# So (200,10,5) is 6 times faster than (500,20,6).  No obvious difference in performance for gaussian dataset,
# results are still super choppy.
Popsize=200
Steps=10
Cycles=5
GAoptions_Fast = f'!H:!V:FitMethod=GA:EffSel:VarProp=FSmart:CreateMVAPdfs:Popsize={Popsize}:Steps={Steps}:Cycles={Cycles}' 

factory.BookMethod(
    dataloader,
    TMVA.Types.kCuts,  # Specify the Cuts method
    'Cuts_GA_Fast',  # Method name
    GAoptions_Fast
) 
# -------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================

# =========================================================================================================================
# Run it all

factory.TrainAllMethods()  # Train all booked methods
factory.TestAllMethods()  # Test all trained methods
factory.EvaluateAllMethods()  # Evaluate the performance of trained methods

output.Close()  # Close the output file
# =========================================================================================================================
