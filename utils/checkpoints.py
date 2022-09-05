import torch

def save_checkpoint(state,filename='my_checkpoint.pth.tar'):
    torch.save(state,filename)
    
# FUNCTION TO LOAD CHEKCPOINT
def load_checkpoint(model,optimizer,PathL_ModelTrained):
    checkpoint = torch.load(PathL_ModelTrained)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

