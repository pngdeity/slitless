# 2023-01-18
# Ulas Kamaci

def outch_adjuster(out=None, true_out=None, outch_type=None, action=None):
    if action=='crop':
        if outch_type == 'int':
            y = true_out[:,[0]]
        elif outch_type == 'vel':
            y = true_out[:,[1]]
        elif outch_type == 'width':
            y = true_out[:,[2]]
        elif outch_type == 'all':
            y = true_out
    elif action=='extend':
        if outch_type == 'int':
            y = true_out[:].clone() if true_out is not None else out[:].clone()
            y[:,0] = out.squeeze()
        elif outch_type == 'vel':
            y = true_out[:].clone() if true_out is not None else out[:].clone()
            y[:,1] = out.squeeze()
        elif outch_type == 'width':
            y = true_out[:].clone() if true_out is not None else out[:].clone()
            y[:,2] = out.squeeze()
        elif outch_type == 'all':
            y = out

    return y