from torch.utils.tensorboard import SummaryWriter
from configs import Tensorboard

#loss(d,g)
#acc(real_label) train valid
#best acc
#generate_img
log_dir = Tensorboard.log_dir
writer = SummaryWriter()

