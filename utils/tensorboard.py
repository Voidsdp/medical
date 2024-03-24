from torch.utils.tensorboard import SummaryWriter
from configs import Tensorboard

log_dir = Tensorboard.log_dir
writer = SummaryWriter(log_dir)

