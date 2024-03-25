class model:
      noise_dim = 100 
      num_class = 7

      checkpoint = 'checkpoint'


class optimizer:
    optimizer_type = 'AdamW'
    betas = [0.5,0.999]


class Tensorboard:
      log_dir = './tensorboard'


class data:
    label_file = 'label.json'
    mean = (0.5,0.5,0.5)
    std = (0.5,0.5,0.5)
    img_size = 224