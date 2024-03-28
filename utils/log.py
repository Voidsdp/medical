from colorama import Fore  
from torch.utils.tensorboard import SummaryWriter
       
from configs import Tensorboard

log_dir = Tensorboard.log_dir
writer = SummaryWriter(log_dir)

def log_scalar(title,value,epoch=None,color='white',cmd=True,tensorboard=False):
    if cmd:
       if epoch is None:
          color_print('{} : {}'.format(title,value),color)
       else: 
          color_print('Epoch {} {} : {}'.format(epoch,title,value),color)
       
       writer.add_scalar(title,value) 
       
    if tensorboard:
       if epoch is None:
          writer.add_scalar(title,value) 
       else: 
          writer.add_scalar(title,value,epoch) 


def args_print(args,length=48,color='white'):  
    with ColorPrint(color):
        banner_print('arguments',length,linestyle='.')
        
        str_list = []       
        for arg in vars(args): 
            str_list.append(arg)
        for arg in sorted(str_list,key=lambda x: x.lower()):
            directory_print(arg,getattr(args,arg),length,'-')

        banner_print('arguments',length,linestyle='.')
    

def banner_print(value,length=48,linestyle='-'):
    banner = linestyle * int((length / 2)) +  str(value) + linestyle * int((length/2))
    print(banner,flush=True)


def directory_print(key,value,length=48,linestyle='-'):
    dots = linestyle * (length-len(key))
    directory = '{}{}{}'.format(key,dots,value)
    print(directory,flush=True)


def color_print(value,color='white',end='\n'):
    color = vars(Fore)[color.upper()]
    print(color+str(value)+Fore.RESET,end=end,flush=True)


def color_str(value,color='white'):
    color = vars(Fore)[color.upper()]
    return color + value + Fore.RESET


class ColorPrint:
      """
      usage:
      color: red, yellow, green, cyan, blue, magenta, white, black ...

      with ColorPrint('color'):
           print('your sentence.')

      or

      ColorPrint.start('color1')
      ColorPrint.start('color2')
      print('your sentence.')
      ColorPrint.end()

      """
      def __init__(self,color):
          self.color = vars(Fore)[color.upper()]
         
      @staticmethod
      def start(color):
          print(vars(Fore)[color.upper()])
    
      @staticmethod
      def end():
          print(Fore.RESET)

      def __enter__(self):
          print(self.color,end='')  #avoid \n
      
      def __exit__(self, exc_type, exc_val, exc_tb):
          print(Fore.RESET,end='')
       
       
      

