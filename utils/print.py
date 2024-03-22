from colorama import Fore     

def parse_args(parser,check_fun=lambda x: x):
    parser = check_fun(parser)
    return parser


def args_print(args,length=48,color='white'):  
    with ColorPrint(color):
        banner_print('arguments',length,linestyle='.')
        
        str_list = []       
        for arg in vars(args): 
            str_list.append(arg)
        for arg in sorted(str_list,key=lambda x: x.lower()):
            directory_print(arg,getattr(args,arg),length,'-')

        banner_print('arguments',length,linestyle='.')
    

def acc_print(acc,color='white',hightlight=False):
    real_label_acc = acc

    color = 'red' if hightlight else color
    color_print('real_label_acc: {:.2f}%'.format(real_label_acc), color,end=' ')


def banner_print(value,length=48,linestyle='-'):
    # length = int(length)
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
     
