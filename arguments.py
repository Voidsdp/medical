from colorama import Fore     

#参数检查
def parse_args(parser,check_fun=lambda x: x):
     #检查参数assert与默认调整
    parser = check_fun(parser)
    return parser


#打印参数
def print_args(args,length=48,color='white'):  
    with ColorPrint(color):
        banner_print('arguments',length,linestyle='.')
        
        str_list = []       
        for arg in vars(args):  #对key排序
            str_list.append(arg)
        for arg in sorted(str_list,key=lambda x: x.lower()):
            directory_print(arg,getattr(args,arg),length,'-')

        banner_print('arguments',length,linestyle='.')
    
#横幅打印
def banner_print(value,length=48,linestyle='-'):
    banner = linestyle*int((length/2)) +  str(value) + linestyle * int((length/2))
    print(banner,flush=True)

#目录打印
def directory_print(key,value,length=48,linestyle='-'):
    dots = linestyle * (length-len(key))
    directory = '{}{}{}'.format(key,dots,value)
    print(directory,flush=True)

#彩色打印函数
def color_print(value,color):
    color = vars(Fore)[color.upper()]
    print(color+str(value)+Fore.RESET)

#彩色打印开关
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

      warnings:maybe leave blank space!

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
          print(self.color)
      
      def __exit__(self, exc_type, exc_val, exc_tb):
          print(Fore.RESET)
     

if __name__ == '__main__':  
   colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'white', 'black']
   banner_print('test-color_print')
   for color in colors:
       with ColorPrint(color):
           print(color)
    

#   banner_print('case1')
#   for color in colors:
#       ColorPrint.start(color)
#       print(color)
#       ColorPrint.end()

   banner_print('test-print_args')
   
   class A:
     def __init__(self) -> None:
         pass  
   args = A()
   
   for i in range(100):
       setattr(args,str(i),i)
   print_args(args,48,color='green')
   


