from tkinter import *  # 导入 Tkinter 库
from tkinter import messagebox
from myCNN import myCNN
root = Tk()  # 创建窗口对象的背景色
root.geometry("400x400")
def train():
    myCNN.train()
def test():
    myCNN.test()
def fuck():
    myCNN.fuck()

B1 = Button(root, text="训练1000次", command=train)
B2 = Button(root,text="测试",command=test)
B3 = Button(root,text="FUCK",command=fuck)
B1.pack()
B2.pack()
B3.pack()
root.mainloop()  # 进入消息循环
