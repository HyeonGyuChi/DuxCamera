import tkinter
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk


window=tkinter.Tk()

#기본 설정
window.title("Dux Camera")
window.geometry("1200x600+100+100")  #너비X높이+X좌표+Y좌표
window.resizable(True, True)        #사이즈 변경 가능

# 프레임
frame_control = tkinter.Frame(window, relief='solid', bd=2, bg = "WHITE", width=200)
frame_info = tkinter.Frame(window, relief='solid', bd=2, bg ="GRAY", width=1000)

frame_control.pack(side="right", fill="both", expand=True)
frame_info.pack(side="left", fill="both", expand=True)

# info_frmae
frame_info_1 = tkinter.Frame(frame_info, relief='solid', bd=2, bg ="GREEN", width=500, height=300)
frame_info_2 = tkinter.Frame(frame_info, relief='solid', bd=2, bg ="GRAY", width=500, height=300)
frame_info_3 = tkinter.Frame(frame_info, relief='solid', bd=2, bg ="GRAY", width=500, height=300)
frame_info_4 = tkinter.Frame(frame_info, relief='solid', bd=2, bg ="GRAY", width=500, height=300)
frame_info_1.grid(row=0, column=0)
frame_info_2.grid(row=0, column=1)
frame_info_3.grid(row=1, column=0)
frame_info_4.grid(row=1, column=1)

# frame_info_1
# Q1
frame_q1 = tkinter.Frame(frame_info_1, relief='solid', bd=2, bg ="red", width=250, height=150)
frame_q1.grid(row=0, column=1)

# Q2
frame_q2 = tkinter.Frame(frame_info_1, relief='solid', bd=2, bg ="red", width=250, height=150)
frame_q2.grid(row=0, column=0)

# Q3
frame_q3 = tkinter.Frame(frame_info_1, relief='solid', bd=2, bg ="red", width=250, height=150)
frame_q3.grid(row=1, column=0)

# Q4
frame_q4 = tkinter.Frame(frame_info_1, relief='solid', bd=2, bg ="red", width=250, height=150)
frame_q4.grid(row=1, column=1)

# label
label_q1=tkinter.Label(frame_q1, text="Q1 IMG", relief="solid")
label_q1.pack()

label_q2=tkinter.Label(frame_q2, text="Q2 IMG", relief="solid")
label_q2.pack()

label_q3=tkinter.Label(frame_q3, text="Q3 IMG", relief="solid")
label_q3.pack()

label_q4=tkinter.Label(frame_q4, text="Q4 IMG", relief="solid")
label_q4.pack()



#레이블
label_1=tkinter.Label(frame_control, text="위젯 테스트용입니다.", width=20, height=5, fg="red", relief="solid")
label_2=tkinter.Label(frame_control, text="", width=20, height=3, fg="red", relief="solid")

label_1.pack()
label_2.pack()

#버튼
button = tkinter.Button(frame_control, text="hi", overrelief="solid", width=20)
button.pack()


# 엔트리 함수
def calc(event):
    label.config(text="결과=" + str(eval(entry.get())))

#엔트리 입력창
entry=tkinter.Entry(frame_control)
entry.bind("<Return>", calc)
entry.pack()

#리스트박스
listbox=tkinter.Listbox(frame_control, selectmode='extended', height=0)
listbox.insert(0, "no1")
listbox.insert(1, "no2")
listbox.insert(2, "no3")
listbox.insert(3, "no4")
listbox.insert(4, "no5")
listbox.pack()

#체크박스 함수
def flash():
    button.flash()

#체크박스
CheckVariety_1=tkinter.IntVar()
CheckVariety_2=tkinter.IntVar()

checkbutton1=tkinter.Checkbutton(frame_control,text="O", variable=CheckVariety_1, activebackground="blue")
checkbutton2=tkinter.Checkbutton(frame_control, text="Y", variable=CheckVariety_2)
checkbutton3=tkinter.Checkbutton(frame_control, text="X", variable=CheckVariety_2, command=flash)

checkbutton1.pack()
checkbutton2.pack()
checkbutton3.pack()

#라디오버튼
RadioVariety_1=tkinter.IntVar()

#value값이 같을 경우 함께 선택됨
radio1=tkinter.Radiobutton(frame_control, text="1번", value=3, variable=RadioVariety_1)
radio1.pack()
radio2=tkinter.Radiobutton(frame_control, text="2번", value=3, variable=RadioVariety_1)
radio2.pack()
radio3=tkinter.Radiobutton(frame_control, text="3번", value=9, variable=RadioVariety_1)
radio3.pack()

#메뉴
def close():
    window.quit()
    window.destroy()

menubar=tkinter.Menu(window)

menu_1=tkinter.Menu(menubar, tearoff=0, selectcolor="red")
menu_1.add_command(label="Project")
menu_1.add_separator()
menu_1.add_command(label="종료", command=close)
menubar.add_cascade(label="New Project", menu=menu_1)

menu_2=tkinter.Menu(menubar, tearoff=0)
menu_2.add_command(label="Calc and Save")
menu_2.add_command(label="Load Calibration")
menubar.add_cascade(label="Calibration", menu=menu_2)

menu_3=tkinter.Menu(menubar, tearoff=0, selectcolor="red")
menu_3.add_command(label="Show Merged Image")
menubar.add_cascade(label="Show", menu=menu_3)

window.config(menu=menubar)

#배치(place(), pack(), grid())
b1=tkinter.Button(frame_control, text="top")
b1.pack(side="top")
b2=tkinter.Button(frame_control, text="bottom")
b2.pack(side="bottom")
b3=tkinter.Button(frame_control, text="left")
b3.pack(side="left")
b4=tkinter.Button(frame_control, text="right")
b4.pack(side="right")

bb1=tkinter.Button(frame_control, text="(50, 50)")
bb2=tkinter.Button(frame_control, text="(50, 100)")
bb3=tkinter.Button(frame_control, text="(100, 150)")
bb4=tkinter.Button(frame_control, text="(0, 200)")
bb5=tkinter.Button(frame_control, text="(0, 300)")
bb6=tkinter.Button(frame_control, text="(0, 300)")

bb1.place(x=50, y=50)
bb2.place(x=50, y=100, width=50, height=50)
bb3.place(x=100, y=150, bordermode="inside")
bb4.place(x=0, y=200, relwidth=0.5)
bb5.place(x=0, y=300, relx=0.5)
bb6.place(x=0, y=300, relx=0.5, anchor="s")

"""     grid() 는 pack()과 함께 쓰일 수 없음

bbb1=tkinter.Button(window, text="(0,0)")
bbb1.grid(row=0, column=0)
bbb2=tkinter.Button(window, text="(1,1)", width=20)
bbb2.grid(row=1, column=1, columnspan=3)
bbb3=tkinter.Button(window, text="(1,2)")
bbb3.grid(row=1, column=2)
bbb4=tkinter.Button(window, text="(2,3)")
bbb4.grid(row=2, column=3)
"""

array = np.ones((40,40)) * 100
print(array)
img =  ImageTk.PhotoImage(image=Image.fromarray(array))

canvas = tk.Canvas(frame_control,width=100,height=100)
canvas.pack()
canvas.create_image(20,20, anchor="nw", image=img)


window.mainloop()