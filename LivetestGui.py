
import PySimpleGUI as sg
from Livetest import * 
sg.theme('DarkAmber') 
layout = [[sg.Text("Live Mask Detection",justification='center',size=(100,1))], [sg.Button("LIVE")]]

# Create the window
window = sg.Window("Demo", layout,size=(300,100))
while True:
    
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event =="LIVE":
        tester=LiveTest()
        tester.Test()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

window.close()
