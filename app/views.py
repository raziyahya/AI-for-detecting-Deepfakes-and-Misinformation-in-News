import subprocess
from django.core.files.storage import FileSystemStorage
from django.http.response import HttpResponse
from django.shortcuts import render
from werkzeug.utils import redirect

from .models import *
from .predict_fn import predictfn
# from .predict_vidio_fn import predictvideofn

from .audio import predict_audio_code
import os
# Create your views here.

from .sampleee import checknews
def main(request):


    return render(request,"index.html")
def login(request):
    uname = request.POST['uname']
    pwd = request.POST['pwd']

    try:
        ob=Login.objects.get(Username=uname,Password=pwd)
        return HttpResponse('''<script>;window.location='/userhome'</script> ''')

    except:
        return HttpResponse('''<script>;window.location='/'</script> ''')
    return render(request,"index.html")


def signup(request):
    return render(request,"registration.html")


def userhome(request):
    return render(request,"user/uhome.html")


def predict_image(request):
    return render(request,"user/image.html")

def predict_video(request):
    return render(request,"user/video.html")

def predict_audio(request):
    return render(request,"user/audio.html")

def predict_text(request):
    return render(request,"user/text.html")

def predict_text_post(request):
    import subprocess
    txt=request.POST['txt'].replace("\n"," ")+"\n====\n"
    print("*"*50)
    print(len(txt))
    print(txt)
    file_path = r'C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\example_txt.txt'

    # Open the file in write mode ('w'), this will clear the file if it exists
    with open(file_path, 'w') as file:
        file.write(str(txt))
    # Run a command or script with input data
    result = subprocess.run(
        [r'C:\Users\Asus\AppData\Local\Programs\Python\Python310\python.exe',
         r'C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\app\textprediction.py'],  # Replace with your script or command
        input=txt.encode('utf-8')
        # The input to be provided to the subprocess
        # Capture the error (if any)

    )
    print("Output:", result.stdout)
    if result.stderr:
        print("Error:", result.stderr)
    print(result)
    print("====================================")
    print("====================================")
    print("====================================")
    with open(r'C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\example.txt', 'r') as file:
        content = file.read()  # Reads the entire file

    p, r = content.split("#")
    p=float(p)*100
    print(p,r)
    res="Real"
    if str(r)=="1":

        res="AI GENERATED"
    else:
        try:
            print ("txt")
            print (txt.split("\n"))
            print ("+=+=+=+=+=+=+=+=+=")
            res = checknews(txt)
        except Exception as e:
            print(e,"==========================")
            pass
        p=100-p
        if p<50:
            p+=50
    import math
    #
    num = p
    truncated_num = math.trunc(num * 100) / 100
    print(truncated_num)  # Output: 3.14

    return render(request,"user/text.html",{"s":"1","p":truncated_num,"r":res,"t":txt})

def predict_imagefn(request):
    if 'file' in request.FILES:

        f=request.FILES['file']

        fs=FileSystemStorage()
        fn=fs.save(f.name,f)
        res,p=predictfn(os.path.join(r"C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\media",fn))
        print(res,"res===========")
        print(res,"res===========")
        print(res,"res===========")
        print(res,"res===========")

        if str(res) == "1":

            res="REAL"
        else:

            res="FAKE"
        return render(request,"user/image.html",{"r":res,"i":fn,"s":True,"p":p})
    else:
        return redirect("/predict_image")

def predict_videofn(request):
    if 'file' in request.FILES:

        f=request.FILES['file']

        fs=FileSystemStorage()
        fn=fs.save(f.name,f)
        # res,per=predictvideofn(os.path.join(r"D:\germany\Deep Fake\Deepfake\media",fn))
        response = subprocess.run(
            [r'C:\Users\Asus\AppData\Local\Programs\Python\Python36\python.exe',
             r'C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\app\predict_vidio_fn.py'],  # Replace with your script or command
            input=os.path.join(r"C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\media",fn).encode('utf-8')
            # The input to be provided to the subprocess
            # Capture the error (if any)

        )
        print("Output:", response)
        print(response)

        print("====================================")
        print("====================================")
        print("====================================")
        with open(r'C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\example.txt', 'r') as file:
            content = file.read()  # Reads the entire file

        res,per = content.split("#")
        print(res,"res===========")
        if str(res) == "1":
            res="REAL"
        else:
            res="FAKE"
        return render(request,"user/video.html",{"r":res,"i":fn,"s":True,"p":per})
    else:
        return redirect("/predict_video")


def predict_audiofn(request):
    if 'file' in request.FILES:

        f=request.FILES['file']

        fs=FileSystemStorage()
        fn=fs.save(f.name,f)
        res,ac=predict_audio_code(os.path.join(r"C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\media",fn))
        print(res,"res===========")
        print(res,"res===========")
        print(res,"res===========")
        print(res,"res===========")
        if str(res) == "1":
            res="REAL"
        else:
            res="FAKE"
        return render(request,"user/audio.html",{"r":res,"i":fn,"s":True,"a":ac})
    else:
        return redirect("/predict_audio")


def reg_code(request):
    name=request.POST['name']
    dob=request.POST['DOB']
    email=request.POST['email']
    gen=request.POST['gen']
    phno=request.POST['phno']
    uname=request.POST['uname']
    pwd=request.POST['pwd']
    file=request.FILES['file']

    obl=Login()
    obl.Username=uname
    obl.Password=pwd
    obl.Type="user"
    obl.save()

    obu=User()
    obu.LOGIN=obl
    obu.Name=name
    obu.DOB=dob
    obu.Gender=gen
    obu.Email=email
    obu.Phoneno=phno
    obu.Image=file
    obu.save()


    return HttpResponse('''<script>alert('Succes');window.location='/'</script> ''')