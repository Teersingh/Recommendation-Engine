from django.shortcuts import render
import app1
# Create your views here.
def index(request):
          
          return render(request,'index.html')
def result(request):
     try:
          data=request.GET['search2']
          
          search=app1.recommendations(data.title())
          
          fetch={'result':search,}

     
     except:
          pass
     return render(request, 'result.html',fetch)
