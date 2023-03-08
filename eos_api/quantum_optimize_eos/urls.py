from quantum_optimize_eos.views import *
from django.urls import path

urlpatterns = [
    path('quantumOptimizeEOS', quantum_optimize_eos_schedule),
    path('euspaOptimizeEOS', euspa_optimize_eos_schedule),

]