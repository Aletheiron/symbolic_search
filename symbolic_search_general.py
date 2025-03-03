import numpy as np
from numpy import random
import matplotlib.pyplot as plt


#Choose data for analysis


x1=np.arange(0.01, 100.01, 0.01)

x2=np.arange(0.01, 200, 0.02)

x=np.vstack([x1,x2])

'''Choosing function for generating data: polynomial and cyclic'''


y=125+23*x[0]-345*x[1]+20*x[0]*x[1]-40*x[0]**2+40*x[1]**2


#y=15*np.sin(3*x[0])+5*np.cos(5*x[1])



#y=15*np.sin(3*x[0])+5*np.cos(5*x[1])+20*x[0]

#y=15*np.sin(3*x[0])+5*np.cos(5*x[1])+20*x[0]-2*x[1]**2


residual=x


#Find row
k=len(x[:,0]) #amount of rows



class Add_a_function ():
    
    def __init__(self):
        
        '''random choice could be done by different rules'''
        
        #self.a=random.randint(-10,10) 
        self.a=random.randn(1)
        
    
    def _params(self):
        
        return self.a
    
    def forward (self, x):
        
        return x+self.a
    
    def gen_with_params (self, x, params):
        
        return x+params


class Product_multi_function():
    
    def __init__(self):
        
        '''random choice could be done by different rules'''
        
        #self.multi=random.randint(-10,10) 
        self.multi=random.randn(1)*5
        #self.multi=random.choice([-2,-1.75 -1.5,-1,-0.5,-0.25, 0, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2])
    
    def _params(self):
        
        return self.multi
    
    def forward (self, x):
        
        return x*self.multi
    
    def gen_with_params (self, x, params):
        
        return x*params


class Scale_function():
    
    def __init__(self):
        
        '''random choice could be done by different rules'''
        
        #self.scale_factor=random.randint(-10,10)
        #self.scale_factor=random.randn(1)*10
        self.scale_factor=random.choice([-2,-1.75 -1.5,-1,-0.5,-0.25, 0, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2])
        #self.scale_factor=random.choice([3, 2, 4,-1])
    
    def _params(self):
        
        return self.scale_factor
    
    def forward (self, x):
        
        #some troubles with scaling in numpy fot negative numbers. This is not real scaling function, but still some function
        return np.sign(x)*(np.abs(x))**self.scale_factor 
    
    def gen_with_params (self, x, params):
        
        return np.sign(x)*(np.abs(x))**params


class Sin_function():
    
    def __init__(self):
        
        '''random choice could be done by different rules'''
        
        self.theta=random.randint(-10,10)
    
    def _params(self):
        
        return self.theta
    
    def forward (self, x):
        
        return np.sin(x)
    
    def gen_with_params (self, x, params):
        
        return np.sin(x)


class Variable_add_function():
    
    #we add a variable from data to previous computings
    
    def __init__(self):
        
        self.raw_number=random.choice(list(i for i in range(k))) #choice among given rows of initial data
    
    def _params(self):
        
        return self.raw_number
    
    def forward (self, x):
        
        return x+residual[self.raw_number]
    
    def gen_with_params (self, x, params):
        
        return x+residual[params]


class Variable_multi_function():
    
    #we multiply a variable from data by previous computings
    
    def __init__(self):
        
        self.raw_number=random.choice(list(i for i in range(k))) #choice among given rows of initial data
        
    
    def _params(self):
        
        return self.raw_number
    
    def forward (self, x):
        
        return x*residual[self.raw_number]
    
    def gen_with_params (self, x, params):
        
        return x*residual[params]
    



    
#List of functions for generate one of them

func_list=[Add_a_function, Product_multi_function, Scale_function, Sin_function, Variable_add_function, Variable_multi_function]


#MSE Function of Joy. Utility function can be anything in general

def mse_joy (y, y_pred):
    
    mse_j = -1*(np.mean((y - y_pred)**2))
    
    return mse_j

#Hyperparametertes
EPOCH=100

#Utiity Function global list
UFg=[]


#List of applied functions
func_applied_list=[]
param_func_list=[]

#init_row=random.choice(list(i for i in range(k))) #case of random data row initiation
#x_init=x[init_row]

x_init=x[0]



UF_initial=mse_joy(y=y, y_pred=x_init) #initial meaning of utility function for later comparison
UFg.append(UF_initial)
print(UFg[-1])

#General loop
x=x_init


for epoch in range(EPOCH):
    
    line=random.choice(func_list) #random choice of function class
    
    line_init=line() #initiation of function
    
    logits=line_init.forward(x) #applying given function
    
    UF=mse_joy(y=y,y_pred=logits) #utility function
    
    
    if UF>=UFg[-1]:
        #print('Oooohhhuuu!')
        UFg.append(UF)
        x=logits #in successful case computations are used for further function applying
        func_applied_list.append(line)
        param_func_list.append(line_init._params())
        
        #preparings for further applying of the same function
        UFtry=[] 
        UFtry.append(-1e30)
        
        UFtry_count=0 #need for skipping very big while loops in case of very little gradual improvements of utility function
        
        #if utility function is still improving
        while UF>=UFg[-1] and UF>UFtry[-1] and UFtry_count<=50:
            
            
            logits=line_init.forward(x)
            
            
            UF=mse_joy(y=y,y_pred=logits)
            
            
            
            if UF>UFg[-1]:
                
                UFtry.append(UFg[-1])
                UFtry_count+=1
                
                UFg.append(UF)
                x=logits
                func_applied_list.append(line)
                param_func_list.append(line_init._params())
                
            else:
                print("Noooo!")
                UFtry.append(1e10)
            
            
#some information during training   
print(UFg[:5], UFg[-5:])
print(func_applied_list[:5], func_applied_list[-5:])
print(param_func_list[:5], param_func_list[-5:])

#Generating method for the model. Could be functionalised

x=residual

x_init=x[0]

x=x_init

#we apply function by function to our data
for i in range(len(func_applied_list)):
    func=func_applied_list[i]
    line=func()
    logits=line.gen_with_params(x,param_func_list[i])
    x=logits

#Logits for the first segment
logits_seg_1=x

print(len(func_applied_list))

#Same steps for the secong segment, but assuming the result of the first one

#List of applied functions
func_applied_list_2=[]
param_func_list_2=[]

print(UFg[-1])

EPOCH_seg_2=10000

x=residual



x_init=x[0]

x=x_init

for epoch in range(EPOCH_seg_2):
    
    line=random.choice(func_list)
    
    line_init=line()
    
    logits_2=line_init.forward(x)
    
    #Sum of two segments for obtaining result of the model
    logits=logits_2*logits_seg_1
    
    UF=mse_joy(y=y,y_pred=logits)
    
    
    if UF>=UFg[-1]:
        #print('Oooohhhuuu!')
        UFg.append(UF)
        x=logits_2
        func_applied_list_2.append(line)
        param_func_list_2.append(line_init._params())
        
        
        UFtry=[]
        UFtry.append(-1e30)
        
        UFtry_count=0
        
        
        while UF>=UFg[-1] and UF>UFtry[-1] and UFtry_count<=20:
            
            
            logits_2=line_init.forward(x)
            
            logits=logits_2*logits_seg_1
            
            UF=mse_joy(y=y,y_pred=logits)
            
            
            
            if UF>UFg[-1]:
                #print('Yeeah')
                UFtry.append(UFg[-1])
                UFtry_count+=1
                
                UFg.append(UF)
                x=logits_2
                func_applied_list_2.append(line)
                param_func_list_2.append(line_init._params())
                
            else:
                print("Noooo!")
                UFtry.append(1e10)
      

           
#some information
print(UFg[:5], UFg[-5:])
print(func_applied_list_2[:5], func_applied_list_2[-5:])
print(param_func_list_2[:5], param_func_list_2[-5:])

#Generating method for the model

x=residual

x_init=x[0]

x=x_init

for i in range(len(func_applied_list_2)):
    func=func_applied_list_2[i]
    line=func()
    logits_2=line.gen_with_params(x,param_func_list_2[i])
    x=logits_2

y_pred=logits_2*logits_seg_1 #output of the model



print(len(func_applied_list))
print(len(func_applied_list_2))

#preparation for drawing
x=residual[0]
y=residual[1]

 
Z=125+23*x-345*y+20*x*y-40*x**2+40*y**2
#Z=15*np.sin(3*x)+5*np.cos(5*y)
#Z=15*np.sin(3*x)+5*np.cos(5*y)+20*x
#Z=15*np.sin(3*x)+5*np.cos(5*y)+20*x-2*y**2


#Some plotting

fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
 
ax.plot3D(x, y, Z, label='true')
ax.plot3D(x, y, y_pred, label='model')
 

ax.set_xlabel('x1', fontsize=12)
ax.set_ylabel('x2', fontsize=12)
ax.set_zlabel('y', fontsize=12)
plt.legend() 
plt.show()

#Plot Utility Function
plt.plot(UFg)
plt.show()