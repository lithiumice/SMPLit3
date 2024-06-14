import torch 
a=torch.randn(30,1024,1024).to('cuda')

print('start!')

while True:
    a=a@a

print('exited!')
