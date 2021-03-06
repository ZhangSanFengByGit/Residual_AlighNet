import os
import time

import torch
from PIL import Image
from torch.optim import Adam

from torchvision import transforms

from AlignNet import AlignNet

import ssl
ssl._create_default_https_context = ssl._create_unverified_context



#arguments definition
#device = torch.device("cuda" if torch.cuda.is_available == True else "cpu")
device = torch.device("cuda")
base_path = "/home/zhangzichun/pedestrian/images/"
log_interval = 1
checkpoint_interval = 2
learning_rate = 0.001
#warmUp_time = 500
#warmUp_lr = learning_rate/warmUp_time
checkpoint_model_dir = "/home/zhangzichun/model_checkPoint_save_test/"
save_model_dir = "/home/zhangzichun/model_finish_save/"
epochs = 1
feature_weight = 1e5
image_weight = 1e5
batch_size = 4


#transform method
transform = transforms.Compose([
        transforms.Resize([828,621]),
        transforms.RandomCrop([768,576]),
        transforms.RandomRotation([-5,5]),
        #transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor()
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform2 = transforms.Compose([
        transforms.Resize([768,576]),
        transforms.ToTensor()
        ])

#variable definition
alignNet = AlignNet().to(device)
#optimizer = Adam(alignNet.parameters(),learning_rate)
#mse_loss = torch.nn.MSELoss()
to_img = transforms.ToPILImage()


imgR = list()
imgT = list()
imgT_trans = list()
imgR_batch = torch.zeros(batch_size,3,768,576)
imgT_batch = torch.zeros(batch_size,3,768,576)
imgT_trans_batch = torch.zeros(batch_size,3,768,576)

flag = 0
global log_check
global log_loss_sum
global e
global imgNo
global results

#best_loss = infinite_great

#temp Area~~~~~~~~
#alignNet.load_state_dict(torch.load('H:\\model_checkPoint_save_test\\ckpt_epoch_0_Sequence_id_1.plk'))




for e in range(epochs):
    alignNet.train()
    turn = 0

    SetNum = len(os.listdir(base_path))
    for i in range(1):#SetNum
        SequenceNum = len(os.listdir(base_path+"set"+str(i+1)+"/"))
        
        for j in range(1):#SequenceNum
            Sequence_path = base_path+"set"+str(i+1)+"/"+"V005/" #+str(j)+"/"
            
            rgb_folder = Sequence_path + "visible/"
            thermal_folder = Sequence_path + "lwir/"
            imgs = os.listdir(rgb_folder)
            imgNum = len(imgs) 
            
            
            flag = 0
            imgR = []
            imgT = []
            imgT_trans = []
            
            log_check = 0
            log_loss_sum = 0
            
            for imgNo in range(4):
                flag = flag + 1
                
                encodeNo = list(str(imgNo+100000))
                encodeNo.pop(0)
                encodeNo = "".join(encodeNo)
                
                img1 = Image.open(rgb_folder+"I"+encodeNo+".jpg")
                img2 = Image.open(thermal_folder+"I"+encodeNo+".jpg")
                img1 = transform2(img1)#.unsqueeze(0)
                img2_trans = transform(img2)#.unsqueeze(0)
                img2 = transform2(img2)#.unsqueeze(0).to(device)
                
                imgR.append(img1)
                imgT_trans.append(img2_trans)
                imgT.append(img2)
                
                if flag == batch_size:
                    
                    flag = 0
                    
                    for p in range (batch_size):
                        imgR_batch[p] = imgR[p]
                        imgT_batch[p] = imgT[p]
                        imgT_trans_batch[p] = imgT_trans[p]
                        
                    optimizer.zero_grad()
                    
                    results = alignNet(imgT_batch.to(device) , imgT_trans_batch.to(device)

                    img_test = to_img(results[0])
                    img_test.save('/home/zhangzichun/test_result')
                    #feature_loss = mse_loss(results.T2_features, results.R_features)/batch_size
                    #image_loss = mse_loss(results, imgT_batch.to(device))/batch_size
                    
                    #loss = image_loss*image_weight
                    #log_loss = loss.item()
                    #loss.backward()
                    #optimizer.step()

                    
                    #print('Epoch %d loss: %f' %(e,log_loss))
                    #log_check = log_check + 1
                    #log_loss_sum = log_loss_sum + log_loss
                    
                    #imgR = []
                    #imgT = []
                    #imgT_trans = []
            
            '''
#            turn = turn + 1
            log_loss_sum = log_loss_sum/log_check
#            if e % log_interval == 0:
            mesg = "{}\tEpoch {}  Turn{}:  set {}  sequence {}  loss: {:.6f}".format(
                      time.ctime(), e + 1, turn, i, j, log_loss_sum)
            print(mesg)


            if checkpoint_model_dir is not None and e % checkpoint_interval == 0:
                alignNet.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_Sequence_id_" + str(turn) + ".plk"
                ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
                torch.save(alignNet.state_dict(), ckpt_model_path)
                alignNet.to(device).train()

alignNet.eval().cpu()
save_model_filename = "Finish_epoch_" + str(epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" +".model"
save_model_path = os.path.join(save_model_dir, save_model_filename)
torch.save(alignNet.state_dict(), save_model_path)

print("\nDone, trained model saved at", save_model_path)'''
    








'''
use torch.CUDA
optmizer = optim(AlignNet.parameters())
current_params = AlignNet.static_params()
best_loss = infinite_great


for epoch in range(25):
    for set in(6):
        for serials in(50):
            for i in (imageNum):
                img1 = get(rgb)
                img2 = get(thermal)
                img2_t = transform(img2)
                img2_masked = AlignNet(img1,img2_t)
                loss = MSE(img2_masked,img2_t)
                loss.backward()
                optmizer.step()

                if loss < best_loss:
                    best_loss = loss
                    current_params = AlignNet.static_params()

save(current_params)
'''




