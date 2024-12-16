import torch
import torch.utils
import torch.utils.data
from dataloader import *
import numpy as np
import tqdm
from data import load_data
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from dataset.semi import SemiDataset
def unimatch(model:torch.nn.Module,optimizer:torch.optim.Optimizer,thresh:float|int,
             train_loader_l,train_loader_u,criterion = None):
    #有标记的 弱干扰的 强干扰的
    if not criterion:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    criterion_u = torch.nn.CrossEntropyLoss(reduction='none')

   
    data_loader = zip(train_loader_l,train_loader_u,train_loader_u)
    for i,((img_l,lbl),(w_img,s_img1,s_img2,ignore_mask_u),(w_img2,s_img2_1,s_img2_2,_)) in enumerate(data_loader):
            #转移到cuda上
            img_l,lbl = img_l.to('cuda:0'),lbl.to('cuda:0')
            w_img,s_img1,s_img2,ignore_mask_u = w_img.to('cuda:0'),s_img1.to('cuda:0'),s_img2.to('cuda:0'),ignore_mask_u.to('cuda:0')

            ###github上的代码这部分只是为了在预测模式下生成对应的mask 然后进行cutmix，和论文没有什么关系
            with torch.no_grad():
                model.eval()
                lbl_u_pred = model(w_img.to(torch.float)).detach()
                mask_u_pred = lbl_u_pred.argmax(dim=1)
                conf_u_pred = lbl_u_pred.softmax(dim=1).max(dim=1)[0]
            
            model.train()
            num_lb = img_l.shape[0]
            num_ulb = w_img.shape[0]
            pred_s1 = model(s_img1.to(torch.float))
            pred_s2 = model(s_img2.to(torch.float))
            #显然这个True是用来产生fp预测值的
            preds, preds_fp = model((torch.cat((img_l, w_img))).to(torch.float), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            pred_u_w = pred_u_w.detach()
            mask_uw = pred_u_w.argmax(dim=1)
            conf_uw = pred_u_w.softmax(dim=1).max(dim=1)[0]

            ##损失计算
            #

            loss_l = criterion(pred_x,lbl)

            loss_fp = criterion_u(pred_u_w_fp,mask_uw)
            loss_fp = loss_fp*((conf_uw>thresh)&(ignore_mask_u!=255))
            loss_fp = loss_fp.sum()/(ignore_mask_u!=255).sum().item()
            
            loss_s1 = criterion_u(pred_s1,mask_uw)
            loss_s1 = loss_s1*((conf_uw>thresh)&(ignore_mask_u!=255))
            loss_s1 = loss_s1.sum()/(ignore_mask_u!=255).sum().item()
            
            loss_s2 = criterion_u(pred_s2,mask_uw)
            loss_s2 = loss_s2*((conf_uw>thresh)&(ignore_mask_u!=255))
            loss_s2 = loss_s2.sum()/(ignore_mask_u!=255).sum().item()
            
            #loss = loss_l
            loss = (loss_l + loss_fp*0.5 + loss_s1*0.25+loss_s2*0.25)/2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iters = epoch * len(train_loader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
if __name__ == '__main__':
    import yaml
    with open('config.yaml','r') as f:
        cfg = yaml.load(f,yaml.Loader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DeepLabV3Plus(cfg)
    model.cuda(device)
    optimizer = torch.optim.SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'])
    epochs = 80
    n_cls = 21
    thresh = 0.00
    avgloss = []
    imgp,lblp = load_data('/home/featurize/data/VOCdevkit/VOC2012',True)
    imgup,lblup = load_data('/home/featurize/data/VOCdevkit/VOC2012',False)
    
    
    train_l_data = SemiDataset('pascal','','train_l',224,imgp,lblp)
    train_u_data = SemiDataset('pascal','','train_u',224,imgup,lblup)
    val_data = SemiDataset('pascal','','val',224,imgup,lblup)
    train_loader_l = torch.utils.data.DataLoader(train_l_data,16,True,drop_last=True)
    train_loader_u = torch.utils.data.DataLoader(train_u_data,16,True,drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data,1,True,drop_last=True)

    total_iters = len(train_loader_u) * cfg['epochs']
    
    for epoch in range(epochs):
        print(f'epoch {epoch} started')
        unimatch(model,optimizer,thresh,train_loader_l,train_loader_u)
        
        print(f'epoch {epoch} finished')
        if epoch%5 == 0:
            miou,ioucls = evaluate(model,val_loader,'original',cfg)
            print(f'epoch {epoch} miou ==> {miou} ioucls {ioucls}')







