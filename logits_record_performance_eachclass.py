import argparse
import os
import torch
from tqdm import tqdm
from pathlib import Path
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser
import torch.nn.functional as F

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=128,
        shuffle=False,
        training=False,
        num_workers=8
    )


    save_dir = str(Path(config['trainer']['save_dir']))    
    if not os.path.exists(save_dir+"/expert1_acc.npy"):     
        print("saving performance per class")
        # build model architecture 
        model = config.init_obj('arch', module_arch)
       
        #logger.info(model)
    
        # get function handles of loss and metrics
        #loss_fn = config.init_obj('loss', module_loss)
        #metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    
        logger.info('Loading checkpoint: {} ...'.format(config.resume))
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
    
        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)   
        model.eval()
    
        #total_loss = 0.0
        #total_metrics = torch.zeros(len(metric_fns))
    
        num_classes = config._config["arch"]["args"]["num_classes"]
        expert1_confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
        expert2_confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
        expert3_confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
        ensemble_confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
    
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(data_loader)):
                data, target = data.to(device), target.to(device) 
                output = model(data)
                expert1_logits = output['logits'][:,0,:]
                expert2_logits = output['logits'][:,1,:]
                expert3_logits = output['logits'][:,2,:]
                ensemble_logit = output['output'] 
                # softmax_logits = F.softmax(logits[:,0,:], dim=1)
                #expert1_logits.append(logits[:,0,:])
                #expert2_logits.append(logits[:,1,:])
                #final_logits.append(final_logit)
                for t, p in zip(target.view(-1), expert1_logits.argmax(dim=1).view(-1)):
                    expert1_confusion_matrix[t.long(), p.long()] += 1
                for t, p in zip(target.view(-1), expert2_logits.argmax(dim=1).view(-1)):
                    expert2_confusion_matrix[t.long(), p.long()] += 1
                for t, p in zip(target.view(-1), expert3_logits.argmax(dim=1).view(-1)):
                    expert3_confusion_matrix[t.long(), p.long()] += 1
                for t, p in zip(target.view(-1), ensemble_logit.argmax(dim=1).view(-1)):
                    ensemble_confusion_matrix[t.long(), p.long()] += 1

        expert1_acc_per_class = expert1_confusion_matrix.diag()/expert1_confusion_matrix.sum(1)
        expert2_acc_per_class = expert2_confusion_matrix.diag()/expert2_confusion_matrix.sum(1)
        expert3_acc_per_class = expert3_confusion_matrix.diag()/expert3_confusion_matrix.sum(1)
        ensemble_acc_per_class = ensemble_confusion_matrix.diag()/ensemble_confusion_matrix.sum(1)



        expert1_acc = expert1_acc_per_class.cpu().numpy() * 100
        expert2_acc = expert2_acc_per_class.cpu().numpy() * 100
        expert3_acc = expert3_acc_per_class.cpu().numpy() * 100
        ensemble_acc = ensemble_acc_per_class.cpu().numpy()  * 100
                     
        with open(save_dir+"/expert1_acc.npy","wb") as f:
            np.save(f, expert1_acc)
        f.close() 
     
        with open(save_dir+"/expert2_acc.npy","wb") as f:
            np.save(f, expert2_acc)
        f.close() 
        with open(save_dir+"/expert3_acc.npy","wb") as f:
            np.save(f, expert3_acc)
        f.close()
     
        with open(save_dir+"/ensemble_acc.npy","wb") as f:
            np.save(f, ensemble_acc)
        f.close() 
        
         
        
        
    else:
        print('loading performance per class')    
        b = np.load("../data/shot_list.npy")
        many_shot = b[0]
        medium_shot = b[1] 
        few_shot = b[2]
     
        many_shot_indicator = np.expand_dims(many_shot, axis=1)
        medium_shot_indicator = np.expand_dims(medium_shot, axis=1)
        few_shot_indicator = np.expand_dims(few_shot, axis=1)
    
        get_class_acc = True
        #if get_class_acc:
        #    test_cls_num_list = np.array(data_loader.cls_num_list)
        
        
        expert1_acc = np.load(save_dir+"/expert1_acc.npy") 
        expert2_acc = np.load(save_dir+"/expert2_acc.npy") 
        expert3_acc = np.load(save_dir+"/expert3_acc.npy") 
        ensemble_acc = np.load(save_dir+"/ensemble_acc.npy") 
  
        expert1_acc = np.expand_dims(np.round(expert1_acc, decimals=4),axis=1)
        expert2_acc = np.expand_dims(np.round(expert2_acc, decimals=4),axis=1)
        expert3_acc = np.expand_dims(np.round(expert3_acc, decimals=4),axis=1)
        ensemble_acc= np.expand_dims(np.round(ensemble_acc, decimals=4),axis=1)
    
    
        """
         
        with open(save_dir+"/expert1_acc.txt","wb") as f:
            np.savetxt(f, , fmt='%.02f')
        f.close() 
     
        with open(save_dir+"/expert2_acc.txt","wb") as f:
            np.savetxt(f, np.round(expert2_acc, decimals=2), fmt='%.02f')
        f.close() 
        with open(save_dir+"/expert3_acc.txt","wb") as f:
            np.savetxt(f, np.round(expert3_acc, decimals=2), fmt='%.02f')
        f.close()
     
        with open(save_dir+"/ensemble_acc.txt","wb") as f:
            np.savetxt(f, np.round(ensemble_acc, decimals=2), fmt='%.02f')
        f.close() 
        """
    
        #print("expert1_acc_per_class.shape",expert1_acc.shape)
        #expert1_acc_per_class1 = np.expand_dims(expert1_acc,  axis=1)  
        #expert2_acc_per_class1 = np.expand_dims(expert2_acc, axis=1)  
        #expert3_acc_per_class1 = np.expand_dims(expert3_acc, axis=1)  
        #ensemble_acc_per_class1 = np.expand_dims(ensemble_acc, axis=1)  
        print("expert1_acc_per_class1.shape",expert1_acc.shape)       
                                       
        expert_concat_per_class = np.concatenate((expert1_acc,expert2_acc,expert3_acc), axis=1)  
        std_acc_per_class = np.std(expert_concat_per_class, axis=1)
        std_acc_per_class = np.expand_dims(np.round(std_acc_per_class, decimals=4),axis=1)
        
        
        idx = np.expand_dims(np.arange(1000),axis=1)
        #title_list = ["idx",  "many_shot",  "medium_shot",  "few_shot", "expert1", "expert2", "expert3", "ensemble", "std of each expert"]
        # title_list = "idx expert1 expert2 expert3 ensemble std"
        title_list = "idx many medium few expert1 expert2 expert3 ensemble std"
        
        performance_table = np.concatenate((idx, many_shot_indicator, medium_shot_indicator, few_shot_indicator, expert1_acc, expert2_acc, expert3_acc, ensemble_acc, std_acc_per_class), axis = 1)
        
        #performance_table = np.concatenate((idx, expert1_acc, expert2_acc, expert3_acc, ensemble_acc, std_acc_per_class), axis = 1)
        
        with open(save_dir+"/performance_table_shot.txt","wb") as f:
            #for i in title_list:
            #    f.write(i)
                
            #f.write("\n")
            np.savetxt(f,  performance_table,  fmt='%.00f', header = title_list)
        f.close()         
        
    
    
     
    
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)

