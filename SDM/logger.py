import prettytable as pt
import json
import time

class Logger:
    def __init__(self, dataset, cfg):
        self.dataset = dataset
        cfg_dict = cfg.to_dict()
        self.cfg = json.dumps(cfg_dict, indent=2, ensure_ascii=False)
        print(self.cfg)
        
    def save_table(self, labels, rows):
        tb = pt.PrettyTable()
        tb.field_names = labels
        for row in rows:
            tb.add_row(row)
        current = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        with open('log-'+self.dataset+'-'+current+'.txt','w') as f:
            f.write(self.cfg+'\n')
            f.write(self.dataset+'\n')
            f.write(tb.get_string())    
        print(tb)