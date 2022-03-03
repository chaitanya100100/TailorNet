import os
import csv
from datetime import datetime
import TailorNet.global_var


class BaseLogger(object):
    def __init__(self, log_name, fields):
        self.fpath = os.path.join(global_var.LOG_DIR, log_name)
        self.fields = fields

    def add_item(self, **kwargs):
        kwargs = kwargs.copy()
        if 'time' not in kwargs:
            kwargs['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for k in list(kwargs):
            if k not in self.fields:
                kwargs.pop(k)
            else:
                kwargs[k] = str(kwargs[k])
        if os.path.exists(self.fpath):
            with open(self.fpath, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writerow(kwargs)
        else:
            with open(self.fpath, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()
                writer.writerow(kwargs)


class TailorNetLogger(BaseLogger):
    def __init__(self, log_name='tailornet.csv'):
        super(TailorNetLogger, self).__init__(log_name,
            ['garment_class', 'gender', 'smooth_level', 'best_error', 'best_epoch', 'time', 'batch_size',
             'lr', 'weight_decay', 'note', 'log_name', 'shape_style', 'checkpoint'])
