import logging
import os

class TableLogger:
    def __init__(self, log_dir='output', log_file='tables.log'):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        self.tables = []
        logging.basicConfig(filename=self.log_path, level=logging.INFO, format='%(message)s')

    def log_table(self, table, title=None):
        if title:
            self.tables.append(f"\n===== {title} =====\n{table}")
        else:
            self.tables.append(str(table))

    def save(self):
        with open(self.log_path, 'w', encoding='utf-8') as f:
            for table in self.tables:
                f.write(table + '\n')
        logging.info('All tables saved to %s', self.log_path)
