import logging
import numpy as np


def set_logger(log_path):
    LOG_LEVEL = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }
    LOG_FORNAT = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'


    logger = logging.getLogger(log_path)

    format_str = logging.Formatter(LOG_FORNAT)
    logger.setLevel(LOG_LEVEL.get('info'))  # 设置日志级别

    # 屏幕输出
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)

    # 文件输出
    th = logging.handlers.TimedRotatingFileHandler(
        filename=log_path, when='D', backupCount=3, encoding='utf-8')
    th.setFormatter(format_str)

    logger.addHandler(sh)
    logger.addHandler(th)

    return logger


def softmax(arr):
    maximum = np.max(arr)
    exps = np.exp(arr - maximum)
    return  exps / np.sum(exps)
