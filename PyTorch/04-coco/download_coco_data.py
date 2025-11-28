import urllib.request
import os

def download_with_progress(url, filename):
    """带进度条的下载函数"""
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100.0 / total_size, 100)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        
        # 显示进度条
        bar_length = 50
        filled_length = int(bar_length * downloaded / total_size)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f'\r下载进度: |{bar}| {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)', end='')
    
    print(f'开始下载: {filename}')
    urllib.request.urlretrieve(url, filename, reporthook)
    print(f'\n✓ 完成下载: {filename}\n')

# 创建保存目录
os.makedirs('coco', exist_ok=True)
os.chdir('coco')

# 先下载小文件测试
print('=' * 60)
download_with_progress(
    'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    'annotations_trainval2017.zip'
)

print('=' * 60)
download_with_progress(
    'http://images.cocodataset.org/zips/val2017.zip',
    'val2017.zip'
)

print('\n所有文件下载完成！')
print('如需下载训练集(18GB)，取消下面的注释：')
print('# download_with_progress("http://images.cocodataset.org/zips/train2017.zip", "train2017.zip")')

#----------------------------------------
'''
这个版本会显示：
- 📊 进度条
- 📈 百分比
- 💾 已下载/总大小（MB）

运行后你会看到类似这样的输出：
```
开始下载: annotations_trainval2017.zip
下载进度: |████████████████████████--------------------------| 48.5% (116.2/241.0 MB)
'''
