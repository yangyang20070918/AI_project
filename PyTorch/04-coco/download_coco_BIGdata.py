import urllib.request
import os
import requests

#-----------------------------------------------------------
#下载中途网络中断需要重新下载
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
    print(f'URL: {url}')
    urllib.request.urlretrieve(url, filename, reporthook)
    print(f'\n✓ 完成下载: {filename}\n')
#------------------------------------------------------------
def download_with_resume(url, filename):
    """支持断点续传的下载函数"""
    
    # 检查本地文件是否存在
    if os.path.exists(filename):
        resume_byte_pos = os.path.getsize(filename)
        print(f'发现未完成的文件，从 {resume_byte_pos/(1024*1024):.1f}MB 处继续下载...')
    else:
        resume_byte_pos = 0
    
    # 设置断点续传的header
    headers = {'Range': f'bytes={resume_byte_pos}-'}
    
    # 发送请求
    response = requests.get(url, headers=headers, stream=True, timeout=30)
    total_size = int(response.headers.get('content-length', 0)) + resume_byte_pos
    
    # 打开文件（追加模式）
    mode = 'ab' if resume_byte_pos else 'wb'
    
    with open(filename, mode) as f:
        downloaded = resume_byte_pos
        
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                # 显示进度
                percent = downloaded * 100.0 / total_size
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                
                bar_length = 50
                filled_length = int(bar_length * downloaded / total_size)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                
                print(f'\r下载进度: |{bar}| {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)', end='')
    
    print(f'\n✓ 完成下载: {filename}')
#---------------------------------------------------------------

# 切换到coco目录
os.chdir('coco')

# 下载训练集（18GB，需要一些时间）
print('=' * 60)
print('⚠️  注意：训练集大小约18GB，下载需要较长时间')
print('=' * 60)
#方法1:下载中途网络中断需要重新下载
##download_with_progress(
##    'http://images.cocodataset.org/zips/train2017.zip',
##    'train2017.zip'
##)
#方法2:支持断点续传的下载
try:
    download_with_resume(
        'http://images.cocodataset.org/zips/train2017.zip',
        'train2017.zip'
    )
except Exception as e:
    print(f'\n下载中断: {e}')
    print('可以重新运行脚本继续下载！')

print('🎉 训练集下载完成！')
#-------------------------------------------------------
# 下载2017测试集（6GB）
#方法1:下载中途网络中断需要重新下载
##download_with_progress(
##    'http://images.cocodataset.org/zips/test2017.zip',
##    'test2017.zip'
##)
#方法2:支持断点续传的下载
try:
    download_with_resume(
        'http://images.cocodataset.org/zips/test2017.zip',
        'test2017.zip'
    )
except Exception as e:
    print(f'\n下载中断: {e}')
    print('可以重新运行脚本继续下载！')

print('🎉 测试集下载完成！')
#-------------------------------------------------------
print('\n下载的文件：')
print('  - annotations_trainval2017.zip (241MB)')
print('  - val2017.zip (1GB)')
print('  - train2017.zip (18GB)')
print('\n接下来可以解压文件了！')
