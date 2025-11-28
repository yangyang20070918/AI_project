#coco数据读取
from pycocotools.coco import COCO
from skimage import io #用于可视化
from matplotlib import pyplot as plt


anno_file = "///DATA/coco/annotations/instances_val2017.json"
coco = COCO(anno_file)

catIds = coco.getCatIds(catNms=['person'])
print(catIds)
imgIds = coco.getImgIds(catIds=catIds)
print(imgIds)

for i in range(len(imgIds)):
    image = coco.loadImgs(imgIds[i])[0]
    I = io.imread(image["coco_url"])
    plt.imshow(I)#可视化
    anno_id = coco.getAnnIds(imgIds=image["id"], catIds=catIds, iscrow=None)
    annotation = coco.loadAnns(anno_id)
    coco.showAnns(annotation)
    plt.show()


















