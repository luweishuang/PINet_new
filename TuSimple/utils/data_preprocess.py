import os

base_dir = "../../dataset/powerline"
imgs_dir = os.path.join(base_dir, "imgs")
imgs_dir_new = os.path.join(base_dir, "imgs_new")
labels_dir = os.path.join(base_dir, "labels")
labels_dir_new = os.path.join(base_dir, "labels_new")
os.makedirs(imgs_dir_new, exist_ok=True)
os.makedirs(labels_dir_new, exist_ok=True)

rename_file = os.path.join(base_dir, "powertower_rename.txt")
rename_map = {}
with open(rename_file, "r") as fr:
    for cur_l in fr:
        src_img, new_name = cur_l.strip().split(" ==> ")
        src_name = os.path.basename(src_img)
        if src_name not in rename_map:
            rename_map[src_name] = new_name

for cur_f in os.listdir(imgs_dir):
    if cur_f in rename_map:
        new_name = rename_map[cur_f]
        cur_img = os.path.join(imgs_dir, cur_f)
        new_img = os.path.join(imgs_dir_new, new_name)
        cur_label = os.path.join(labels_dir, cur_f.replace(".jpg", ".png"))
        new_label = os.path.join(labels_dir_new, new_name.replace(".jpg", ".png"))
        os.system("cp %s %s" % (cur_img, new_img))
        os.system("cp %s %s" % (cur_label, new_label))
    else:
        print(cur_f, "not in rename_map")