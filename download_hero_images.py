import os
from tqdm import tqdm

hero_image_path = "hero_images"
hero_name_link = "hero_images.txt"

if __name__ == "__main__":
    # create directory
    os.makedirs(hero_image_path, exist_ok=True)

    with open(hero_name_link, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    for line in tqdm(lines):
        filename = line.split("/")[-1].strip()
        hero_name = filename.split("_OriginalSquare_WR")[0].strip()
        if "%27" in hero_name:
            hero_name = hero_name.replace("%27", "")

        path = f"{hero_image_path}/{hero_name}.png"
        if os.path.exists(path):
            continue

        cmd = f"curl {line} -o {path}" 
        os.system(cmd)

    print("Download Done!")

    # verify
    with open('test_data/hero_names.txt', 'r') as f:
        hero_names = f.readlines()
        hero_names = [name.strip() for name in hero_names]

    available_heroes = os.listdir(hero_image_path)
    available_heroes = [name.split('.png')[0] for name in available_heroes]
    for name in hero_names:
        if name not in available_heroes:
            print(f"{name} not found!")
