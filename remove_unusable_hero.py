import os
# verify
with open('test_data/hero_names.txt', 'r') as f:
    hero_names = f.readlines()
    hero_names = [name.strip() for name in hero_names]

available_heroes = os.listdir("hero_images")
available_heroes = [name.split('.png')[0] for name in available_heroes]
for name in hero_names:
    if name not in available_heroes:
        print(f"{name} not found!")

with open("test_data/test.txt", 'r') as f:
    lines = f.readlines()
    lines = [x.strip() for x in lines]
ground_truth = list(map(lambda x: x.split('\t'), lines))


hero_names = list(set(hero_names + list(map(lambda x: x[1], ground_truth))))

for name in available_heroes:
    if name not in hero_names:
        print(f"{name} not found!")
        os.system(f"rm hero_images/{name}.png")

len(os.listdir("hero_images"))