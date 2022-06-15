import re

keywords = {
    "plaquette",
    "stylo",
    "tube",
    "seringue",
    "cachet",
    "gelule",
    "flacon",
    "ampoule",
    "ml",
    "g",
    "pilulier",
    "comprime",
    "film",
    "poche",
    "capsule",
}
dict_prod = {keyword: 0 for keyword in keywords}

description = (
    "seringue en aluminium avec 2 cachets d'aspirine de 10 ml et 3 gelules de 0.8 g"
)


def edit_dict(description: str, dict=dict_prod):
    pattern = "\d+\.*\d* [a-z]+"
    for prod in re.findall(pattern, description):
        prod_count = re.search("\d+\.*\d*", prod).group()
        prod_nature = re.search("[a-z]+", prod).group()
        dict_prod[prod_nature] = prod_count


edit_dict(description)
print(dict_prod)
